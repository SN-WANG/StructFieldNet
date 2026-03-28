# Static field visualization with PyVista
# Author: Shengning Wang

import os
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
import torch

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - pyvista is an explicit dependency
    pv = None


class FieldVis:
    """Render ground-truth, prediction, and error side by side."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        mesh_mode: Literal["auto", "delaunay", "point_cloud"] = "auto",
        off_screen: bool = True,
        window_size: Tuple[int, int] = (2400, 800),
        point_size: float = 8.0,
        screenshot_scale: int = 1,
        theme: str = "document",
    ) -> None:
        if pv is None:
            raise ImportError("PyVista is required for field visualization.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_mode = mesh_mode
        self.off_screen = off_screen
        self.window_size = window_size
        self.point_size = point_size
        self.screenshot_scale = screenshot_scale
        pv.set_plot_theme(theme)

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def _prepare_points(self, coords: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        points = self._to_numpy(coords)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {points.shape}")
        return points

    def _build_projected_delaunay_mesh(self, points: np.ndarray) -> pv.PolyData:
        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        projected = centered @ basis
        planar_points = np.column_stack(
            [projected[:, 0], projected[:, 1], np.zeros(points.shape[0], dtype=np.float32)]
        ).astype(np.float32)

        nn_distances = cKDTree(projected).query(projected, k=2)[0][:, 1]
        alpha = max(float(np.mean(nn_distances)) * 2.5, 1e-6)

        planar_mesh = pv.PolyData(planar_points).delaunay_2d(alpha=alpha)
        if planar_mesh.n_cells == 0 or planar_mesh.faces.size == 0:
            raise RuntimeError("Delaunay triangulation produced no cells.")

        mesh = pv.PolyData(points, faces=planar_mesh.faces)
        return mesh

    def _build_mesh(self, points: np.ndarray) -> Tuple[pv.PolyData, str]:
        if self.mesh_mode == "point_cloud":
            return pv.PolyData(points), "point_cloud"

        if self.mesh_mode in {"auto", "delaunay"}:
            try:
                return self._build_projected_delaunay_mesh(points), "delaunay"
            except Exception:
                if self.mesh_mode == "delaunay":
                    raise

        return pv.PolyData(points), "point_cloud"

    @staticmethod
    def _percentile_clim(values: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> Tuple[float, float]:
        low = float(np.percentile(values, q_low))
        high = float(np.percentile(values, q_high))
        if abs(high - low) < 1e-9:
            center = 0.5 * (low + high)
            low, high = center - 1e-6, center + 1e-6
        return low, high

    def compare_fields(
        self,
        gt: Union[np.ndarray, torch.Tensor],
        pred: Union[np.ndarray, torch.Tensor],
        coords: Union[np.ndarray, torch.Tensor],
        case_name: str,
        field_name: str = "Stress",
    ) -> Path:
        """Render a three-panel comparison figure."""
        points = self._prepare_points(coords)
        gt_array = self._to_numpy(gt).reshape(-1)
        pred_array = self._to_numpy(pred).reshape(-1)
        error_array = np.abs(pred_array - gt_array)

        mesh, render_mode = self._build_mesh(points)
        shared_clim = self._percentile_clim(np.concatenate([gt_array, pred_array], axis=0))
        error_clim = self._percentile_clim(error_array, q_low=0.0, q_high=98.0)

        output_path = self.output_dir / f"{case_name}_comparison.png"
        plotter = pv.Plotter(shape=(1, 3), off_screen=self.off_screen, window_size=self.window_size)

        panels = [
            ("Ground Truth", gt_array, "viridis", shared_clim),
            ("Prediction", pred_array, "viridis", shared_clim),
            ("Absolute Error", error_array, "Reds", error_clim),
        ]

        is_surface = mesh.n_cells > 0 and mesh.faces.size > 0

        for subplot_idx, (title, scalars, cmap, clim) in enumerate(panels):
            plotter.subplot(0, subplot_idx)
            mesh_copy = mesh.copy(deep=True)
            mesh_copy[field_name] = scalars.astype(np.float32)

            if is_surface:
                plotter.add_mesh(
                    mesh_copy,
                    scalars=field_name,
                    cmap=cmap,
                    clim=clim,
                    smooth_shading=True,
                    show_edges=False,
                    scalar_bar_args={"title": title, "vertical": True},
                )
            else:
                plotter.add_mesh(
                    mesh_copy,
                    scalars=field_name,
                    cmap=cmap,
                    clim=clim,
                    render_points_as_spheres=True,
                    point_size=self.point_size,
                    scalar_bar_args={"title": title, "vertical": True},
                )

            plotter.add_text(f"{title}\n({render_mode})", font_size=16)
            plotter.view_isometric()
            plotter.camera.zoom(1.15)

        plotter.link_views()
        plotter.screenshot(str(output_path), scale=self.screenshot_scale)
        plotter.close()
        return output_path
