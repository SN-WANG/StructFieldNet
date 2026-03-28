"""Point-cloud visualization for structural fields."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - explicit dependency
    pv = None


class FieldVis:
    """Render ground truth, prediction, and absolute error as point clouds."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        off_screen: bool = True,
        window_size: Tuple[int, int] = (2200, 760),
        point_size: float = 7.0,
        screenshot_scale: int = 1,
        theme: str = "document",
    ) -> None:
        if pv is None:
            raise ImportError("PyVista is required for field visualization.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.off_screen = off_screen
        self.window_size = window_size
        self.point_size = point_size
        self.screenshot_scale = screenshot_scale
        pv.set_plot_theme(theme)

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert a tensor or array to a float32 NumPy array."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def _prepare_points(self, coords: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Validate and return nodal coordinates."""
        points = self._to_numpy(coords)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"coords must have shape (num_nodes, 3), got {points.shape}")
        return points

    @staticmethod
    def _percentile_clim(values: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> Tuple[float, float]:
        """Compute a stable percentile-based color range."""
        low = float(np.percentile(values, q_low))
        high = float(np.percentile(values, q_high))
        if abs(high - low) < 1e-9:
            center = 0.5 * (low + high)
            low, high = center - 1e-6, center + 1e-6
        return low, high

    def _add_panel(
        self,
        plotter: "pv.Plotter",
        subplot_idx: int,
        points: np.ndarray,
        scalars: np.ndarray,
        title: str,
        cmap: str,
        clim: Tuple[float, float],
    ) -> None:
        """Add one point-cloud panel to the figure."""
        plotter.subplot(0, subplot_idx)
        cloud = pv.PolyData(points)
        cloud["field"] = scalars.astype(np.float32)
        plotter.add_mesh(
            cloud,
            scalars="field",
            cmap=cmap,
            clim=clim,
            render_points_as_spheres=True,
            point_size=self.point_size,
            scalar_bar_args={"title": title, "vertical": True},
        )
        plotter.add_text(title, font_size=16)
        plotter.view_isometric()
        plotter.camera.zoom(1.18)

    def compare_fields(
        self,
        gt: Union[np.ndarray, torch.Tensor],
        pred: Union[np.ndarray, torch.Tensor],
        coords: Union[np.ndarray, torch.Tensor],
        case_name: str,
    ) -> Path:
        """Render a three-panel comparison figure.

        Args:
            gt: Ground-truth field with shape (num_nodes, 1) or (num_nodes,).
            pred: Predicted field with the same shape as gt.
            coords: Coordinate array with shape (num_nodes, 3).
            case_name: Case identifier used in the output filename.

        Returns:
            Saved figure path.
        """
        points = self._prepare_points(coords)
        gt_array = self._to_numpy(gt).reshape(-1)
        pred_array = self._to_numpy(pred).reshape(-1)
        error_array = np.abs(pred_array - gt_array)

        shared_clim = self._percentile_clim(np.concatenate([gt_array, pred_array], axis=0))
        error_clim = self._percentile_clim(error_array, q_low=0.0, q_high=98.0)

        output_path = self.output_dir / f"{case_name}_comparison.png"
        plotter = pv.Plotter(shape=(1, 3), off_screen=self.off_screen, window_size=self.window_size)

        self._add_panel(plotter, 0, points, gt_array, "Ground Truth", "viridis", shared_clim)
        self._add_panel(plotter, 1, points, pred_array, "Prediction", "viridis", shared_clim)
        self._add_panel(plotter, 2, points, error_array, "Absolute Error", "Reds", error_clim)

        plotter.link_views()
        plotter.screenshot(str(output_path), scale=self.screenshot_scale)
        plotter.close()
        return output_path
