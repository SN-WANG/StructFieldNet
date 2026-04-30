"""Point-cloud visualization for structural fields."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import torch

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "structfieldnet_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
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

    @staticmethod
    def _scalar_bar_args(title: str) -> dict[str, object]:
        """Return scalar-bar settings that keep labels inside each panel."""
        return {
            "title": title,
            "vertical": True,
            "position_x": 0.80,
            "position_y": 0.08,
            "width": 0.035,
            "height": 0.42,
            "title_font_size": 12,
            "label_font_size": 11,
            "fmt": "%.2e",
            "n_labels": 5,
        }

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
            scalar_bar_args=self._scalar_bar_args(title),
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

    @staticmethod
    def save_comparison_movie(
        frame_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        fps: float = 2.0,
    ) -> Path:
        """
        Save an MP4 animation from all rendered comparison figures.

        Args:
            frame_paths: Ordered comparison image paths.
            output_path: Output MP4 path.
            fps: Video frame rate.

        Returns:
            Saved MP4 path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_paths = [Path(path) for path in frame_paths]

        with tempfile.TemporaryDirectory(prefix="structfieldnet_movie_") as temp_name:
            temp_dir = Path(temp_name)
            loop_paths = frame_paths + frame_paths[:1]
            for frame_idx, frame_path in enumerate(loop_paths):
                shutil.copyfile(frame_path, temp_dir / f"frame_{frame_idx:05d}.png")

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(temp_dir / "frame_%05d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            subprocess.run(command, check=True)

        return output_path
