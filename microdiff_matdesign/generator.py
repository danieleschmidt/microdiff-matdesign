"""
MicrostructureGenerator: synthetic 2D microstructure images via Voronoi tessellation.

Generates grain boundary maps and phase distribution images that resemble
real polycrystalline material cross-sections observed under electron microscopy.
"""

import numpy as np
from scipy.spatial import Voronoi


class MicrostructureGenerator:
    """
    Generates synthetic 2D microstructure images using Voronoi tessellation.

    Each image represents a 2D cross-section of a polycrystalline material,
    where Voronoi cells correspond to individual grains. Grain boundaries
    and phase distributions are encoded in grayscale.

    Parameters
    ----------
    image_size : int
        Side length of the square output image in pixels.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, image_size: int = 64, seed: int | None = None):
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_grains: int = 20,
        noise_sigma: float = 0.02,
        phase_fraction: float | None = None,
    ) -> np.ndarray:
        """
        Generate a single microstructure image.

        Parameters
        ----------
        n_grains : int
            Number of Voronoi seed points (grains). More grains → finer grain size.
        noise_sigma : float
            Standard deviation of Gaussian noise added to simulate imaging artifacts.
        phase_fraction : float or None
            If provided (0–1), fraction of grains assigned to a secondary phase
            (represented by a different intensity level).

        Returns
        -------
        np.ndarray
            Float32 array of shape (1, image_size, image_size) with values in [0, 1].
            0 = grain interior (phase 1), ~0.5 = grain boundary, ~1.0 = phase 2.
        """
        size = self.image_size

        # Random seed points inside the image (with border padding to avoid edge effects)
        pad = 0.1
        pts = self.rng.uniform(pad, 1 - pad, size=(n_grains, 2)) * size

        # Build Voronoi diagram; add mirror points to handle boundary cells
        mirror = self._mirror_points(pts, size)
        all_pts = np.vstack([pts, mirror])
        vor = Voronoi(all_pts)

        # Assign each pixel to its nearest seed = grain index
        ys, xs = np.mgrid[0:size, 0:size]
        pixel_coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
        # Simple nearest-neighbor assignment
        diffs = pixel_coords[:, None, :] - pts[None, :, :]  # (N_px, n_grains, 2)
        dist2 = (diffs ** 2).sum(axis=2)
        grain_map = dist2.argmin(axis=1).reshape(size, size)

        # Optional: assign secondary phase to some grains
        grain_phase = np.zeros(n_grains, dtype=np.float32)  # 0 = primary
        if phase_fraction is not None and phase_fraction > 0:
            n_secondary = max(1, int(n_grains * phase_fraction))
            secondary_idx = self.rng.choice(n_grains, size=n_secondary, replace=False)
            grain_phase[secondary_idx] = 1.0

        # Base image from grain phases
        image = grain_phase[grain_map].astype(np.float32)

        # Detect grain boundaries: pixels where neighbor has different grain
        boundary = self._compute_boundary(grain_map)
        image[boundary] = 0.5  # grain boundaries at mid-intensity

        # Add mild Gaussian noise for realism
        if noise_sigma > 0:
            noise = self.rng.normal(0, noise_sigma, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return image[np.newaxis, :, :]  # (1, H, W)

    def generate_batch(
        self,
        n: int,
        n_grains_range: tuple[int, int] = (10, 50),
        phase_fraction_range: tuple[float, float] = (0.0, 0.4),
    ) -> np.ndarray:
        """
        Generate a batch of microstructure images with randomized parameters.

        Parameters
        ----------
        n : int
            Number of images to generate.
        n_grains_range : (int, int)
            Min/max number of grains to sample uniformly.
        phase_fraction_range : (float, float)
            Min/max secondary phase fraction.

        Returns
        -------
        np.ndarray
            Float32 array of shape (n, 1, image_size, image_size).
        """
        images = []
        for _ in range(n):
            n_grains = int(self.rng.integers(n_grains_range[0], n_grains_range[1] + 1))
            pf = float(self.rng.uniform(*phase_fraction_range))
            images.append(self.generate(n_grains=n_grains, phase_fraction=pf))
        return np.stack(images, axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mirror_points(self, pts: np.ndarray, size: float) -> np.ndarray:
        """Create mirror points outside the image to handle Voronoi boundary cells."""
        mirrors = []
        for dx, dy in [(-size, 0), (size, 0), (0, -size), (0, size)]:
            mirrors.append(pts + np.array([dx, dy]))
        return np.vstack(mirrors)

    def _compute_boundary(self, grain_map: np.ndarray) -> np.ndarray:
        """Return boolean mask of boundary pixels (4-connected neighbors differ)."""
        diff_h = grain_map[:-1, :] != grain_map[1:, :]
        diff_v = grain_map[:, :-1] != grain_map[:, 1:]
        boundary = np.zeros_like(grain_map, dtype=bool)
        boundary[:-1, :] |= diff_h
        boundary[1:, :] |= diff_h
        boundary[:, :-1] |= diff_v
        boundary[:, 1:] |= diff_v
        return boundary
