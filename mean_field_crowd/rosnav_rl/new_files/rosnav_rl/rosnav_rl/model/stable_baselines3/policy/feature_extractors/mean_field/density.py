"""Robot-frame pedestrian density on a coarse grid.

The grid uses the same projection as ``BaseFeatureMapSpace``: ``roi_m`` is the
full width of the map, the robot sits on cell ``grid_size // 2``, and a
coordinate maps to

    index = position * (grid_size / roi_m) + grid_size // 2

AGENT_5's 80×80 map with ``roi_in_m=20`` therefore covers ±10 m. A 16×16
density with the same ``roi_m`` covers that same window, so an upsample lines
up with the pedestrian channels. A pedestrian is counted only when its cell
would land inside the map; the positive edge (exactly +roi_m/2) is outside,
matching ``_get_map_indices_vectorized``.
"""

from __future__ import annotations

import numpy as np

def rasterize_density(
    pedestrian_positions: np.ndarray,
    grid_size: int = 16,
    roi_m: float = 20.0,
    sigma: float = 0.5,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize pedestrian positions into a density with a Gaussian footprint.

    Args:
        pedestrian_positions: Robot-frame ``(x, y)`` in meters, shape ``(N, 2)``
            or ``(B, N, 2)``. ``N`` may be 0.
        grid_size: Cells along each side.
        roi_m: Full width of the window in meters, not the radius.
        sigma: Gaussian standard deviation in meters.
        valid: Optional mask of shape ``(N,)`` or ``(B, N)``. False entries are
            ignored, which is how a padded batch drops empty slots.

    Returns:
        Density of shape ``(B, 1, grid_size, grid_size)``, float32. Each counted
        pedestrian adds a discrete Gaussian whose cell values sum to about 1
        when the kernel lies inside the window.
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if roi_m <= 0:
        raise ValueError(f"roi_m must be positive, got {roi_m}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    positions = np.asarray(pedestrian_positions, dtype=np.float64)
    # PedestrianRelativeLocationGenerator returns np.array([]) when nobody is present.
    if positions.ndim == 1 and positions.size == 0:
        positions = np.zeros((1, 0, 2), dtype=np.float64)
    elif positions.ndim == 2:
        positions = positions[None, ...]
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(
            "pedestrian_positions must have shape (N, 2) or (B, N, 2), "
            f"got {positions.shape}"
        )

    batch, num_peds, _ = positions.shape
    density = np.zeros((batch, 1, grid_size, grid_size), dtype=np.float32)
    if num_peds == 0:
        return density

    if valid is None:
        mask = np.isfinite(positions).all(axis=-1)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask, (batch, num_peds)).copy()
        if mask.shape != (batch, num_peds):
            raise ValueError(
                f"valid must have shape {(batch, num_peds)}, got {mask.shape}"
            )
        mask &= np.isfinite(positions).all(axis=-1)

    resolution = roi_m / grid_size
    half = roi_m / 2.0
    xs = positions[..., 0]
    ys = positions[..., 1]
    mask &= (xs >= -half) & (xs < half) & (ys >= -half) & (ys < half)
    if not mask.any():
        return density

    center = grid_size // 2
    coords = (np.arange(grid_size, dtype=np.float64) - center) * resolution
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    dx = xx[None, None, ...] - xs[:, :, None, None]
    dy = yy[None, None, ...] - ys[:, :, None, None]
    kernel = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
    cell_area = resolution * resolution
    kernel *= cell_area / (2.0 * np.pi * sigma * sigma)
    kernel *= mask[:, :, None, None]
    density[:, 0] = kernel.sum(axis=1).astype(np.float32)
    return density


def next_density_observation(
    previous: np.ndarray,
    pedestrian_positions: np.ndarray,
    grid_size: int = 16,
    roi_m: float = 20.0,
    sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the current density on top of the previous frame.

    Returns ``(observation, current)``. ``observation`` has shape
    ``(2, grid_size, grid_size)``: channel 0 is this step, channel 1 is
    ``previous``. The caller stores ``current`` and passes it back next step.
    An episode reset passes a zero ``previous``.
    """
    current = rasterize_density(
        pedestrian_positions,
        grid_size=grid_size,
        roi_m=roi_m,
        sigma=sigma,
    )[0, 0]
    observation = np.stack([current, np.asarray(previous, dtype=np.float32)], axis=0)
    return observation, current
