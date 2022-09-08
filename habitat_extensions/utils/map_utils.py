import numpy as np


def within_grid(coords, num_cells):
    """Check whether coordinates are within range.

    Args:
        coords: [..., D] integer tensor
        num_cells: [D] (broadcastable)

    Returns:
        mask: [...] bool tensor
    """
    coords = np.array(coords)
    num_cells = np.array(num_cells)
    mask = np.logical_and(
        coords >= np.zeros_like(num_cells), coords < num_cells
    )
    return np.all(mask, axis=-1)


def to_grid(
    points,
    origin,
    cell_size,
    round=False,
):
    """Discretize continuous coordinates.

    Args:
        points: [N, D] or [D]
        origin: [D] (broadcastable)
        cell_size: [D] or scalar (broadcastable)
        round: If True, points are discretized by round(x).
            If False, floor(x) is used.

    Returns:
        coords: [N, D]
    """
    points = np.array(points)
    coords = (points - origin) / cell_size
    if round:
        coords = np.round(coords).astype(int)
    else:
        coords = np.floor(coords).astype(int)
    return coords


def from_grid(coords, origin, cell_size, offset=None):
    """Get continuous coordinates from discrete coordinates in the grid.

    Args:
        coords: [N, D]
        origin: [D] (broadcastable)
        cell_size: [D] or scalar (broadcastable)
        offset: scalar or None. offset added to @coords.

    Returns:
        np.ndarray: [N, D]
    """
    coords = np.array(coords)
    if offset is not None:
        coords = coords + offset
    return coords * cell_size + origin


def get_boundary(coords, window, size):
    """Get the boundary of a window.

    Args:
        coords: [..., D]
        window: [D] or scalar. The window size.
        size: [D]

    Returns:
        lower: [..., D]
        upper: [..., D]
    """
    window = np.array(window)
    half_window = window // 2
    lower = np.maximum(coords - half_window, 0)
    upper = np.minimum(lower + window, size)
    lower = upper - window
    return lower, upper
