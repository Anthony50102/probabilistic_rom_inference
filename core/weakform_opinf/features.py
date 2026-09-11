"""Input-aware GP trend features, normalized using training times only."""

import numpy as np


def integrate_linear_input(grid_times, grid_values, times):
    """Integrate a piecewise-linear input exactly from its first table time."""
    grid = np.asarray(grid_times, dtype=float)
    values = np.asarray(grid_values, dtype=float)
    query = np.asarray(times, dtype=float)
    if (grid.ndim != 1 or values.shape != grid.shape or len(grid) < 2
            or not np.all(np.isfinite(grid)) or not np.all(np.isfinite(values))
            or np.any(np.diff(grid) <= 0)):
        raise ValueError("Input table must be finite, one-dimensional, and strictly increasing")
    if (not np.all(np.isfinite(query))
            or np.any(query < grid[0]) or np.any(query > grid[-1])):
        raise ValueError("Integration times must lie within the input table")
    widths = np.diff(grid)
    slopes = np.diff(values) / widths
    cumulative = np.concatenate([
        [0.], np.cumsum(widths * (values[:-1] + values[1:]) / 2)])
    indices = np.minimum(np.searchsorted(grid, query, side="right") - 1, len(grid) - 2)
    offsets = query - grid[indices]
    return cumulative[indices] + offsets * values[indices] + .5 * slopes[indices] * offsets ** 2


def input_trend_features(t_sampled, t_eval, grid_times, grid_values):
    """Constant, time, and integrated-input features with analytic derivatives.

    Their coefficients have a Gaussian prior and are integrated into the GP
    kernel; these features do not prescribe any ROM operator or fitted mean.
    """
    train_times = np.asarray(t_sampled, dtype=float)
    eval_times = np.asarray(t_eval, dtype=float)
    if (train_times.ndim != 1 or len(train_times) < 2
            or eval_times.ndim != 1 or not np.all(np.isfinite(train_times))
            or np.any(np.diff(train_times) <= 0)):
        raise ValueError("Feature construction requires ordered training times and a 1D evaluation grid")
    train_values = np.column_stack([
        train_times, integrate_linear_input(grid_times, grid_values, train_times)])
    eval_values = np.column_stack([
        eval_times, integrate_linear_input(grid_times, grid_values, eval_times)])
    center = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Input trend requires nonconstant cumulative exposure during training")
    return dict(
        train=np.column_stack([np.ones(len(train_times)), (train_values - center) / scale]),
        eval=np.column_stack([np.ones(len(eval_times)), (eval_values - center) / scale]),
        derivative=np.column_stack([
            np.zeros(len(eval_times)), np.full(len(eval_times), 1 / scale[0]),
            np.interp(eval_times, grid_times, grid_values) / scale[1]]),
        center=center, scale=scale)
