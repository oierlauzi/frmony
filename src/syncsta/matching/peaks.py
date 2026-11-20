"""Finding where a correlation is largest, and where between samples.

A correlation is sampled on a grid, so the best sample is only as good as
the grid is fine. Fitting a parabola through the sample and its two
neighbours along each axis recovers most of the rest, which matters here
because the angular grid of a rotational search is coarse: at bandwidth 32
its step is five and a half degrees, and the interpolated peak is good to a
fraction of that.

The precedent this package grew out of had no interpolation at all, and so
was capped at the step of the grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ['refine_peak', 'top_peaks']


def _parabola(
    values: jax.Array, grid: jax.Array, axis: int, periodic: bool
) -> tuple[jax.Array, jax.Array]:
    """Offset and curvature of a parabola through three samples.

    Returns the offset in ``[-0.5, 0.5]`` and the height it adds.
    """
    size = grid.shape[axis]
    index = values[axis]
    if periodic:
        before = (index - 1) % size
        after = (index + 1) % size
    else:
        before = jnp.clip(index - 1, 0, size - 1)
        after = jnp.clip(index + 1, 0, size - 1)

    def at(position: jax.Array) -> jax.Array:
        where = [values[other] for other in range(values.shape[0])]
        where[axis] = position
        return grid[tuple(where)]

    low, middle, high = at(before), at(index), at(after)
    curvature = low - 2.0 * middle + high
    # A flat or upward curvature means this is not a peak along this axis,
    # so stay put rather than extrapolate off it.
    safe = jnp.where(jnp.abs(curvature) < 1e-20, -1.0, curvature)
    offset = jnp.clip(0.5 * (low - high) / safe, -0.5, 0.5)
    offset = jnp.where(curvature < 0.0, offset, 0.0)
    return offset, -0.25 * (low - high) * offset


def refine_peak(
    grid: ArrayLike,
    index: ArrayLike,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[jax.Array, jax.Array]:
    """Fit a parabola along each axis through a sample of a grid.

    Args:
        grid: A three-dimensional array of correlation values.
        index: The integer index of the sample, ``(3,)``.
        periodic: Whether each axis wraps. The angles of a rotational search
            do; the shifts of a translational one, once masked, do not.

    Returns:
        The index as a fractional position, ``(3,)``, and the height of the
        interpolated peak.
    """
    grid = jnp.asarray(grid)
    index = jnp.asarray(index)
    if grid.ndim != 3:
        raise ValueError('A peak is refined on a three-dimensional grid.')

    offsets = []
    height = grid[tuple(index[axis] for axis in range(3))]
    for axis in range(3):
        offset, added = _parabola(index, grid, axis, periodic[axis])
        offsets.append(offset)
        height = height + added
    return index.astype(grid.dtype) + jnp.stack(offsets), height


def top_peaks(
    grid: ArrayLike, count: int, separation: int = 1
) -> tuple[jax.Array, jax.Array]:
    """The largest samples of a grid, no two of them close together.

    Greedy: take the largest, suppress everything within ``separation`` of
    it along every axis, and repeat. Suppression wraps, since the angular
    axes do.

    Several candidates matter because the rotational stage works on
    amplitudes alone, which discard the phases and so discriminate less
    than the full correlation does. The right rotation is reliably among the
    best few even when it is not the best one, and the translational stage
    settles between them on the full score.

    Args:
        grid: A three-dimensional array of correlation values.
        count: How many peaks to return.
        separation: How far apart they must be, in samples.

    Returns:
        Indices ``(count, 3)`` and the value at each, largest first.
    """
    grid = jnp.asarray(grid)
    if grid.ndim != 3:
        raise ValueError('Peaks are found on a three-dimensional grid.')
    if count < 1:
        raise ValueError(f'At least one peak is needed, not {count}.')

    shape = grid.shape
    axes = [jnp.arange(extent) for extent in shape]

    def step(
        remaining: jax.Array, _: None
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        flat = jnp.argmax(remaining)
        index = jnp.stack(jnp.unravel_index(flat, shape))
        value = remaining.reshape(-1)[flat]

        near = jnp.ones(shape, dtype=bool)
        for axis in range(3):
            distance = jnp.abs(axes[axis] - index[axis])
            distance = jnp.minimum(distance, shape[axis] - distance)
            close = distance <= separation
            near = near & jnp.expand_dims(
                close, [other for other in range(3) if other != axis]
            )
        return jnp.where(near, -jnp.inf, remaining), (index, value)

    _, (indices, values) = jax.lax.scan(
        step, grid.astype(jnp.float32), None, length=count
    )
    return indices, values
