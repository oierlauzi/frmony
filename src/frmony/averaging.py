"""Putting the aligned subtomograms back together.

Once every particle has an orientation and a centre, the structure is the
average of them -- but not the average of the volumes. Each particle
measured a different part of Fourier space, and turning it turns its
missing wedge with it, so a plain mean in real space adds the gaps together
instead of filling them.

What fills them is a weighted average in Fourier space:

    S(k) = sum_i Lambda_i[D_i](k) / (sum_i Lambda_i[W_i](k) + epsilon),

where each particle contributes what it measured, weighted by how well it
measured it. A frequency one particle never saw is supplied by the others,
because they were pointing elsewhere. Where no particle saw a frequency at
all, the numerator and the denominator are both nothing and the
regularizer leaves it empty, which is the honest answer.

The resolution is reported by Fourier shell correlation between two halves
of the data, each averaged on its own, which is the usual gold standard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .matching.translation import frequency_grid, rotate_spectrum

__all__ = ['average', 'fourier_shell_correlation', 'resolution_at']


def _phase_ramp(size: int, shift: ArrayLike) -> jax.Array:
    """``exp(-2 pi i k . a)``, which moves a volume by ``a``.

    The shift is ``(x, y, z)`` while the volume is indexed ``[z, y, x]``.
    """
    axis = (jnp.arange(size) - size // 2) / size
    offset = jnp.asarray(shift)
    phase = (
        axis[:, None, None] * offset[2]
        + axis[None, :, None] * offset[1]
        + axis[None, None, :] * offset[0]
    )
    return jnp.exp(-2j * jnp.pi * phase)


def average(
    data: ArrayLike,
    weight: ArrayLike,
    rotations: ArrayLike,
    shifts: ArrayLike,
    epsilon: float = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    """The structure the aligned particles agree on.

    Args:
        data: Centred spectra ``(particles, n, n, n)``.
        weight: Their Fourier weights, the same shape.
        rotations: The orientation of each particle ``(particles, 3, 3)``.
        shifts: Its centre ``(particles, 3)``, in voxels of the spectra
            and ordered ``(x, y, z)``.
        epsilon: Added to the denominator, as a fraction of its largest
            value, so that frequencies nobody measured stay empty instead
            of being amplified out of nothing.

    Returns:
        The structure's centred spectrum and the total weight behind each
        of its frequencies. The second is worth keeping: it says which
        parts of the answer are supported and which are guesses.
    """
    data = jnp.asarray(data)
    weight = jnp.asarray(weight)
    rotations = jnp.asarray(rotations)
    shifts = jnp.asarray(shifts)
    size = data.shape[-1]
    grid = jnp.asarray(frequency_grid(size))

    def one(inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
        spectrum, mask, rotation, shift = inputs
        # Undo the particle's own shift, then bring it into the common
        # frame. Turning by the inverse is what takes a particle onto the
        # structure rather than the other way about.
        centred = spectrum * _phase_ramp(size, -shift)
        turned = rotate_spectrum(centred, rotation.T, grid)
        turned_weight = rotate_spectrum(mask, rotation.T, grid)
        covered = jnp.maximum(turned_weight, 0.0)
        return turned * covered, covered * covered

    def accumulate(
        carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        top, bottom = carry
        contribution, support = one(inputs)
        return (top + contribution, bottom + support), None

    start = (
        jnp.zeros(data.shape[1:], dtype=jnp.complex64),
        jnp.zeros(data.shape[1:], dtype=jnp.float32),
    )
    (top, bottom), _ = jax.lax.scan(
        accumulate, start, (data, weight, rotations, shifts)
    )
    floor = epsilon * jnp.maximum(jnp.max(bottom), 1e-20)
    return top / (bottom + floor), bottom


def fourier_shell_correlation(
    first: ArrayLike, second: ArrayLike, shells: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """How far two structures agree, frequency by frequency.

    Args:
        first: A centred spectrum ``(n, n, n)``.
        second: Another of the same shape.
        shells: How many shells to report. Half the box by default.

    Returns:
        The frequency of each shell, as a fraction of the sampling rate,
        and the correlation there.
    """
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape:
        raise ValueError('Two structures to compare have the same shape.')
    size = left.shape[-1]
    count = shells if shells is not None else size // 2

    axis = np.arange(size) - size // 2
    radius = np.sqrt(
        axis[:, None, None] ** 2
        + axis[None, :, None] ** 2
        + axis[None, None, :] ** 2
    )
    index = np.clip(
        np.floor(radius * count / (size / 2)).astype(np.int64), 0, count
    )

    def total(values: NDArray[np.floating]) -> NDArray[np.float64]:
        summed: NDArray[np.float64] = np.bincount(
            index.ravel(), weights=values.ravel(), minlength=count + 1
        )[:count].astype(np.float64)
        return summed

    cross = total((left * np.conjugate(right)).real)
    power = np.sqrt(total(np.abs(left) ** 2) * total(np.abs(right) ** 2))
    correlation = np.where(
        power > 0.0, cross.real / np.maximum(power, 1e-30), 0.0
    )
    frequency = (np.arange(count) + 0.5) / (2.0 * count)
    return frequency, correlation


def resolution_at(
    frequency: NDArray[np.float64],
    correlation: NDArray[np.float64],
    pixel_size: float,
    threshold: float = 0.143,
) -> float:
    """Where a shell correlation first falls below a threshold [A].

    Returns infinity if it never does, which means the data ran out before
    the agreement did.
    """
    below = np.flatnonzero(correlation < threshold)
    if len(below) == 0:
        return float('inf')
    first = int(below[0])
    if first == 0:
        return float('inf')
    # Straight line between the two shells that straddle the threshold.
    before, after = correlation[first - 1], correlation[first]
    span = before - after
    fraction = (before - threshold) / span if span > 0 else 0.0
    crossing = frequency[first - 1] + fraction * (
        frequency[first] - frequency[first - 1]
    )
    return float(pixel_size / crossing) if crossing > 0 else float('inf')
