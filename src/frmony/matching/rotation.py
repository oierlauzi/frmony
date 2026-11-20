"""Fast rotational matching: the correlation of two volumes at every
rotation at once.

Expanded into spherical harmonics, the correlation of a volume with a
rotated copy of another is

    c(R) = sum_{l,m,n} conj(D^l_{nm}(R)) * I^l_{nm},

where ``I`` is the radial overlap of the two sets of coefficients and ``D``
is a Wigner matrix. Two of the three Euler angles enter ``D`` as
exponentials already. The quarter-turn identity in `frmony.harmonics.wigner`
makes the third one an exponential too, and the whole correlation collapses
into a Fourier series in all three:

    c(phi, theta, psi) = sum_{n,h,m} T[n,h,m]
                         * exp(i n phi) exp(-i h theta) exp(i m psi).

One inverse FFT then evaluates it on a grid of every rotation. That is the
whole of Kovacs and Wriggers (2002), and it is why a rotational search costs
about as much as a single three-dimensional transform rather than one per
angle tried.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._transfers import to_device
from ..harmonics.legendre import orders
from ..harmonics.wigner import half_turn_table

__all__ = [
    'correlation_grid',
    'grid_angles',
    'grid_to_euler',
    'radial_overlap',
    'rotational_correlation',
]


@functools.lru_cache(maxsize=16)
def _phases(bandwidth: int) -> np.ndarray:
    """``i**m`` for each column, which the quarter-turn identity leaves over.

    Folding them into the coefficients once is cheaper than carrying them
    through the contraction, and it keeps the contraction itself readable.
    """
    return (1j ** orders(bandwidth)).astype(np.complex128)


def radial_overlap(left: ArrayLike, right: ArrayLike) -> jax.Array:
    """The radial integral of two sets of coefficients, ``I^l_{nm}``.

    Both sides already carry their share of the ``r**2 dr`` of the integral,
    put there when they were expanded, so this is a plain sum over shells.
    That split is what lets a particle be expanded once and then correlated
    with any number of others.

    Args:
        left: Coefficients ``(radii, bandwidth, 2 * bandwidth)``.
        right: Coefficients of the same shape.

    Returns:
        ``(bandwidth, 2 * bandwidth, 2 * bandwidth)``, indexed by degree and
        then by the order taken from each side.
    """
    left = to_device(left)
    right = to_device(right)
    if left.shape != right.shape:
        raise ValueError(
            f'Coefficients of shapes {left.shape} and {right.shape} are not '
            'of the same expansion.'
        )
    return jnp.einsum('rln,rlm->lnm', left, jnp.conjugate(right))


def rotational_correlation(
    left: ArrayLike,
    right: ArrayLike,
    bandwidth: int,
) -> jax.Array:
    """Correlate two expansions at every rotation of a grid.

    Args:
        left: Coefficients of the first volume,
            ``(radii, bandwidth, 2 * bandwidth)``.
        right: Coefficients of the second.
        bandwidth: The bandwidth both were expanded at.

    Returns:
        A real array ``(2B, 2B, 2B)`` holding

            c(Q) = integral of left(u) * conj(right(Q^-1 u)),

        indexed by `grid_to_euler`. It is largest at the ``Q`` that brings
        ``right`` onto ``left``: if ``right`` is ``left`` turned by ``R``,
        meaning ``right(u) = left(R^-1 u)``, the peak is at ``Q = R^-1``,
        because that is the turn that undoes it.
    """
    overlap = radial_overlap(left, right)
    return correlation_grid(overlap, bandwidth)


def correlation_grid(overlap: ArrayLike, bandwidth: int) -> jax.Array:
    """Evaluate the Fourier series of the correlation on its grid.

    Split from `rotational_correlation` because the constrained score needs
    three of these from overlaps it combines itself.

    Args:
        overlap: ``I^l_{nm}`` from `radial_overlap`.
        bandwidth: The bandwidth of the expansion.

    Returns:
        A real array ``(2B, 2B, 2B)``.
    """
    overlap = to_device(overlap)
    size = 2 * bandwidth
    # Under a trace these cross once and become constants. Called eagerly
    # they cross per call, which is why they are small.
    quarter = to_device(half_turn_table(bandwidth), dtype=overlap.dtype)
    phase = to_device(_phases(bandwidth), dtype=overlap.dtype)

    # The i**(n - m) the identity leaves over, split between the two sides.
    weighted = (
        overlap * phase[None, :, None] * jnp.conjugate(phase)[None, None, :]
    )
    series = jnp.einsum('lnm,lhn,lhm->nhm', weighted, quarter, quarter)

    # exp(+i n phi) and exp(+i m psi) are inverse transforms, exp(-i h theta)
    # a forward one. The factors of the size undo what ifft divides by, so
    # that the result is the sum written in the docstring and not a scaled
    # copy of it.
    values = jnp.fft.ifft(series, axis=0) * size
    values = jnp.fft.fft(values, axis=1)
    values = jnp.fft.ifft(values, axis=2) * size
    return values.real


def grid_angles(bandwidth: int) -> np.ndarray:
    """The Euler angles [rad] each index of the grid stands for.

    Returns:
        ``(2B,)`` of angles, the same for each of the three axes.
    """
    size = 2 * bandwidth
    return 2.0 * np.pi * np.arange(size) / size


def grid_to_euler(
    indices: ArrayLike, bandwidth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Euler angles [rad] that an index into the grid stands for.

    The three axes of the grid are the three Fourier variables, ``n``, ``h``
    and ``m``, and they turn out to be the negated ``psi``, ``tilt`` and
    ``rot`` of RELION's convention, in that order. That correspondence was
    not derived; it was found by holding the grid against a direct
    quadrature of the correlation over the sphere, and
    `test_the_grid_index_means_what_it_says` keeps it honest. The precedent
    this package grew out of left four unresolved questions about exactly
    this chain of conventions, which is reason enough not to reason it out
    on paper.

    Note that the grid covers every rotation twice: ``(rot, tilt, psi)`` and
    ``(rot + pi, -tilt, psi + pi)`` are the same rotation, so an index and
    its reflection in the second axis hold the same value. `peaks` searches
    half of that axis for exactly this reason.

    Args:
        indices: Indices ``(..., 3)`` into the correlation grid.
        bandwidth: The bandwidth the grid was built at.

    Returns:
        The angles ``rot``, ``tilt`` and ``psi``, each of shape ``(...)``.
    """
    indices = np.asarray(indices)
    if indices.shape[-1] != 3:
        raise ValueError('An index into the grid has three components.')
    step = np.pi / bandwidth
    return (
        -step * indices[..., 2],
        -step * indices[..., 1],
        -step * indices[..., 0],
    )
