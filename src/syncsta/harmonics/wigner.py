"""The small Wigner d functions, and the quarter turn the method turns on.

A rotation mixes spherical harmonics of one degree among themselves, through
the Wigner matrices

    D^l_{mn}(phi, theta, psi) = exp(-i m phi) d^l_{mn}(theta) exp(-i n psi).

Two of the three Euler angles are already exponentials, and the third is not.
What makes fast rotational matching possible is that the third can be turned
into one as well, because a rotation by ``theta`` about y is a rotation by a
quarter turn, one by ``theta`` about z, and a quarter turn back:

    d^l_{mn}(theta) = sum_h d^l_{mh}(pi/2) exp(-i h theta) d^l_{hn}(pi/2).

So the whole correlation becomes a Fourier series in three angles, and a
single inverse FFT evaluates it everywhere at once. The quarter-turn matrices
are the only thing that has to be precomputed, and they depend on nothing but
the bandwidth.

They are built here as the matrix exponential of the generator of rotations
about y, which is a real antisymmetric tridiagonal matrix with known entries.
That is stable at every degree, and it is right by construction rather than
by a recurrence that has to be trusted.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .legendre import orders

__all__ = ['half_turn_table', 'quarter_turn_identity', 'wigner_d']


@functools.lru_cache(maxsize=256)
def _generator(degree: int) -> NDArray[np.float64]:
    """The generator of rotations about y, in the basis of ascending m.

    Real and antisymmetric, so its exponential is a rotation matrix and the
    d functions come out orthogonal to the last bit.
    """
    order = np.arange(-degree, degree, dtype=np.float64)
    coupling = 0.5 * np.sqrt((degree - order) * (degree + order + 1.0))
    size = 2 * degree + 1
    matrix = np.zeros((size, size))
    matrix[1:, :-1] += np.diag(coupling)
    matrix[:-1, 1:] -= np.diag(coupling)
    return matrix


@functools.lru_cache(maxsize=256)
def wigner_d(degree: int, angle: float) -> NDArray[np.float64]:
    """``d^l_{mn}(angle)`` for one degree, ``(2l + 1, 2l + 1)``.

    Args:
        degree: The degree ``l``.
        angle: The second Euler angle [rad].

    Returns:
        A matrix indexed by ``m`` and ``n``, both running from ``-l`` to
        ``l`` in that order.
    """
    if degree < 0:
        raise ValueError(f'A degree is not negative, unlike {degree}.')
    matrix: NDArray[np.float64] = expm(-angle * _generator(degree))
    return matrix


@functools.lru_cache(maxsize=16)
def half_turn_table(bandwidth: int) -> NDArray[np.float64]:
    """``d^l_{mn}(pi/2)`` for every degree below ``bandwidth``.

    Returns:
        An array ``(bandwidth, 2 * bandwidth, 2 * bandwidth)``, indexed by
        degree and then by ``m`` and ``n`` in the order an FFT lays them
        out. Entries with ``abs(m) > l`` or ``abs(n) > l`` are zero.
    """
    if bandwidth < 1:
        raise ValueError(f'The bandwidth must be positive, not {bandwidth}.')
    index = orders(bandwidth)
    table = np.zeros((bandwidth, 2 * bandwidth, 2 * bandwidth))
    for degree in range(bandwidth):
        block = wigner_d(degree, 0.5 * math.pi)
        inside = np.flatnonzero(np.abs(index) <= degree)
        rows = index[inside] + degree
        table[degree][np.ix_(inside, inside)] = block[np.ix_(rows, rows)]
    return table


def quarter_turn_identity(degree: int, angle: float) -> NDArray[np.complex128]:
    """``d^l(angle)`` rebuilt from the quarter turn, for checking.

    The factorization the method rests on, written out:

        d^l(b) = Z(-pi/2) @ d^l(pi/2).T @ Z(b) @ d^l(pi/2) @ Z(pi/2),

    with ``Z(a) = diag(exp(-i m a))`` over ascending ``m``. In indices,

        d^l_{mn}(b) = i**(m - n)
                      * sum_h d^l_{hm}(pi/2) exp(-i h b) d^l_{hn}(pi/2),

    which is what turns the second Euler angle into an exponential like the
    other two, and so the correlation into a Fourier series in all three.

    This is not used in the hot path, where the phases are folded into the
    surrounding contraction. It exists so a test can hold the identity to
    account directly.
    """
    index = np.arange(-degree, degree + 1)
    half = wigner_d(degree, 0.5 * math.pi).astype(np.complex128)
    rebuilt: NDArray[np.complex128] = (
        np.diag(np.exp(0.5j * math.pi * index))
        @ half.T
        @ np.diag(np.exp(-1j * index * angle))
        @ half
        @ np.diag(np.exp(-0.5j * math.pi * index))
    )
    return rebuilt
