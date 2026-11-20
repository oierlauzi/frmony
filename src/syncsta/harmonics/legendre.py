"""Spherical harmonics on the colatitude, and the quadrature that integrates
them exactly.

The transform of this package separates: an exact FFT around the azimuth,
and a matrix against the colatitude. This module builds that matrix.

The convention is the usual one of physics, with the Condon-Shortley phase
folded in, so that

    Y_lm(theta, phi) = legendre[l, m](cos theta) * exp(i m phi)

is orthonormal over the sphere. Writing the normalization into the recurrence
rather than applying it afterwards is what keeps the values bounded: the
unnormalized functions overflow long before the bandwidths this package uses.

Coefficients are laid out as ``(l, m)`` rectangles rather than packed
triangles, with ``m`` in the order an FFT produces it, ``0, 1, ..., B-1,
-(B-1), ..., -1``, and zeros where ``abs(m) > l``. The waste is a factor of
two and it buys a layout that needs no gather anywhere: the azimuthal FFT
writes straight into it, and the rotational correlation reads straight out of
it.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.typing import NDArray

__all__ = ['gauss_legendre', 'legendre_table', 'quadrature_table']


@functools.lru_cache(maxsize=16)
def gauss_legendre(
    nodes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes in ``cos(theta)`` and weights of a Gauss-Legendre quadrature.

    ``nodes`` of them integrate a polynomial of degree ``2 * nodes - 1``
    exactly. A product of two spherical harmonics of degree below ``B`` is a
    polynomial of degree at most ``2B - 2`` in ``cos(theta)``, so ``B`` nodes
    make the transform of a band-limited function exact. The uniform grid
    with a ``sin(theta)`` weight that is often used instead is not exact at
    any number of nodes.

    Returns:
        The nodes, increasing in ``cos(theta)``, and their weights.
    """
    if nodes < 1:
        raise ValueError(f'A quadrature needs at least one node, not {nodes}.')
    points, weights = np.polynomial.legendre.leggauss(nodes)
    return points.astype(np.float64), weights.astype(np.float64)


def _ravel_m(bandwidth: int) -> NDArray[np.int64]:
    """The order ``m`` that each column of a rectangle stands for."""
    return np.fft.fftfreq(2 * bandwidth, d=1.0 / (2 * bandwidth)).astype(
        np.int64
    )


@functools.lru_cache(maxsize=16)
def legendre_table(bandwidth: int, nodes: int) -> NDArray[np.float64]:
    """Normalized associated Legendre functions on the quadrature nodes.

    Built by the recurrence in its normalized form, which is stable: every
    value it carries is bounded by one.

    Args:
        bandwidth: Degrees ``l`` below this are built.
        nodes: How many Gauss-Legendre nodes to evaluate on.

    Returns:
        An array ``(nodes, bandwidth, 2 * bandwidth)``, indexed by node,
        degree and the order in the FFT's own layout. Entries with
        ``abs(m) > l`` are zero.
    """
    if bandwidth < 1:
        raise ValueError(f'The bandwidth must be positive, not {bandwidth}.')
    cosine, _ = gauss_legendre(nodes)
    sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))

    # Triangular in (l, m) with m >= 0 while recurring, then mirrored.
    upper = np.zeros((bandwidth, bandwidth, nodes))
    upper[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
    for degree in range(1, bandwidth):
        # The sectoral term, carrying the Condon-Shortley phase.
        upper[degree, degree] = (
            -np.sqrt((2.0 * degree + 1.0) / (2.0 * degree))
            * sine
            * upper[degree - 1, degree - 1]
        )
        # The one just below it.
        upper[degree, degree - 1] = (
            np.sqrt(2.0 * degree + 1.0) * cosine * upper[degree - 1, degree - 1]
        )
        # And the rest, by the three-term recurrence in the degree.
        for order in range(degree - 1):
            first = np.sqrt(
                (4.0 * degree * degree - 1.0) / (degree**2 - order**2)
            )
            second = np.sqrt(
                (
                    (2.0 * degree + 1.0)
                    * (degree + order - 1.0)
                    * (degree - order - 1.0)
                )
                / ((2.0 * degree - 3.0) * (degree - order) * (degree + order))
            )
            upper[degree, order] = (
                first * cosine * upper[degree - 1, order]
                - second * upper[degree - 2, order]
            )

    # Y_{l,-m} = (-1)^m conj(Y_lm) and the functions are real, so the
    # negative orders are the positive ones with an alternating sign.
    table = np.zeros((nodes, bandwidth, 2 * bandwidth))
    orders = _ravel_m(bandwidth)
    for column, order in enumerate(orders):
        magnitude = abs(int(order))
        if magnitude >= bandwidth:
            continue
        sign = -1.0 if (order < 0 and magnitude % 2) else 1.0
        table[:, magnitude:, column] = sign * np.moveaxis(
            upper[magnitude:, magnitude], -1, 0
        )
    return table


@functools.lru_cache(maxsize=16)
def quadrature_table(bandwidth: int, nodes: int) -> NDArray[np.float64]:
    """`legendre_table` with the quadrature weights already in it.

    This is the matrix the forward transform contracts the colatitude
    against. The azimuthal step carries its own ``2 pi / n`` separately.
    """
    _, weights = gauss_legendre(nodes)
    return legendre_table(bandwidth, nodes) * weights[:, None, None]


def orders(bandwidth: int) -> NDArray[np.int64]:
    """The order ``m`` that each column of a rectangle stands for."""
    return _ravel_m(bandwidth)
