"""Five degrees of freedom at once, after Kovacs et al. (2003).

The engine in `frmony.matching.pairwise` splits the six degrees of freedom
in two: three rotational ones found on amplitudes, then three translational
ones found by a transform. This finds five of them together.

The trick is to rotate *both* objects and translate one of them along a
single axis. A correlation written that way,

    c(R, R'; rho) = integral of Lambda_R f * conj(T_rho Lambda_R' g),

depends on five angles and one distance, and the five angles all become
Fourier variables, exactly as the three did in the purely rotational case.
One inverse transform in five dimensions then evaluates the correlation at
every rotation of both objects at once, and only ``rho`` is left to scan.

Why this is not the default is a matter of arithmetic. The correlation is
``(2B)**5`` numbers *per value of rho*: a quarter of a gigabyte at
bandwidth 16, eight at bandwidth 32, against one megabyte for the three
dimensional grid at bandwidth 32. Worse, the two-centre integrals depend on
both particles and on ``rho``, so nothing can be shared between pairs and
nothing batches. Across a graph of a hundred thousand pairs that is the
difference between minutes and days.

What it is for is checking. Run on a sample of pairs it says, independently
and by a different route, whether the default engine is finding the right
answer.

The two-centre integrals are the whole cost, and Appendix A of the paper is
what makes them affordable. In cylindrical coordinates, with ``s = r sin
beta`` and ``z = r cos beta``,

    r  = sqrt(s**2 + z**2),   r' = sqrt(s**2 + (z - rho)**2),
    cos beta = z / r,         cos beta' = (z - rho) / r',

and the integrand separates into a factor belonging to each particle:

    F[n, lm, sz]  = conj(f[l,m](r))  * d^l_{n0}(beta)     (no rho in it)
    G[n, l'm', sz] = g[l',m'](r')    * d^l'_{n0}(beta')
    I[n, lm, l'm'] = F[n] @ (w * G[n]).T

which is one batched matrix product over ``n``. The Wigner functions with a
zero index are just the Legendre functions already built for the expansion,
since ``d^l_{n0}(beta) = sqrt(4 pi / (2l + 1)) * Ybar_l^n(cos beta)``, so
nothing new is needed for them.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from ..harmonics.legendre import _legendre_on
from ..harmonics.wigner import half_turn_table

__all__ = [
    'CylindricalGrid',
    'correlation_5d',
    'cylindrical_grid',
    'memory_for',
    'series_5d',
    'two_centre_integrals',
]


def memory_for(bandwidth: int, dtype: int = 8) -> int:
    """Bytes one five-dimensional correlation takes, before anything else.

    Args:
        bandwidth: The bandwidth it would be run at.
        dtype: Bytes per entry; eight for single-precision complex.
    """
    return int((2 * bandwidth) ** 5) * dtype


class CylindricalGrid:
    """Where the two-centre integrals are evaluated, and with what weight."""

    __slots__ = ('inner', 'outer', 's', 'weight', 'z')

    def __init__(
        self,
        s: NDArray[np.float64],
        z: NDArray[np.float64],
        weight: NDArray[np.float64],
        inner: float,
        outer: float,
    ) -> None:
        self.s = s
        self.z = z
        self.weight = weight
        self.inner = inner
        self.outer = outer


@functools.lru_cache(maxsize=8)
def cylindrical_grid(
    inner: float, outer: float, reach: float, samples: int = 0
) -> CylindricalGrid:
    """A quadrature over the half plane the integrals run on.

    Args:
        inner: Innermost radius the coefficients describe.
        outer: Outermost radius they describe.
        reach: Largest distance the second object is moved by, which is how
            far beyond the first object's own extent the grid has to go.
        samples: Points along each axis. Twice the outer radius, rounded
            up, if not given, which samples about one point per voxel.

    Returns:
        A `CylindricalGrid`, flattened so that the integrals are a matrix
        product.
    """
    if not 0.0 <= inner < outer:
        raise ValueError(f'A band from {inner} to {outer} is not a band.')
    count = samples if samples > 0 else max(math.ceil(2 * outer), 8)

    radial = (np.arange(count) + 0.5) * outer / count
    axial = np.linspace(-outer, outer + reach, 2 * count)
    mesh_s, mesh_z = np.meshgrid(radial, axial, indexing='ij')

    # The s ds dz of the integral, with the whole half plane uniform.
    area = (outer / count) * (axial[1] - axial[0])
    weight = mesh_s * area
    return CylindricalGrid(
        s=mesh_s.ravel(),
        z=mesh_z.ravel(),
        weight=weight.ravel(),
        inner=inner,
        outer=outer,
    )


def _interpolate(
    coefficients: jax.Array, radii: jax.Array, query: jax.Array
) -> jax.Array:
    """Coefficients at arbitrary radii, linearly between the shells.

    Args:
        coefficients: ``(radii, bandwidth, 2 * bandwidth)``.
        radii: Where those shells are.
        query: Radii wanted, ``(points,)``.

    Returns:
        ``(points, bandwidth, 2 * bandwidth)``, zero outside the shells.
    """
    position = jnp.interp(
        query, radii, jnp.arange(radii.shape[0], dtype=radii.dtype)
    )
    low = jnp.clip(jnp.floor(position), 0, radii.shape[0] - 1).astype(jnp.int32)
    high = jnp.clip(low + 1, 0, radii.shape[0] - 1)
    fraction = (position - low)[:, None, None]
    inside = ((query >= radii[0]) & (query <= radii[-1]))[:, None, None]
    blended = (
        coefficients[low] * (1.0 - fraction) + coefficients[high] * fraction
    )
    return jnp.where(inside, blended, 0.0)


def _wigner_zero(
    bandwidth: int, cosine: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``d^l_{n0}`` at the colatitudes whose cosines are given.

    Returns:
        ``(points, bandwidth, 2 * bandwidth)``, zero where ``abs(n) > l``.
    """
    # The table built for the expansion lives on a quadrature, whose
    # nodes the bandwidth fixes; here the colatitudes are wherever the
    # cylindrical grid put them, so the recurrence is run on those.
    table = _legendre_on(bandwidth, np.asarray(cosine, dtype=np.float64))
    degree = np.arange(bandwidth)
    scale = np.sqrt(4.0 * np.pi / (2.0 * degree + 1.0))
    return table * scale[None, :, None]


def two_centre_integrals(
    left: ArrayLike,
    right: ArrayLike,
    radii: ArrayLike,
    radial_weight: ArrayLike,
    distance: float,
    grid: CylindricalGrid,
    bandwidth: int,
) -> jax.Array:
    """The integrals that carry both particles and the distance between.

    Args:
        left: Coefficients of the first particle,
            ``(radii, bandwidth, 2 * bandwidth)``.
        right: Coefficients of the second, the same shape.
        radii: Where the shells of both are.
        radial_weight: What `frmony.harmonics.make_basis` put into the
            coefficients, which has to come back out. An expansion carries
            its share of the ``r**2 dr`` of the radial integral so that the
            three-dimensional correlation is a plain sum over shells; here
            the quadrature supplies that factor itself, so leaving it in
            would count it twice.
        distance: How far the second is moved along ``+z``.
        grid: What `cylindrical_grid` built.
        bandwidth: The bandwidth both were expanded at.

    Returns:
        ``(2B, B, 2B, B, 2B)``, indexed by the Fourier variable ``n`` and
        then by the degree and order taken from each particle.
    """
    weight_per_shell = jnp.asarray(radial_weight)[:, None, None]
    bare = jnp.where(jnp.abs(weight_per_shell) > 0, weight_per_shell, 1.0)
    left = jnp.asarray(left) / bare
    right = jnp.asarray(right) / bare
    shells = jnp.asarray(radii, dtype=jnp.float32)

    s = jnp.asarray(grid.s, dtype=jnp.float32)
    z = jnp.asarray(grid.z, dtype=jnp.float32)
    weight = jnp.asarray(grid.weight, dtype=jnp.float32)

    first = jnp.sqrt(s * s + z * z)
    shifted = z - distance
    second = jnp.sqrt(s * s + shifted * shifted)
    cosine = jnp.where(first > 0, z / jnp.maximum(first, 1e-12), 1.0)
    cosine_shifted = jnp.where(
        second > 0, shifted / jnp.maximum(second, 1e-12), 1.0
    )

    here = _interpolate(left, shells, first)
    there = _interpolate(right, shells, second)
    turn_here = jnp.asarray(
        _wigner_zero(bandwidth, np.asarray(cosine)), dtype=jnp.float32
    )
    turn_there = jnp.asarray(
        _wigner_zero(bandwidth, np.asarray(cosine_shifted)),
        dtype=jnp.float32,
    )

    # F[p, n, l, m] and G[p, n, l, m], then contracted over the points.
    taken = jnp.einsum('plm,pln->pnlm', jnp.conjugate(here), turn_here)
    given = jnp.einsum('plm,pln->pnlm', there, turn_there)
    integrals = jnp.einsum('pnlm,p,pnLM->nlmLM', taken, weight, given)

    degree = jnp.arange(bandwidth, dtype=jnp.float32) + 0.5
    normal = jnp.sqrt(degree)
    return (
        integrals
        * normal[None, :, None, None, None]
        * normal[None, None, None, :, None]
    )


def series_5d(integrals: ArrayLike, bandwidth: int) -> jax.Array:
    """The Fourier coefficients of the correlation in five angles.

    Contracted in two steps so that no array of five free indices and two
    summed ones is ever built.

    Args:
        integrals: What `two_centre_integrals` returned.
        bandwidth: The bandwidth of the expansion.

    Returns:
        ``(2B, 2B, 2B, 2B, 2B)``, indexed by ``n, h, m, h', m'``.
    """
    integrals = jnp.asarray(integrals)
    size = 2 * bandwidth
    quarter = jnp.asarray(half_turn_table(bandwidth), dtype=integrals.dtype)

    # d^l_{-n, h}, which in the FFT's layout is the row at -n modulo 2B.
    index = jnp.arange(size)
    mirrored = quarter[:, (-index) % size, :]

    # Sum over the second particle's degree first.
    partial = jnp.einsum('LnH,LHM,nlmLM->nHMlm', mirrored, quarter, integrals)
    # Then over the first's.
    series = jnp.einsum('lnh,lhm,nHMlm->nhmHM', quarter, quarter, partial)

    sign = jnp.where(index % 2 == 0, 1.0, -1.0).astype(series.dtype)
    return series * sign[:, None, None, None, None]


def correlation_5d(series: ArrayLike) -> jax.Array:
    """Evaluate the five-dimensional series on its grid of angles.

    Args:
        series: What `series_5d` returned.

    Returns:
        A real array ``(2B,) * 5``.
    """
    series = jnp.asarray(series)
    size = series.shape[0]
    values = series
    # Every axis carries exp(+i x), so every one is an inverse transform,
    # scaled back up by what an inverse transform divides out.
    for axis in range(5):
        values = jnp.fft.ifft(values, axis=axis) * size
    return values.real
