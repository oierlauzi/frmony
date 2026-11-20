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

What it is for is measuring. It searches the six degrees of freedom of a
pair at once, so it is what a cheaper search of them should be held
against, and `frmony.matching.pairwise` does the same search another way.

Four things have to be right before it says anything about real
subtomograms, and each was found by it saying nothing.

The volumes are real-space ones. A shift is a displacement there, which
is what the two-centre integrals are built on; in Fourier space it is a
phase.

They are band-passed to the frequencies the rotational search uses. The
lowest ones carry a particle's mass and swamp the sum, exactly as they
do in three dimensions.

They are sampled twice as finely as the band, and the shells are half a
voxel of it apart. A density whose shortest wave is two voxels cannot be
read off shells a voxel apart and interpolated between.

And the score is divided by the denominators of the constrained
correlation, which depend on the relative rotation alone and so come
from the three-dimensional grids.

With those, on pairs of EMPIAR-10304 at bandwidth 24: 51 in a hundred
aligned to within fifteen degrees with the distance held at nothing,
which is what the three-dimensional search gives, 27 with the distance
searched on centred pairs, and 22 to 24 on pairs forty angstroms apart,
where a search of rotations alone gives 6. At 1.4 seconds a pair.

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
    'grid_to_rotations',
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

    Note that the first particle is the conjugated one here, following the
    2003 paper, while `frmony.matching.rotation.radial_overlap` conjugates
    the second, following the 2002 one. The two papers differ, and the
    result is that at a distance of zero these integrals come out as the
    *conjugate* of that overlap rather than equal to it. A test pins that
    down, since it is exactly the sort of thing that would otherwise be
    found much later and blamed on something else.
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

    Derived from the three-dimensional series and not from the papers.
    `frmony.matching.rotation.correlation_grid` is held to a direct
    quadrature by a test, so it is a trustworthy expression for the
    matrix elements of a rotation: for any ``I``,

        grid(Q) = sum_{l,n,m} conj(D^l_{nm}(Q)) I^l_{nm}
                = sum_{n,h,m} [sum_l I p_n conj(p_m) q_{lhn} q_{lhm}]
                  exp(+i n a) exp(-i h b) exp(+i m c),

    with ``p`` the powers of ``i``, ``q`` the quarter-turn table and
    ``(a, b, c)`` the angles of the grid index. The two-body correlation
    is

        c(Q1, Q2; rho) = sum D^l_{nm}(Q1) conj(D^L_{nM}(Q2)) K^n_{lm,LM}

    with ``K`` the two-centre integrals, and putting the first into the
    second gives

        U[n, h, m, H, M] = sum_{l,L} p_m conj(p_M)
                           q_{lhn} q_{lhm} q_{LHn} q_{LHM} K^n_{lm,LM},
        c = sum U exp(-i n d) exp(+i h b) exp(-i m c)
                  exp(-i H b') exp(+i M c').

    Both bodies share the order ``n``, so only the difference ``d`` of
    their first angles enters.

    What stood here before left out the powers of ``i``, mirrored the
    order of the second body and took every axis as an inverse
    transform. It was never held against a direct sum, only its
    integrals were, at a distance of zero. Its peak did not put two
    copies of a volume on top of each other: the best reading of its
    index overlapped them to 0.91 and 0.71 on two trials, where this one
    gives 0.999 on both.

    Contracted in two steps so that no array of five free indices and two
    summed ones is ever built.

    Args:
        integrals: What `two_centre_integrals` returned. Those conjugate
            the first particle; the correlation here conjugates the
            second, so they enter conjugated.
        bandwidth: The bandwidth of the expansion.

    Returns:
        ``(2B, 2B, 2B, 2B, 2B)``, indexed by ``n, h, m, H, M``.
    """
    from .rotation import _phases

    kernel = jnp.conjugate(jnp.asarray(integrals))
    quarter = jnp.asarray(half_turn_table(bandwidth), dtype=kernel.dtype)
    phase = jnp.asarray(_phases(bandwidth), dtype=kernel.dtype)

    # Sum over the second particle's degree first.
    partial = jnp.einsum('LHn,LHM,nlmLM->nlmHM', quarter, quarter, kernel)
    # Then over the first's.
    series = jnp.einsum('lhn,lhm,nlmHM->nhmHM', quarter, quarter, partial)
    return (
        series
        * phase[None, None, :, None, None]
        * jnp.conjugate(phase)[None, None, None, None, :]
    )


def correlation_5d(series: ArrayLike) -> jax.Array:
    """Evaluate the five-dimensional series on its grid of angles.

    Args:
        series: What `series_5d` returned.

    Returns:
        A real array ``(2B,) * 5``, indexed as `grid_to_rotations` reads
        it.
    """
    series = jnp.asarray(series)
    size = series.shape[0]
    # exp(-i n d), exp(+i h b), exp(-i m c), exp(-i H b'), exp(+i M c'):
    # a minus is a forward transform, a plus an inverse one scaled back
    # up by what it divides out.
    values = jnp.fft.fft(series, axis=0)
    values = jnp.fft.ifft(values, axis=1) * size
    values = jnp.fft.fft(values, axis=2)
    values = jnp.fft.fft(values, axis=3)
    values = jnp.fft.ifft(values, axis=4) * size
    return values.real


def grid_to_rotations(
    indices: ArrayLike, bandwidth: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The rotation of each body that an index of the grid stands for.

    The value at an index is

        sum over x of f(Q1^-1 x) * g(Q2^-1 (x - rho z)),

    the first body turned by ``Q1`` and the second turned by ``Q2`` and
    then moved by ``rho`` along ``+z``. Both rotations are read as
    `frmony.matching.rotation.grid_to_euler` reads the three-dimensional
    grid, which a test pins: the first body from the first three
    components, the second from the last two with no first angle of its
    own.

    In the terms the rest of the package uses, a pair aligned as
    ``V_i(x) = V_j(A.T @ (x - a))`` has

        A = Q1.T @ Q2,        a = rho * Q1.T @ z.

    Args:
        indices: Indices ``(..., 5)`` into the grid.
        bandwidth: The bandwidth the grid was built at.

    Returns:
        ``Q1`` and ``Q2``, each ``(..., 3, 3)``.
    """
    from ..geometry import euler_to_matrix
    from .rotation import grid_to_euler

    indices = np.asarray(indices)
    if indices.shape[-1] != 5:
        raise ValueError('An index into the grid has five components.')
    first = euler_to_matrix(*grid_to_euler(indices[..., :3], bandwidth))
    alone = np.concatenate(
        (np.zeros_like(indices[..., :1]), indices[..., 3:]), axis=-1
    )
    second = euler_to_matrix(*grid_to_euler(alone, bandwidth))
    return first, second
