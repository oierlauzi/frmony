"""From spheres to spherical harmonic coefficients.

The transform separates into an exact FFT around the azimuth and a matrix
against the colatitude:

    f[l, m](r) = sum_t w_t * P[t, l, m] * (2 pi / n) * FFT(shell)[t, m].

With Gauss-Legendre nodes in the colatitude the quadrature is exact for a
band-limited function, so the transform is not an approximation of the
coefficients, it is the coefficients.

Coefficients are rectangles, ``(radii, bandwidth, 2 * bandwidth)``, with the
order ``m`` laid out as an FFT lays it out. That is why no step here needs a
gather: the azimuthal FFT already writes its output in the layout the rest of
the package reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._transfers import to_device
from .legendre import quadrature_table
from .shells import sample_shells, shell_grid

__all__ = ['Basis', 'expand', 'expand_shells', 'make_basis']


@dataclass(frozen=True)
class Basis:
    """Everything needed to expand a volume, precomputed.

    Built once for a given bandwidth and set of radii, then reused for every
    particle. The tables cost nothing to keep: a bandwidth of 32 with 32
    radii is a few hundred kilobytes.
    """

    grid: jax.Array
    """Where to read the volume, ``(3, radii, nodes, azimuths)``."""
    colatitude: jax.Array
    """The weighted Legendre matrix, ``(nodes, bandwidth, 2 * bandwidth)``."""
    radial: jax.Array
    """Radial quadrature weight of each shell, ``(radii,)``."""
    bandwidth: int = 1
    """Degrees below this are kept."""
    azimuths: int = 2
    """Samples around the azimuth, and the length of the FFT."""

    def __post_init__(self) -> None:
        if self.bandwidth < 1:
            raise ValueError('The bandwidth must be positive.')


# The tables are what JAX traces over; the two sizes shape the computation
# and so have to stay static, or every bandwidth would retrace.
jax.tree_util.register_dataclass(
    Basis,
    data_fields=['grid', 'colatitude', 'radial'],
    meta_fields=['bandwidth', 'azimuths'],
)


def make_basis(
    radii: ArrayLike,
    bandwidth: int,
    size: int,
    sampling_bandwidth: int | None = None,
) -> Basis:
    """Precompute what the expansion of a volume needs.

    Args:
        radii: Radii of the spheres, in voxels. They need not be uniform.
        bandwidth: Degrees below this are kept.
        size: Extent of the volumes along each axis.
        sampling_bandwidth: Bandwidth the spheres are sampled at, which may
            be higher than the one kept. A product of two band-limited
            functions, which is what the normalization of the correlation
            takes, has twice the bandwidth of its factors, so sampling
            higher than it keeps is how that product is integrated without
            aliasing. Defaults to ``bandwidth``.

    Raises:
        ValueError: If the bandwidth is not positive, if the sampling
            bandwidth is below it, or if a radius is negative.
    """
    if bandwidth < 1:
        raise ValueError(f'The bandwidth must be positive, not {bandwidth}.')
    sampling = bandwidth if sampling_bandwidth is None else sampling_bandwidth
    if sampling < bandwidth:
        raise ValueError(
            f'Sampling at bandwidth {sampling} cannot support coefficients '
            f'up to {bandwidth}.'
        )

    shells = np.asarray(radii, dtype=np.float64)
    if shells.ndim != 1:
        raise ValueError('The radii are a sequence of numbers.')
    if np.any(shells < 0.0):
        raise ValueError('A radius is not negative.')

    nodes, azimuths = sampling, 2 * sampling
    grid = shell_grid(tuple(shells.tolist()), nodes, azimuths, size)
    colatitude = quadrature_table(bandwidth, nodes)

    # The r^2 dr of the radial integral, split evenly between the two sides
    # of a correlation so that each particle can be expanded once and then
    # correlated with any other. Trapezoidal in r, which is what a
    # non-uniform set of radii allows.
    spacing = np.gradient(shells) if len(shells) > 1 else np.ones(1)
    weight = np.sqrt(np.abs(shells * shells * spacing))

    # The one crossing a basis makes. Everything after it reads these
    # from the device.
    return Basis(
        grid=to_device(grid),
        colatitude=to_device(colatitude),
        radial=to_device(weight),
        bandwidth=bandwidth,
        azimuths=azimuths,
    )


def expand(volume: ArrayLike, basis: Basis) -> jax.Array:
    """Spherical harmonic coefficients of a volume, shell by shell.

    Args:
        volume: A centred volume ``(size, size, size)``, indexed
            ``[z, y, x]``, real or complex.
        basis: What `make_basis` built.

    Returns:
        Coefficients ``(radii, bandwidth, 2 * bandwidth)``, already carrying
        their share of the radial weight.
    """
    return expand_shells(sample_shells(volume, basis.grid), basis)


def expand_shells(shells: ArrayLike, basis: Basis) -> jax.Array:
    """Spherical harmonic coefficients of values already read off spheres.

    The half of `expand` that does the mathematics, split out so that it can
    be held to account on values whose coefficients are known exactly,
    without the interpolation of the volume in the way.

    Args:
        shells: Values on the sampling grid, ``(radii, nodes, azimuths)``.
        basis: What `make_basis` built.

    Returns:
        Coefficients ``(radii, bandwidth, 2 * bandwidth)``, already carrying
        their share of the radial weight.
    """
    shells = jnp.asarray(shells)
    spectrum = jnp.fft.fft(shells, axis=-1) * (2.0 * jnp.pi / basis.azimuths)
    kept = _keep_orders(spectrum, basis.bandwidth)
    colatitude = basis.colatitude.astype(kept.dtype)
    coefficients = jnp.einsum('rtm,tlm->rlm', kept, colatitude)
    return coefficients * basis.radial[:, None, None]


def _keep_orders(spectrum: jax.Array, bandwidth: int) -> jax.Array:
    """Take the orders below the bandwidth out of a longer FFT.

    In the FFT's layout the orders wanted are the first ``bandwidth`` and
    the last ``bandwidth - 1``, so this is two slices and never a gather.
    """
    length = spectrum.shape[-1]
    if length == 2 * bandwidth:
        return spectrum
    # Column `bandwidth` is the Nyquist slot, m = -bandwidth, which no
    # degree below the bandwidth reaches. It stays empty so that every
    # other order keeps the column the layout gives it.
    nyquist = jnp.zeros((*spectrum.shape[:-1], 1), dtype=spectrum.dtype)
    return jnp.concatenate(
        (
            spectrum[..., :bandwidth],
            nyquist,
            spectrum[..., length - bandwidth + 1 :],
        ),
        axis=-1,
    )
