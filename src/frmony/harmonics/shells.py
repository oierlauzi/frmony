"""Reading a volume off concentric spheres.

Fast rotational matching works on the restriction of a volume to spheres
around its centre, because a rotation moves each sphere onto itself and so
acts on each one separately. This module does the reading: given a volume and
a set of radii, it samples each sphere on the grid the transform integrates
over, Gauss-Legendre in the colatitude and uniform around the azimuth.

The volumes here are Fourier transforms, centred so that the zero frequency
sits at ``shape // 2``. That is where the missing wedge lives and where the
weight of a pseudo-subtomogram is defined, and a correlation of amplitudes
there does not move when the particle does, which is what lets the rotation
be found before the shift is known.

Arrays are indexed ``[z, y, x]``, as the files store them, while directions
and rotations are ``(x, y, z)``. The one place the two meet is here.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._transfers import to_device
from .legendre import gauss_legendre

__all__ = [
    'band_radii',
    'sample_shells',
    'shell_directions',
    'shell_grid',
]


@functools.lru_cache(maxsize=16)
def shell_directions(nodes: int, azimuths: int) -> NDArray[np.float64]:
    """Unit vectors of the sampling grid, ``(nodes, azimuths, 3)``.

    The components are ``(x, y, z)``.
    """
    cosine, _ = gauss_legendre(nodes)
    sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
    azimuth = 2.0 * np.pi * np.arange(azimuths) / azimuths
    return np.stack(
        (
            sine[:, None] * np.cos(azimuth)[None, :],
            sine[:, None] * np.sin(azimuth)[None, :],
            np.broadcast_to(cosine[:, None], (nodes, azimuths)),
        ),
        axis=-1,
    )


@functools.lru_cache(maxsize=16)
def shell_grid(
    radii: tuple[float, ...], nodes: int, azimuths: int, size: int
) -> NDArray[np.float64]:
    """Where to read the volume, as indices into it.

    Args:
        radii: Radii of the spheres, in voxels.
        nodes: Gauss-Legendre nodes in the colatitude.
        azimuths: Samples around the azimuth.
        size: Extent of the volume along each axis.

    Returns:
        Indices ``(3, radii, nodes, azimuths)``, the leading axis ordered
        ``[z, y, x]`` as the volume is.
    """
    directions = shell_directions(nodes, azimuths)
    offsets = (
        np.asarray(radii, dtype=np.float64)[:, None, None, None]
        * (directions[None])
    )
    # (x, y, z) to the [z, y, x] the array is indexed by.
    return np.moveaxis(offsets[..., ::-1], -1, 0) + size // 2


def sample_shells(
    volume: ArrayLike,
    grid: ArrayLike,
) -> jax.Array:
    """Read a volume on the spheres of a grid.

    Linear interpolation, and zero beyond the edge of the volume. A complex
    volume is read in its two parts, since interpolation is linear and so
    commutes with taking them.

    Args:
        volume: A centred volume, ``(size, size, size)``, indexed
            ``[z, y, x]``. Real or complex.
        grid: Indices from `shell_grid`, ``(3, ...)``.

    Returns:
        The values at the grid, of shape ``grid.shape[1:]``.
    """
    volume = to_device(volume)
    grid = to_device(grid)
    if volume.ndim != 3:
        raise ValueError('A shell is read off a three-dimensional volume.')
    coordinates = [grid[axis] for axis in range(3)]

    def read(values: jax.Array) -> jax.Array:
        sampled: jax.Array = jax.scipy.ndimage.map_coordinates(
            values, coordinates, order=1, mode='constant', cval=0.0
        )
        return sampled

    if jnp.iscomplexobj(volume):
        return read(volume.real) + 1j * read(volume.imag)
    return read(volume)


def band_radii(
    size: int,
    lowest: float = 0.125,
    highest: float = 0.49,
    spacing: float = 1.0,
) -> NDArray[np.float64]:
    """Radii of the spheres to expand on, in voxels.

    Which spheres are used matters more than almost anything else here, and
    the ends of the range are both worth leaving out.

    The lowest few frequencies carry most of a particle's amplitude and
    almost none of its orientation: a sphere of radius two holds a handful
    of voxels, and what they say is mostly how much mass the particle has.
    Left in, they swamp the correlation. On simulated subtomograms with a
    sixty degree tilt range, expanding from radius one rather than from an
    eighth of the box turned a median error of three degrees into eight,
    and sent four pairs in ten past thirty degrees.

    The outermost sphere is the last that fits inside the box. Past it the
    samples run into the corners, where a sphere is no longer whole. It
    used to stop well short of that, at four tenths of the box, on top of
    the crop the loader had already made, so the search ended at a third
    of the sampling rate while being told four tenths. Those spheres are
    data. On EMPIAR-10304, taking them in raised the pairs whose best
    rotation was within fifteen degrees of a refinement from 33 in a
    hundred to 42, and to 49 with no crop at all.

    Args:
        size: Extent of the volume along each axis.
        lowest: Innermost radius, as a fraction of ``size``.
        highest: Outermost radius, as a fraction of ``size``.
        spacing: Gap between spheres, in voxels. One sphere per voxel of
            radius is the natural choice; wider trades accuracy for speed.

    Raises:
        ValueError: If the band is empty or runs outside the volume.
    """
    if not 0.0 <= lowest < highest <= 0.5:
        raise ValueError(
            f'The band from {lowest} to {highest} of the box is not a band '
            'inside it.'
        )
    if not spacing > 0.0:
        raise ValueError(f'The spacing must be positive, not {spacing}.')
    start = max(lowest * size, 1.0)
    radii = np.arange(start, highest * size, spacing)
    if len(radii) < 2:
        raise ValueError(
            f'A box of {size} voxels has no room for a band from {lowest} '
            f'to {highest} of it.'
        )
    return radii
