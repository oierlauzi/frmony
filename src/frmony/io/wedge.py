"""Where a tomogram measured anything, as a weight in Fourier space.

A tomogram is built from tilts over a limited range, so a wedge of Fourier
space is never measured. What is measured is multiplied by the contrast
transfer function, which crosses zero at frequencies of its own. Both are
the same kind of fact about the data, and this package carries both as one
non-negative weight ``W``: how much of each frequency reached the
subtomogram.

For a plain subtomogram the weight is synthesised from the tilt range. For
a RELION pseudo-subtomogram it is read from the file RELION wrote beside
the particle, which needs unpicking: RELION stores two half spectra stacked
along z, in the layout FFTW uses rather than a centred one.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ['cone_weight', 'half_to_centred', 'relion_weight', 'wedge_weight']


def _frequencies(size: int) -> tuple[NDArray[np.float64], ...]:
    """Offsets from the centre of a centred volume, indexed ``[z, y, x]``."""
    axis = np.arange(size) - size // 2
    return (
        axis[:, None, None].astype(np.float64),
        axis[None, :, None].astype(np.float64),
        axis[None, None, :].astype(np.float64),
    )


def wedge_weight(size: int, tilt: float) -> NDArray[np.float32]:
    """A single-tilt missing wedge, one where the tomogram measured.

    The tilt axis is y and the beam is along z, which is what RELION and
    every tomography package this interoperates with assume.

    Args:
        size: Extent of the volume along each axis.
        tilt: Half the tilt range [deg]. A series from -60 to 60 degrees
            has a tilt of 60, and the wedge is what is left over.

    Returns:
        A centred weight ``(size, size, size)``, one inside the measured
        region and zero in the wedge.
    """
    if not 0.0 < tilt <= 90.0:
        raise ValueError(
            f'A tilt range of {tilt} degrees measures nothing, or measures '
            'everything.'
        )
    z, _, x = _frequencies(size)
    inside = np.abs(z) <= np.abs(x) * math.tan(math.radians(tilt))
    return np.broadcast_to(inside, (size,) * 3).astype(np.float32)


def cone_weight(size: int, aperture: float) -> NDArray[np.float32]:
    """A missing cone, which is what a tilt series about two axes leaves.

    Args:
        size: Extent of the volume along each axis.
        aperture: Half angle of the measured region about the equator
            [deg].
    """
    if not 0.0 < aperture <= 90.0:
        raise ValueError(
            f'An aperture of {aperture} degrees measures nothing, or '
            'measures everything.'
        )
    z, y, x = _frequencies(size)
    radial = np.sqrt(x * x + y * y)
    inside = np.abs(z) <= radial * math.tan(math.radians(aperture))
    return np.broadcast_to(inside, (size,) * 3).astype(np.float32)


def half_to_centred(half: NDArray[np.floating]) -> NDArray[np.float32]:
    """A real half spectrum in FFTW's layout, as a centred full one.

    FFTW keeps only the frequencies with a non-negative last component,
    the rest following from the symmetry of a real volume, and it puts the
    zero frequency at index zero rather than in the middle.

    Args:
        half: ``(size, size, size // 2 + 1)``, indexed ``[z, y, x]``.

    Returns:
        ``(size, size, size)``, centred.
    """
    half = np.asarray(half)
    size = half.shape[0]
    if half.shape != (size, size, size // 2 + 1):
        raise ValueError(
            f'A half spectrum of a {size} voxel volume has shape '
            f'{(size, size, size // 2 + 1)}, not {half.shape}.'
        )

    full = np.empty((size, size, size), dtype=np.float64)
    full[:, :, : size // 2 + 1] = half
    # The weight is a real, even function, so the missing half is the
    # mirror of the kept one through the origin.
    index = np.arange(size // 2 + 1, size)
    full[:, :, index] = half[
        np.ix_(
            (-np.arange(size)) % size,
            (-np.arange(size)) % size,
            (size - index) % (size // 2 + 1),
        )
    ]
    return np.fft.fftshift(full).astype(np.float32)


def relion_weight(
    volume: NDArray[np.floating], transform: str = 'sqrt'
) -> NDArray[np.float32]:
    """The Fourier weight of a RELION pseudo-subtomogram.

    RELION writes, beside each particle, one file holding two half spectra
    stacked along z: first the sum of the squared transfer over the tilts
    that saw the particle, and then the plain count of them. The first is
    what matters here.

    Per tilt, RELION multiplies the data by ``c = CTF * dose weighting``
    and accumulates ``c * c`` into that first slab. So the data carries one
    power of the transfer and the file carries two, which is why the
    default takes a square root: the weight wanted is the amplitude that
    reached the particle, not its square. For a single tilt that is exact,
    and for a sum it is close. The square root is safe -- the slab is a sum
    of squares -- and nothing anywhere divides by it.

    Args:
        volume: What the file holds, ``(2 * size, size, size // 2 + 1)``.
        transform: ``'sqrt'`` as above, ``'identity'`` to take the slab as
            it stands, or ``'none'`` to ignore it and weight every
            measured frequency alike.

    Returns:
        A centred weight ``(size, size, size)``.

    Raises:
        ValueError: If the file is not shaped as a pair of half spectra,
            which is what a 2D particle stack from RELION 5 would give, or
            if the transform is not one of the three.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3 or volume.shape[0] % 2:
        raise ValueError(
            f'A weight of shape {volume.shape} is not two half spectra '
            'stacked along z. RELION writes these only for three '
            'dimensional pseudo-subtomograms; a particle extracted as a '
            'stack of tilt images has none, and this package cannot use '
            'one. Extract the particles as 3D pseudo-subtomograms.'
        )
    size = volume.shape[0] // 2
    if volume.shape[1:] != (size, size // 2 + 1):
        raise ValueError(
            f'A weight of shape {volume.shape} is not two half spectra of '
            f'a {size} voxel volume.'
        )

    transfer = half_to_centred(volume[:size])
    if transform == 'sqrt':
        return np.sqrt(np.maximum(transfer, 0.0)).astype(np.float32)
    if transform == 'identity':
        return np.maximum(transfer, 0.0).astype(np.float32)
    if transform == 'none':
        return (transfer > 0.0).astype(np.float32)
    raise ValueError(
        f"A weight transform is 'sqrt', 'identity' or 'none', not "
        f'{transform!r}.'
    )
