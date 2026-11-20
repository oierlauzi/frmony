"""Reading subtomograms, and putting them into the form the matching wants.

Every particle becomes the pair ``(D, W)``: a centred Fourier transform and
a non-negative weight of the same shape. Three things happen on the way.

The transform is cropped to the band the matching actually uses. Frequencies
above it are not looked at, and carrying them costs memory everywhere:
cropping a 96 voxel box to the band below four tenths of Nyquist takes a
particle from seven megabytes to under one, which is the difference between
a cache that fits on a device and one that does not.

The zero frequency is cleared. A constrained correlation is supposed to
subtract the mean of each volume under the mask the two of them share, and
that mean is exactly this one voxel; since the shared weight at the zero
frequency does not depend on the rotation, clearing it once here settles the
subtraction for every rotation at once.

And a soft mask may be applied in real space first, which is worth knowing
about: multiplying in real space convolves in Fourier space, so a mask does
not leave ``D = W * S`` exactly true. A mask that only rounds off the
corners of the box costs little and takes away the parts of a subtomogram
that hold a neighbour rather than the particle.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ['crop_spectrum', 'prepare', 'soft_ball']


def soft_ball(
    size: int, radius: float | None = None, falloff: float = 0.1
) -> NDArray[np.float32]:
    """A ball with a raised-cosine edge, as a real-space mask.

    Args:
        size: Extent of the volume along each axis.
        radius: Where the mask reaches zero, in voxels. Half the box by
            default, which rounds off the corners and nothing else. The
            edge runs inward from here over ``falloff``, so the mask is
            one inside ``radius - falloff * size``.
        falloff: Width of the edge, as a fraction of ``size``.

    Returns:
        ``(size, size, size)``, one inside and zero outside.
    """
    if radius is None:
        radius = 0.5 * size
    if not radius > 0.0:
        raise ValueError(f'A radius is positive, not {radius}.')
    if not falloff >= 0.0:
        raise ValueError(f'A falloff is not negative, unlike {falloff}.')

    axis = np.arange(size) - size // 2
    distance = np.sqrt(
        axis[:, None, None] ** 2
        + axis[None, :, None] ** 2
        + axis[None, None, :] ** 2
    )
    width = max(falloff * size, 1e-6)
    ramp = np.clip((radius - distance) / width, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)


def crop_spectrum(
    spectrum: NDArray[np.complexfloating], size: int
) -> NDArray[np.complexfloating]:
    """Keep the middle of a centred spectrum, which is a low-pass filter.

    Args:
        spectrum: A centred spectrum ``(n, n, n)``.
        size: Extent to keep. No larger than the spectrum.

    Returns:
        ``(size, size, size)``, still centred.
    """
    extent = spectrum.shape[-1]
    if size > extent:
        raise ValueError(
            f'A band of {size} voxels does not fit in a box of {extent}.'
        )
    if size == extent:
        return spectrum
    start = extent // 2 - size // 2
    stop = start + size
    return spectrum[start:stop, start:stop, start:stop]


def prepare(
    volume: NDArray[np.floating],
    weight: NDArray[np.floating],
    crop: int | None = None,
    mask: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.complex64], NDArray[np.float32]]:
    """Turn a subtomogram and its weight into what the matching takes.

    Args:
        volume: A real subtomogram ``(n, n, n)``, indexed ``[z, y, x]``.
        weight: Its centred Fourier weight, the same shape.
        crop: Extent of the band to keep. The whole box by default.
        mask: A real-space mask to apply first, the shape of the volume.

    Returns:
        The centred spectrum, with its zero frequency cleared, and the
        weight, both cropped to the band.

    Raises:
        ValueError: If the volume is not a cube, or the weight is not its
            shape.
    """
    volume = np.asarray(volume, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(
            f'A subtomogram is a cube, unlike one of shape {volume.shape}.'
        )
    if weight.shape != volume.shape:
        raise ValueError(
            f'A weight of shape {weight.shape} does not go with a volume '
            f'of shape {volume.shape}.'
        )

    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != volume.shape:
            raise ValueError('A mask has the shape of the volume.')
        volume = volume * mask

    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))
    extent = volume.shape[-1]
    spectrum = crop_spectrum(spectrum, crop if crop is not None else extent)
    kept = crop_spectrum(
        weight.astype(np.complex128), crop if crop is not None else extent
    ).real

    middle = spectrum.shape[-1] // 2
    spectrum[middle, middle, middle] = 0.0
    return (
        spectrum.astype(np.complex64),
        np.maximum(kept, 0.0).astype(np.float32),
    )
