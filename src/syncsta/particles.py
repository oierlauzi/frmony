"""A set of subtomograms, loaded and ready to align.

Each particle is the pair ``(D, W)`` the matching works on: a centred
Fourier transform, cropped to the band that is actually used, and a
non-negative weight of the same shape saying how much of each frequency was
ever measured.

Cropping is what makes the set fit. A 96 voxel particle is seven megabytes
of spectrum; the band below four tenths of Nyquist is under one. Ten
thousand of those are about nine gigabytes, which a workstation holds, and
the spherical expansions that the rotational search actually runs on are
smaller again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io.volumes import prepare, soft_ball
from .io.wedge import cone_weight, relion_weight, wedge_weight

__all__ = ['Subtomograms', 'load_subtomograms']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Subtomograms:
    """Subtomograms in the form the matching takes them."""

    data: NDArray[np.complex64]
    """Centred spectra ``(particles, band, band, band)``."""
    weight: NDArray[np.float32]
    """Fourier weights, the same shape, non-negative."""
    pixel_size: float
    """Of the original box [A]."""
    box: int
    """Extent of the original box, before cropping."""

    def __len__(self) -> int:
        return len(self.data)

    @property
    def band(self) -> int:
        """Extent of the cropped spectra."""
        return int(self.data.shape[-1])

    @property
    def scale(self) -> float:
        """Voxels of the original box per voxel of the cropped one.

        A shift found in the cropped band is this many voxels of the box
        the particles were extracted in.
        """
        return self.box / self.band

    def __post_init__(self) -> None:
        if self.data.shape != self.weight.shape:
            raise ValueError(
                f'{len(self.data)} spectra of shape {self.data.shape[1:]} '
                f'do not go with weights of shape {self.weight.shape[1:]}.'
            )
        if len(self.data) < 2:
            raise ValueError(
                f'Aligning pairs needs at least two particles, not '
                f'{len(self.data)}.'
            )


def _read_volume(path: str) -> NDArray[np.float32]:
    """One volume, through vitrio."""
    import vitrio as vio

    location = (
        vio.ImageLocation.from_string(path) if '@' in path else Path(path)
    )
    return np.asarray(vio.read(location, data_type='float32'))


def load_subtomograms(
    data_paths: list[str],
    weight_paths: list[str] | None = None,
    pixel_size: float = 1.0,
    band: float = 0.4,
    tilt: float | None = None,
    cone: float | None = None,
    weight_transform: str = 'sqrt',
    mask_radius: float | None = None,
    batch: int = 32,
) -> Subtomograms:
    """Read subtomograms and put them into the form the matching takes.

    Args:
        data_paths: Where each particle's volume is.
        weight_paths: Where each particle's weight volume is, as RELION
            writes them. Without these a wedge or a cone must be named.
        pixel_size: Of the box [A].
        band: Highest frequency kept, as a fraction of the sampling rate.
            Everything above it is dropped, which is what keeps the set
            small enough to hold.
        tilt: Half the tilt range [deg], if the weight is to be
            synthesised as a missing wedge.
        cone: Half aperture [deg], for a missing cone instead.
        weight_transform: How to read a RELION weight; see
            `syncsta.io.wedge.relion_weight`.
        mask_radius: Radius of a soft real-space mask [voxels]. None
            leaves the volumes alone.
        batch: How many volumes to read at a time.

    Returns:
        The loaded `Subtomograms`.

    Raises:
        ValueError: If no weight is given in any form, if both a file and
            a synthesised shape are given, or if the volumes disagree in
            size.
    """
    if not 0.0 < band <= 0.5:
        raise ValueError(
            f'A band of {band} is not a fraction of the sampling rate below '
            'Nyquist.'
        )
    synthesised = sum(value is not None for value in (tilt, cone))
    if weight_paths is None and synthesised == 0:
        raise ValueError(
            'There is no weight for these particles. Give the weight '
            'volumes that RELION writes, or name a tilt range or a cone to '
            'build one from.'
        )
    if weight_paths is not None and synthesised:
        raise ValueError(
            'Give either the weight volumes or a shape to build one from, '
            'not both.'
        )
    if synthesised > 1:
        raise ValueError('A missing region is a wedge or a cone, not both.')
    if weight_paths is not None and len(weight_paths) != len(data_paths):
        raise ValueError('There must be one weight for each particle.')

    first = _read_volume(data_paths[0])
    if first.ndim != 3 or len(set(first.shape)) != 1:
        raise ValueError(
            f'{data_paths[0]} is not a cube; it has shape {first.shape}. A '
            'particle extracted as a stack of tilt images cannot be used.'
        )
    box = int(first.shape[-1])
    crop = max(2 * round(band * box), 4)
    crop = min(crop, box)
    _LOGGER.info(
        'Reading %d particles of %d voxels, keeping a band of %d.',
        len(data_paths),
        box,
        crop,
    )

    mask = None if mask_radius is None else soft_ball(box, mask_radius)
    if tilt is not None:
        shared: NDArray[np.float32] | None = wedge_weight(box, tilt)
    elif cone is not None:
        shared = cone_weight(box, cone)
    else:
        shared = None

    spectra = np.empty((len(data_paths), crop, crop, crop), dtype=np.complex64)
    weights = np.empty((len(data_paths), crop, crop, crop), dtype=np.float32)
    for start in range(0, len(data_paths), max(batch, 1)):
        stop = min(start + max(batch, 1), len(data_paths))
        for index in range(start, stop):
            volume = first if index == 0 else _read_volume(data_paths[index])
            if volume.shape != first.shape:
                raise ValueError(
                    f'{data_paths[index]} has shape {volume.shape}, while '
                    f'{data_paths[0]} has {first.shape}.'
                )
            if shared is not None:
                weight = shared
            else:
                assert weight_paths is not None
                weight = relion_weight(
                    _read_volume(weight_paths[index]), weight_transform
                )
                if weight.shape != volume.shape:
                    raise ValueError(
                        f'{weight_paths[index]} describes a {weight.shape} '
                        f'volume, not a {volume.shape} one.'
                    )
            spectra[index], weights[index] = prepare(
                volume, weight, crop=crop, mask=mask
            )

    return Subtomograms(
        data=spectra, weight=weights, pixel_size=pixel_size, box=box
    )
