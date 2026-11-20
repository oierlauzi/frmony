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

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._transfers import to_device
from .io.volumes import prepare, soft_ball
from .io.wedge import cone_weight, relion_weight, wedge_weight

__all__ = ['Subtomograms', 'load_subtomograms', 'recentred', 'whiten']

_LOGGER = logging.getLogger(__name__)

_WHITEN_FLOOR = 1e-4
"""Below this fraction of a particle's largest transfer a frequency is
treated as unmeasured. It bounds the division in `whiten`: nothing is ever
divided by less than a hundredth of the largest amplitude, and what is
given up is a ten-thousandth of the transfer."""


@dataclass(frozen=True)
class Subtomograms:
    """Subtomograms in the form the matching takes them."""

    data: NDArray[np.complex64]
    """Centred spectra ``(particles, band, band, band)``."""
    weight: NDArray[np.float32]
    """Fourier weights, the same shape, non-negative. The amplitude of
    the transfer: a spectrum is its weight times the structure, plus
    white noise. `whiten` is what makes that so."""
    pixel_size: float
    """Of the original box [A]."""
    box: int
    """Extent of the original box, before cropping."""
    mask_radius: float | None = None
    """Radius of a soft real-space mask, in voxels of the original box.
    Kept and not applied: a mask belongs on the particle, and where the
    particle is in its box is one of the things being found. `recentred`
    applies it."""

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


def whiten(
    spectrum: NDArray[np.complex64], transfer: NDArray[np.float32]
) -> tuple[NDArray[np.complex64], NDArray[np.float32]]:
    """Put a particle into the form the matching takes, whatever it was.

    Every kind of input is read as one model. A spectrum ``D`` comes with
    a transfer ``W``, how much of each frequency it holds, and

        D = W S + n,    var n = W.

    A frequency that was measured ``W`` times over holds ``W`` times the
    signal and ``W`` times the variance of the noise. What comes out is

        D / sqrt(W) = sqrt(W) S + white noise,

    together with ``sqrt(W)`` as the weight, which is the pair that
    everything after the loader works on.

    For a subtomogram cropped from a tomogram the transfer is its wedge,
    one or zero. For a clean volume it is one everywhere. In both the
    square root changes nothing and neither does this function, so there
    is no branch on where a particle came from. It only does anything to
    a transfer that is uneven, which is RELION's: its pseudo-subtomograms
    are sums of tilts each multiplied by its own transfer, and the weight
    file beside them is that sum. Every tilt contains the tilt axis, so
    the transfer there is many times what it is elsewhere, and a
    correlation of the data as stored is decided by that line and the few
    lowest frequencies.

    The division is bounded. Frequencies whose transfer is under
    ``1e-4`` of the largest are dropped, so nothing is divided by less
    than a hundredth of the largest amplitude. And it is not a
    deconvolution, which would divide by ``W`` itself and amplify the
    noise where little was measured. This leaves the noise at one
    everywhere.

    On EMPIAR-10304, taking the share of pairs whose best rotation was
    within fifteen degrees of a refinement: 11 in a hundred for the data
    as stored, 37 for this.

    Args:
        spectrum: A centred spectrum as stored.
        transfer: How much of each frequency it holds, the same shape,
            non-negative.

    Returns:
        The whitened spectrum and its weight, the square root of the
        transfer, both zero wherever the transfer was too small to
        divide by.
    """
    transfer = np.maximum(transfer, 0.0)
    measured = transfer > _WHITEN_FLOOR * float(np.max(transfer, initial=0.0))
    amplitude = np.sqrt(np.where(measured, transfer, 1.0))
    return (
        np.where(measured, spectrum / amplitude, 0.0).astype(np.complex64),
        np.where(measured, amplitude, 0.0).astype(np.float32),
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
    weight_transform: str = 'identity',
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
        weight_transform: How to read a RELION weight as a transfer; see
            `frmony.io.wedge.relion_weight`. As it stands by default,
            which is what the file is.
        mask_radius: Radius of a soft real-space mask [voxels]. It is
            carried with the particles and applied by `recentred`, once
            for each centre they are believed to have. None leaves the
            volumes alone.
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
            spectra[index], weights[index] = whiten(
                *prepare(volume, weight, crop=crop)
            )

    return Subtomograms(
        data=spectra,
        weight=weights,
        pixel_size=pixel_size,
        box=box,
        mask_radius=mask_radius,
    )


def recentred(
    subtomograms: Subtomograms,
    centres: NDArray[np.float64] | None = None,
    batch: int = 32,
) -> tuple[jax.Array, jax.Array]:
    """The particles brought back to their centres and masked there.

    Each spectrum is taken back to what was stored, moved, masked in real
    space and whitened again, in that order: a mask multiplies a volume
    and so mixes the frequencies of its spectrum, which the whitening
    then has to see. With no centres and no mask nothing is done to
    them, and they cross to the device as they are.

    Args:
        subtomograms: What `load_subtomograms` built.
        centres: Where each particle is believed to be, ``(particles,
            3)`` in voxels of the band and ordered ``(x, y, z)``. Nothing
            if not given.
        batch: How many particles to work on at once.

    Returns:
        The spectra and their weights, on the device.
    """
    data = to_device(subtomograms.data)
    weight = to_device(subtomograms.weight)
    if subtomograms.mask_radius is None and centres is None:
        return data, weight

    size = subtomograms.band
    moves = to_device(
        np.zeros((len(subtomograms), 3)) if centres is None else centres,
        dtype=jnp.float32,
    )
    axis = (jnp.arange(size) - size // 2) / size
    middle = size // 2
    if subtomograms.mask_radius is None:
        mask = None
    else:
        mask = to_device(
            soft_ball(size, subtomograms.mask_radius / subtomograms.scale)
        )

    def one(inputs: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        spectrum, amplitude, move = inputs
        # exp(+2 pi i k . c) moves a volume back by c.
        ramp = jnp.exp(
            2j
            * jnp.pi
            * (
                axis[:, None, None] * move[2]
                + axis[None, :, None] * move[1]
                + axis[None, None, :] * move[0]
            )
        )
        stored = spectrum * amplitude * ramp
        if mask is not None:
            volume = jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(stored)))
            stored = jnp.fft.fftshift(
                jnp.fft.fftn(jnp.fft.ifftshift(volume.real * mask))
            )
            stored = stored.at[middle, middle, middle].set(0.0)
        measured = amplitude > 0.0
        return jnp.where(
            measured, stored / jnp.where(measured, amplitude, 1.0), 0.0
        ).astype(jnp.complex64)

    moved = jax.lax.map(one, (data, weight, moves), batch_size=max(batch, 1))
    return moved, weight
