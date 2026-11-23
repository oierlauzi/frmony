"""The average at the size and pixel the particles were extracted at.

The alignment works on a cropped band, which is what makes ten thousand
particles fit on a device, and the average it makes along the way is that
band too. Once every particle has an orientation and a centre, nothing
stops the average being made from the particles as they were extracted:
the same weighted sum in Fourier space, over the whole box.

They are too many to hold whole, so they are read again a chunk at a time
and the sums carried from one chunk to the next. Nothing waits on
anything else. vitrio reads the next chunk while the device works on this
one, and everything the loader would do to a particle on the host -- its
transform, the unpacking of its weight, the whitening -- is done on the
device instead, so the host only hands the work over.

It also makes the shell correlation of the two halves worth more. The
alignment never saw the frequencies beyond its band, so out there the
halves could not have been made to agree by being aligned together.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._progress import progress
from ._transfers import to_device, to_host
from .averaging import (
    contribution,
    fourier_shell_correlation,
    ratio,
    resolution_at,
)
from .io.wedge import cone_weight, wedge_weight
from .matching.translation import frequency_grid
from .particles import _WHITEN_FLOOR, _location, _read_volume

__all__ = ['Reconstruction', 'chunk_for', 'reconstruct']

_LOGGER = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 30
"""About how much of the particles is read at once, volumes and weights
together. Two chunks are in flight: one on the device and one being
read."""


@dataclass(frozen=True)
class Reconstruction:
    """The average over the whole box, and how far its halves agree."""

    structure: NDArray[np.complex64]
    """The average, as a centred spectrum."""
    support: NDArray[np.float32]
    """The total weight behind each of its frequencies."""
    frequency: NDArray[np.float64]
    """Shells of the correlation, as fractions of the sampling rate."""
    correlation: NDArray[np.float64]
    """Shell correlation between the two halves."""
    resolution: float
    """Where that correlation falls below 0.143 [A]."""
    pixel_size: float
    """Of the average [A]."""


def chunk_for(box: int) -> int:
    """How many particles of a box to read at once."""
    per_particle = box**3 * 4 * 3  # a volume, and its weight file twice over
    return int(max(8, _CHUNK_BYTES // per_particle))


@functools.lru_cache(maxsize=8)
def _half_index(size: int) -> NDArray[np.int64]:
    """Where each voxel of a centred full spectrum is in a half one.

    `frmony.io.wedge.half_to_centred`, as a gather the device can do: the
    same arithmetic applied to the indices instead of the values.
    """
    half = size // 2 + 1
    z, y, x = np.meshgrid(
        np.arange(size), np.arange(size), np.arange(size), indexing='ij'
    )
    mirrored = x >= half
    zz = np.where(mirrored, (-z) % size, z)
    yy = np.where(mirrored, (-y) % size, y)
    xx = np.where(mirrored, (size - x) % half, x)
    flat = (zz * size + yy) * half + xx
    return np.asarray(np.fft.fftshift(flat), dtype=np.int64)


@functools.partial(jax.jit, static_argnames=('crop', 'transform'))
def _accumulate(
    sums: tuple[jax.Array, jax.Array],
    volumes: jax.Array,
    stored: jax.Array | None,
    shared: jax.Array | None,
    index: jax.Array,
    rotations: jax.Array,
    shifts: jax.Array,
    halves: jax.Array,
    grid: jax.Array,
    crop: int,
    transform: str,
) -> tuple[jax.Array, jax.Array]:
    """Prepare a chunk as the loader would, and add it to the sums.

    ``halves`` says which half each particle goes to, and padding goes to
    neither: its value is two.
    """
    box = volumes.shape[-1]
    start = box // 2 - crop // 2
    cut = slice(start, start + crop)
    middle = crop // 2
    axes = (-3, -2, -1)

    def one(
        carry: tuple[jax.Array, jax.Array], inputs: tuple[jax.Array, ...]
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        volume, raw, rotation, shift, half = inputs
        spectrum = jnp.fft.fftshift(
            jnp.fft.fftn(jnp.fft.ifftshift(volume, axes=axes), axes=axes),
            axes=axes,
        )[cut, cut, cut]
        spectrum = spectrum.at[middle, middle, middle].set(0.0)
        if shared is None:
            transfer = jnp.take(raw[:box].reshape(-1), index)
            if transform == 'sqrt':
                transfer = jnp.sqrt(jnp.maximum(transfer, 0.0))
            elif transform == 'none':
                transfer = (transfer > 0.0).astype(jnp.float32)
            else:
                transfer = jnp.maximum(transfer, 0.0)
        else:
            transfer = shared
        transfer = jnp.maximum(transfer[cut, cut, cut], 0.0)
        # `frmony.particles.whiten`, as the loader does it.
        measured = transfer > _WHITEN_FLOOR * jnp.max(transfer)
        amplitude = jnp.sqrt(jnp.where(measured, transfer, 1.0))
        data = jnp.where(measured, spectrum / amplitude, 0.0)
        weight = jnp.where(measured, amplitude, 0.0)
        share, support = contribution(data, weight, rotation, shift, grid)
        top, bottom = carry
        select = (jnp.arange(2) == half)[:, None, None, None]
        return (top + select * share, bottom + select * support), None

    raws = stored if stored is not None else jnp.zeros((len(volumes), 1))
    result, _ = jax.lax.scan(
        one, sums, (volumes, raws, rotations, shifts, halves)
    )
    return result


def reconstruct(
    data_paths: list[str],
    orientations: NDArray[np.float64],
    shifts: NDArray[np.float64],
    pixel_size: float,
    weight_paths: list[str] | None = None,
    weight_transform: str = 'identity',
    tilt: float | None = None,
    cone: float | None = None,
    chunk: int | None = None,
) -> Reconstruction:
    """Average the particles as they were extracted.

    Args:
        data_paths: Where each particle's volume is.
        orientations: The orientation of each ``(particles, 3, 3)``.
        shifts: Its centre ``(particles, 3)``, in voxels of the box it was
            extracted in, ordered ``(x, y, z)``, as
            `frmony.pipeline.Result.shifts` gives it.
        pixel_size: Of the box [A].
        weight_paths: The weights RELION wrote, if the particles have
            them; otherwise a ``tilt`` or a ``cone``, as for
            `frmony.particles.load_subtomograms`.
        weight_transform: How to read those weights.
        tilt: Half the tilt range [deg] of a wedge to build instead.
        cone: Half aperture [deg] of a cone to build instead.
        chunk: How many to read at once. `chunk_for` the box by default.

    Returns:
        A `Reconstruction`. The halves are the particles of even and of
        odd index, as in `frmony.pipeline.run`.
    """
    import vitrio as vio

    count = len(data_paths)
    if len(orientations) != count or len(shifts) != count:
        raise ValueError('There must be one orientation and centre a particle.')
    if (weight_paths is None) == (tilt is None and cone is None):
        raise ValueError('Give either the weight volumes or a shape for them.')

    first = _read_volume(data_paths[0])
    box = int(first.shape[-1])
    crop = 2 * (box // 2)
    stored_shape = (
        None if weight_paths is None else _read_volume(weight_paths[0]).shape
    )
    shared = None
    if tilt is not None:
        shared = to_device(wedge_weight(box, tilt))
    elif cone is not None:
        shared = to_device(cone_weight(box, cone))
    index = to_device(_half_index(box)) if stored_shape is not None else None
    grid = to_device(frequency_grid(crop))
    chunk = chunk or chunk_for(box)
    loader = vio.loader()

    def start(begin: int) -> tuple[Any, NDArray[np.float32], Any]:
        end = min(begin + chunk, count)
        volumes = np.zeros((chunk, *first.shape), dtype=np.float32)
        reads = [
            vio.read_batch_async(
                loader,
                volumes[: end - begin],
                [_location(p) for p in data_paths[begin:end]],
            )
        ]
        stored = None
        if stored_shape is not None:
            assert weight_paths is not None
            stored = np.zeros((chunk, *stored_shape), dtype=np.float32)
            reads.append(
                vio.read_batch_async(
                    loader,
                    stored[: end - begin],
                    [_location(p) for p in weight_paths[begin:end]],
                )
            )
        return reads, volumes, stored

    sums = (
        jnp.zeros((2, crop, crop, crop), dtype=jnp.complex64),
        jnp.zeros((2, crop, crop, crop), dtype=jnp.float32),
    )
    # A centre comes in voxels of the box; the spectra are a voxel short
    # of it if the box was odd.
    moves = shifts * (crop / box)
    begins = list(range(0, count, chunk))
    pending = start(0)
    for number in progress(
        range(len(begins)), 'Averaging particles', unit='particle', scale=chunk
    ):
        begin = begins[number]
        end = min(begin + chunk, count)
        reads, volumes, stored = pending
        for read in reads:
            read.get()
        # The next chunk is read while this one is on the device. Its
        # buffers are new: the ones handed over may still be in use.
        if number + 1 < len(begins):
            pending = start(begins[number + 1])
        halves = np.full(chunk, 2, dtype=np.int32)
        halves[: end - begin] = np.arange(begin, end) % 2
        rotations = np.zeros((chunk, 3, 3))
        rotations[:] = np.eye(3)
        rotations[: end - begin] = orientations[begin:end]
        centres = np.zeros((chunk, 3))
        centres[: end - begin] = moves[begin:end]
        sums = _accumulate(
            sums,
            to_device(volumes),
            None if stored is None else to_device(stored),
            shared,
            index if index is not None else to_device(np.zeros(1, np.int64)),
            to_device(rotations, dtype=jnp.float32),
            to_device(centres, dtype=jnp.float32),
            to_device(halves),
            grid,
            crop,
            weight_transform,
        )

    top, bottom = sums
    structure = ratio(top[0] + top[1], bottom[0] + bottom[1])
    frequency, correlation = fourier_shell_correlation(
        to_host(ratio(top[0], bottom[0])), to_host(ratio(top[1], bottom[1]))
    )
    resolution = resolution_at(frequency, correlation, pixel_size)
    _LOGGER.info(
        'Averaged %d particles at %.2f A, in a box of %d.',
        count,
        pixel_size,
        crop,
    )
    return Reconstruction(
        structure=to_host(structure),
        support=to_host(bottom[0] + bottom[1]),
        frequency=frequency,
        correlation=correlation,
        resolution=resolution,
        pixel_size=pixel_size,
    )
