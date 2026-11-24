"""Working through more particles than a device holds, a block at a time.

The particles live on the host. A pass over them hands the device blocks
of them, sized from what the device has, and lets each block go once
nothing needs it any more. While the device works on one block, the next
is already on its way to it: handing a block over and preparing it are
dispatched without waiting, and JAX runs them beside the work in hand.

When every particle fits, there is one block, and nothing is handed over
twice.

Pairs are taken a pair of blocks at a time. Block ``i`` is held while
every block ``j >= i`` comes through, and the pairs between the two are
aligned. With ``T`` blocks, that hands over about ``T / 2`` times the
particles a pass, which over PCIe is seconds against the minutes the
alignment takes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._progress import progress
from ._transfers import to_device, to_host
from .averaging import accumulate, empty_sums, ratio
from .matching.pairwise import Alignment, Particles, chunk_of
from .matching.translation import frequency_grid

__all__ = [
    'Aligned',
    'BlockedAverage',
    'align_in_blocks',
    'average_in_blocks',
    'block_size',
    'device_memory',
    'pad_to',
    'per_particle',
    'spectra_on_device',
]

_LOGGER = logging.getLogger(__name__)

_SHARE = 0.5
"""Of a device's memory, how much the particles in it may take. The rest
is what the pairs in flight need: their grids of rotations and of shifts,
and the work of the transforms."""

_RESIDENT = 3
"""Blocks on the device at once: the one held, the one coming through it,
and the next, on its way."""


def per_particle(extent: int, shells: int, bandwidth: int) -> int:
    """Bytes one particle takes on the device while it is aligned.

    Its spectrum, its measured region, its weight while that is worked
    out, and the three expansions it is compared through.

    Args:
        extent: Of the band the particles were cropped to.
        shells: Spheres the expansions are made on.
        bandwidth: Of the expansions.
    """
    return extent**3 * (8 + 4 + 4) + 3 * shells * bandwidth * 2 * bandwidth * 8


def device_memory() -> int | None:
    """Bytes the default device can hold, if it says.

    A GPU says; the host, standing in as a device, does not, and is then
    taken to hold everything.
    """
    device = jnp.zeros(()).devices().pop()
    stats = device.memory_stats()
    if not stats:
        return None
    limit = stats.get('bytes_limit')
    return int(limit) if limit else None


def block_size(count: int, per_particle: int, memory: float | None) -> int:
    """How many particles a block holds.

    Args:
        count: How many particles there are.
        per_particle: Bytes one takes on the device.
        memory: Bytes the device has. None takes what it says it has,
            and everything if it says nothing.

    Returns:
        A block size, ``count`` itself when they all fit at once.
    """
    available = device_memory() if memory is None else memory
    if available is None:
        return count
    # All of them at once, if they fit: then one block is all there is,
    # and it is handed over once.
    if count * per_particle <= _SHARE * available:
        return count
    fits = int(_SHARE * available // (_RESIDENT * max(per_particle, 1)))
    if fits < 1:
        raise ValueError(
            f'Not one particle of {per_particle / 2**20:.1f} MB fits in '
            f'{available / 2**30:.1f} GB of device memory. Ask for a '
            'narrower band or a lower bandwidth.'
        )
    return min(count, fits)


def _blocks(count: int, size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + size, count)) for start in range(0, count, size)
    ]


class _Resident:
    """Blocks on the device, made when first wanted and let go when told.

    Making one dispatches its transfer and preparation and returns at
    once, which is what lets the next block travel while the current one
    is worked on.
    """

    def __init__(self, make: Callable[[int], Particles]) -> None:
        self._make = make
        self._held: dict[int, Particles] = {}

    def want(self, number: int) -> Particles:
        if number not in self._held:
            self._held[number] = self._make(number)
        return self._held[number]

    def keep_only(self, numbers: set[int]) -> None:
        for number in [n for n in self._held if n not in numbers]:
            del self._held[number]


@dataclass(frozen=True)
class Aligned:
    """What aligning every pair found, on the host, in the order of the
    pairs."""

    rotation: NDArray[np.float32]
    """``(pairs, 3, 3)``, taking each second particle onto its first."""
    shift: NDArray[np.float32]
    """``(pairs, 3)`` in voxels, ordered ``(x, y, z)``."""
    score: NDArray[np.float32]
    """``(pairs,)``."""


def align_in_blocks(
    count: int,
    size: int,
    make: Callable[[int, int], Particles],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    align: Callable[[Particles, Particles, jax.Array], Alignment],
    batch: int,
) -> Aligned:
    """Align every pair, holding no more than a few blocks at a time.

    Args:
        count: How many particles there are.
        size: How many a block holds; see `block_size`.
        make: Given a first and one past a last index, those particles on
            the device, ``size`` of them, padded at the end if fewer.
            It should dispatch and not wait.
        left: First particle of each pair.
        right: Second particle of each pair.
        align: `frmony.matching.pairwise.align_chunk` with everything but
            the two sets and the pairs given.
        batch: Pairs aligned at once, which chunks are a whole number of.

    Returns:
        An `Aligned`.
    """
    ranges = _blocks(count, size)
    block_of_left = np.asarray(left) // size
    block_of_right = np.asarray(right) // size
    lower = np.minimum(block_of_left, block_of_right)
    upper = np.maximum(block_of_left, block_of_right)
    steps = [
        (i, j)
        for i in range(len(ranges))
        for j in range(i, len(ranges))
        if np.any((lower == i) & (upper == j))
    ]
    if len(ranges) > 1:
        _LOGGER.info(
            'Holding %d particles of %d at a time: %d blocks, %d pairs of '
            'them.',
            size,
            count,
            len(ranges),
            len(steps),
        )

    chunk = chunk_of(len(left), batch)
    rotations = np.zeros((len(left), 3, 3), dtype=np.float32)
    shifts = np.zeros((len(left), 3), dtype=np.float32)
    scores = np.zeros(len(left), dtype=np.float32)
    total = sum(
        -(-int(np.sum((lower == i) & (upper == j))) // chunk) for i, j in steps
    )
    bar = progress(
        range(total), 'Aligning pairs', unit='pair', scale=chunk, total=total
    )
    resident = _Resident(lambda number: make(*ranges[number]))

    for position, (i, j) in enumerate(steps):
        first = resident.want(i)
        second = resident.want(j)
        # The next pair of blocks, on its way while these are worked on.
        if position + 1 < len(steps):
            for number in steps[position + 1]:
                resident.want(number)
            resident.keep_only({i, j, *steps[position + 1]})

        chosen = np.flatnonzero((lower == i) & (upper == j))
        local = np.stack(
            (
                _local(left[chosen], i, j, size),
                _local(right[chosen], i, j, size),
            ),
            axis=1,
        ).astype(np.int32)
        padding = -len(chosen) % chunk
        local = np.concatenate((local, np.repeat(local[-1:], padding, axis=0)))
        for start in range(0, len(local), chunk):
            piece = align(
                first, second, to_device(local[start : start + chunk])
            )
            # Waited on, or the bar would only say how much was dispatched.
            piece.score.block_until_ready()
            taken = chosen[start : start + chunk]
            kept = len(taken)
            rotations[taken] = to_host(piece.rotation)[:kept]
            shifts[taken] = to_host(piece.shift)[:kept]
            scores[taken] = to_host(piece.score)[:kept]
            bar.update(1)
    bar.close()
    return Aligned(rotation=rotations, shift=shifts, score=scores)


def _local(
    particles: NDArray[np.int64], i: int, j: int, size: int
) -> NDArray[np.int64]:
    """Where particles of blocks ``i`` and ``j`` sit in the two sets
    `frmony.matching.pairwise.align_chunk` reads as one: block ``i``
    first, then block ``j`` after it, unless the two are the same."""
    block = particles // size
    offset = particles - block * size
    return np.where((block == i) | (i == j), offset, size + offset)


@dataclass(frozen=True)
class BlockedAverage:
    """An average made a block at a time, and its two halves."""

    structure: NDArray[np.complex64]
    """The average of every particle, as a centred spectrum."""
    support: NDArray[np.float32]
    """The total weight behind each of its frequencies."""
    halves: tuple[NDArray[np.complex64], NDArray[np.complex64]]
    """The averages of the particles of even and of odd index."""


def average_in_blocks(
    count: int,
    size: int,
    make: Callable[[int, int], tuple[jax.Array, jax.Array]],
    rotations: NDArray[np.float64],
    shifts: NDArray[np.float64],
) -> BlockedAverage:
    """Average every particle, and each half, a block at a time.

    Args:
        count: How many particles there are.
        size: How many a block holds.
        make: Given a first and one past a last index, the spectra and
            weights of those particles on the device, ``size`` of them,
            padded with nothing at the end if fewer.
        rotations: The orientation of each ``(count, 3, 3)``.
        shifts: Its centre ``(count, 3)``, in voxels of the spectra.
    """
    ranges = _blocks(count, size)
    sums: tuple[jax.Array, jax.Array] | None = None
    pending = make(*ranges[0])
    grid: jax.Array | None = None
    for number, (start, stop) in enumerate(ranges):
        data, weight = pending
        if number + 1 < len(ranges):
            pending = make(*ranges[number + 1])
        extent = data.shape[-1]
        if sums is None:
            sums = empty_sums(extent)
            grid = to_device(frequency_grid(extent), dtype=jnp.float32)
        assert grid is not None
        filled = stop - start
        halves = np.full(size, 2, dtype=np.int32)
        halves[:filled] = np.arange(start, stop) % 2
        turned = np.tile(np.eye(3), (size, 1, 1))
        turned[:filled] = rotations[start:stop]
        moved = np.zeros((size, 3))
        moved[:filled] = shifts[start:stop]
        sums = accumulate(
            sums,
            data,
            weight,
            to_device(turned, dtype=jnp.float32),
            to_device(moved, dtype=jnp.float32),
            to_device(halves),
            grid,
        )
    assert sums is not None
    top, bottom = sums
    return BlockedAverage(
        structure=to_host(ratio(top[0] + top[1], bottom[0] + bottom[1])),
        support=to_host(bottom[0] + bottom[1]),
        halves=(
            to_host(ratio(top[0], bottom[0])),
            to_host(ratio(top[1], bottom[1])),
        ),
    )


def spectra_on_device(
    data: NDArray[np.complex64], weight: NDArray[np.float32], size: int
) -> Callable[[int, int], tuple[jax.Array, jax.Array]]:
    """What hands a block of spectra and weights to the device, to average,
    padded to ``size`` with nothing."""

    def make(start: int, stop: int) -> tuple[jax.Array, jax.Array]:
        return (
            to_device(pad_to(data[start:stop], size)),
            to_device(pad_to(weight[start:stop], size)),
        )

    return make


def pad_to(values: NDArray[np.generic], size: int) -> NDArray[np.generic]:
    """A block of particles padded with nothing to ``size`` of them."""
    if len(values) == size:
        return values
    padded = np.zeros((size, *values.shape[1:]), dtype=values.dtype)
    padded[: len(values)] = values
    return padded
