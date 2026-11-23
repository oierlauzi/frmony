"""Particles handed to the device a block at a time."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import simulate
from frmony.blocks import block_size, per_particle
from frmony.geometry import relative_angle
from frmony.harmonics.shells import band_radii
from frmony.particles import Subtomograms
from frmony.pipeline import Design, run


def test_everything_fits_in_one_block() -> None:
    assert block_size(100, 1000, memory=1e9) == 100


def test_too_many_are_split_into_blocks_three_of_which_fit() -> None:
    # Half the memory is for the particles, and three blocks are on the
    # device at once: the one held, the one coming through, the next.
    size = block_size(100, 1000, memory=60_000)
    assert size == 10


def test_not_one_particle_fitting_is_refused() -> None:
    with pytest.raises(ValueError, match='narrower band'):
        block_size(10, 10**9, memory=1e6)


def _set(rng: np.random.Generator, count: int, size: int):
    simulation = simulate(rng, count, size=size, tilt=60.0, shift=1.0)
    return simulation, Subtomograms(
        data=simulation.data,
        weight=simulation.weight,
        pixel_size=4.0,
        box=size,
    )


def _memory_for(blocks_of: int, size: int, bandwidth: int) -> float:
    """Gigabytes in which three blocks of so many particles fit and no
    more: half of a device's memory is for particles."""
    each = per_particle(size, len(band_radii(size)), bandwidth)
    return 3 * blocks_of * each / 0.5 / 2**30


def test_blocks_change_nothing_but_rounding(
    rng: np.random.Generator,
) -> None:
    """One pass, a block at a time and with every particle held at once.

    Fifteen particles in blocks of four: four blocks, the last one short
    and padded, and pairs within a block and across two. The pairs are
    the same and so are their scores; the angles differ by rounding,
    because a program compiled for blocks of four rounds differently from
    one compiled for fifteen. Measured: scores within 2e-7 and angles
    within 0.025 degrees.
    """
    count, size, bandwidth = 15, 24, 8
    _, subtomograms = _set(rng, count, size)
    design = dict(partners=8, bandwidth=bandwidth, candidates=4, max_shift=2.0)
    whole = run(subtomograms, Design(**design))
    blocked = run(
        subtomograms,
        Design(memory=_memory_for(4, size, bandwidth), **design),
    )

    np.testing.assert_array_equal(blocked.left, whole.left)
    np.testing.assert_array_equal(blocked.right, whole.right)
    np.testing.assert_allclose(blocked.scores, whole.scores, atol=1e-5)
    turned = np.degrees(
        relative_angle(blocked.measurements, whole.measurements)
    )
    assert turned.max() < 0.1

    # Orientations are known only up to one global rotation, which the
    # synchronization settles anew each time, so what is compared is
    # what they say about each pair.
    def relative(found: np.ndarray) -> np.ndarray:
        return found[whole.left] @ np.swapaxes(found[whole.right], -2, -1)

    agreed = np.degrees(
        relative_angle(
            relative(blocked.orientations), relative(whole.orientations)
        )
    )
    assert agreed.max() < 0.5
    # The averages are made in those two frames too, so what is compared
    # is how far their halves agree. That is the same up to what
    # interpolating onto a grid turned one way rather than another
    # costs: measured, 0.013 at most.
    np.testing.assert_allclose(
        blocked.correlation, whole.correlation, atol=0.02
    )


def test_blocks_align_as_well_over_passes(rng: np.random.Generator) -> None:
    """Over two passes rounding can tip a pair whose two best rotations
    were all but tied, so the result is held to the truth instead."""
    count, size, bandwidth = 15, 24, 8
    simulation, subtomograms = _set(rng, count, size)
    design = dict(
        partners=8, bandwidth=bandwidth, candidates=4, max_shift=2.0, passes=2
    )
    errors = []
    for memory in (None, _memory_for(4, size, bandwidth)):
        result = run(subtomograms, Design(memory=memory, **design))
        relative = [
            relative_angle(
                result.orientations[i] @ result.orientations[j].T,
                simulation.rotations[i] @ simulation.rotations[j].T,
            )
            for i in range(count)
            for j in range(i + 1, count)
        ]
        errors.append(float(np.degrees(np.median(relative))))
    whole, blocked = errors
    assert blocked == pytest.approx(whole, abs=0.5)
