"""Particles aligned to a reference, mirrors included or not."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import simulate
from frmony.engine import Search
from frmony.geometry import relative_angle
from frmony.io.maps import write_map
from frmony.particles import Subtomograms, load_reference
from frmony.reference import align_to_reference

SIZE = 32


def _particles(
    rng: np.random.Generator, tilt: float, noise: float, mirror: bool = False
):
    simulation = simulate(rng, 10, size=SIZE, tilt=tilt, noise=noise, shift=2.0)
    flip = np.zeros(10, dtype=bool)
    if mirror:
        flip[[1, 4, 7]] = True
    # A volume inverted through its centre has its spectrum conjugated;
    # its orientation is then minus the rotation, and its centre minus
    # the centre.
    data = np.where(
        flip[:, None, None, None], np.conj(simulation.data), simulation.data
    )
    orientations = np.where(
        flip[:, None, None], -simulation.rotations, simulation.rotations
    )
    centres = np.where(flip[:, None], -simulation.shifts, simulation.shifts)
    subtomograms = Subtomograms(
        data=data, weight=simulation.weight, pixel_size=4.0, box=SIZE
    )
    return simulation, subtomograms, orientations, centres, flip


def _reference(tmp_path: Path, simulation, subtomograms) -> Subtomograms:
    path = tmp_path / 'reference.mrc'
    write_map(path, simulation.structure, 4.0)
    return load_reference(str(path), subtomograms)


@pytest.mark.parametrize(
    ('tilt', 'noise'), [(60.0, 0.0), (60.0, 1.0), (90.0, 1.0)]
)
def test_particles_are_put_in_the_frame_of_the_reference(
    tmp_path: Path, rng: np.random.Generator, tilt: float, noise: float
) -> None:
    """Clean or noisy, with a wedge or without, and off centre.

    No global rotation is left free: the reference fixes the frame, so the
    orientations and centres are compared as they are. Measured at a
    bandwidth of sixteen, a step of 11.25 degrees: a median of 3.9 to 5.2
    degrees and at most 11.4, and centres within 0.86 of a voxel.
    """
    simulation, subtomograms, truth, centres, _ = _particles(rng, tilt, noise)
    reference = _reference(tmp_path, simulation, subtomograms)
    found = align_to_reference(
        subtomograms, reference, Search(bandwidth=16, max_shift=4.0)
    )
    errors = np.degrees(relative_angle(found.orientations, truth))
    assert np.median(errors) < 7.0
    assert errors.max() < 15.0
    assert np.linalg.norm(found.shifts - centres, axis=1).max() < 1.5
    assert np.all(np.linalg.det(found.orientations) > 0.0)


def test_mirror_images_are_found_when_asked_for(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """Three of ten particles inverted through their centres. Asked for,
    mirrors find them and put them where they are; not asked for, those
    three cannot be right. Measured with mirrors: 1.8 degrees at the
    median and 4.4 at worst."""
    simulation, subtomograms, truth, _, flip = _particles(
        rng, 60.0, 0.0, mirror=True
    )
    reference = _reference(tmp_path, simulation, subtomograms)
    found = align_to_reference(
        subtomograms,
        reference,
        Search(bandwidth=16, max_shift=4.0, mirrors=True),
    )
    mirrored = np.linalg.det(found.orientations) < 0.0
    np.testing.assert_array_equal(mirrored, flip)
    errors = np.degrees(relative_angle(found.orientations, truth))
    assert errors.max() < 15.0


def test_a_reference_must_be_the_size_of_the_box(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    _, subtomograms, *_ = _particles(rng, 60.0, 0.0)
    path = tmp_path / 'small.mrc'
    write_map(path, np.zeros((16, 16, 16)), 4.0)
    with pytest.raises(ValueError, match='Resample'):
        load_reference(str(path), subtomograms)


def test_mirrors_without_phases_are_refused() -> None:
    with pytest.raises(ValueError, match='phases'):
        Search(mirrors=True, phases=False)
