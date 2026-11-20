"""The whole pipeline, on simulated subtomograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import simulate, wedge_mask
from frmony.averaging import (
    average,
    fourier_shell_correlation,
    resolution_at,
)
from frmony.geometry import relative_angle
from frmony.particles import Subtomograms, load_subtomograms
from frmony.pipeline import Design, Result, run


def _subtomograms(rng: np.random.Generator, count: int, **kwargs: object):
    simulation = simulate(rng, count, **kwargs)  # type: ignore[arg-type]
    size = simulation.data.shape[-1]
    return simulation, Subtomograms(
        data=simulation.data,
        weight=simulation.weight,
        pixel_size=4.0,
        box=size,
    )


def _pair_errors(result: Result, truth: np.ndarray) -> np.ndarray:
    count = len(truth)
    return np.degrees(
        [
            relative_angle(
                result.orientations[i] @ result.orientations[j].T,
                truth[i] @ truth[j].T,
            )
            for i in range(count)
            for j in range(i + 1, count)
        ]
    )


# --------------------------------------------------------------- averaging


def test_the_average_fills_the_missing_wedge(
    rng: np.random.Generator,
) -> None:
    """The point of averaging in Fourier space rather than in real space.

    Each particle is missing a wedge, but the wedge is fixed in its own
    tomogram and the particles point every way, so between them they cover
    nearly everything. A mean of the volumes would add the gaps together
    instead.
    """
    simulation, _ = _subtomograms(rng, 24, size=32, tilt=60.0)
    _, support = average(
        simulation.data,
        simulation.weight,
        simulation.rotations,
        np.zeros((24, 3)),
    )
    alone = float(simulation.weight[0].mean())
    together = float((np.asarray(support) > 0.0).mean())

    assert alone < 0.75
    assert together > 0.95


def test_the_average_is_the_structure(rng: np.random.Generator) -> None:
    simulation, _ = _subtomograms(rng, 24, size=32, tilt=60.0)
    structure, _ = average(
        simulation.data,
        simulation.weight,
        simulation.rotations,
        np.zeros((24, 3)),
    )
    truth = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(simulation.structure)))
    frequency, correlation = fourier_shell_correlation(
        np.asarray(structure), truth
    )
    # Skip the first shell: the zero frequency was cleared on purpose.
    assert np.all(correlation[1:8] > 0.9)
    assert len(frequency) == len(correlation)


def test_a_shell_correlation_of_noise_is_nothing(
    rng: np.random.Generator,
) -> None:
    size = 24
    left = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    right = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    _, correlation = fourier_shell_correlation(left, right)
    # Not the innermost shells: a sphere of radius one holds a handful of
    # voxels, and a handful of random numbers correlate by chance.
    assert np.abs(correlation[2:]).max() < 0.3


def test_a_shell_correlation_with_itself_is_one(
    rng: np.random.Generator,
) -> None:
    size = 16
    volume = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    _, correlation = fourier_shell_correlation(volume, volume)
    np.testing.assert_allclose(correlation, 1.0, atol=1e-9)


def test_an_empty_shell_has_no_correlation_and_is_passed_over() -> None:
    # The zero frequency is cleared on loading, so the innermost shell of
    # every real average is empty. Reporting zero there, which is below
    # any threshold, made the resolution of every run infinite. It was
    # found on the first real dataset, where a curve that crossed at 60 A
    # was reported as never having agreed at all.
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(16,) * 3) + 1j * rng.normal(size=(16,) * 3)
    volume[8, 8, 8] = 0.0
    frequency, correlation = fourier_shell_correlation(volume, volume)
    assert np.isnan(correlation[0])
    np.testing.assert_allclose(correlation[1:], 1.0, atol=1e-6)

    falling = np.where(np.arange(len(correlation)) < 5, correlation, 0.0)
    falling[0] = np.nan
    resolution = resolution_at(frequency, falling, 2.0)
    assert np.isfinite(resolution)
    assert 2.0 / frequency[5] < resolution < 2.0 / frequency[4]


def test_resolution_reads_the_crossing() -> None:
    frequency = np.array([0.1, 0.2, 0.3, 0.4])
    correlation = np.array([1.0, 1.0, 0.143, 0.0])
    assert resolution_at(frequency, correlation, 2.0) == pytest.approx(
        2.0 / 0.3, rel=1e-6
    )
    # Never crossing means the data ran out before the agreement did.
    assert resolution_at(frequency, np.ones(4), 2.0) == float('inf')


def test_mismatched_structures_are_refused() -> None:
    with pytest.raises(ValueError, match='same shape'):
        fourier_shell_correlation(np.zeros((8,) * 3), np.zeros((4,) * 3))


# ---------------------------------------------------------------- the whole


def test_the_pipeline_recovers_the_orientations(
    rng: np.random.Generator,
) -> None:
    """Pairs, synchronization, shifts and the average, end to end."""
    count = 20
    simulation, subtomograms = _subtomograms(rng, count, size=32, tilt=60.0)
    result = run(
        subtomograms,
        Design(partners=8, bandwidth=16, candidates=8, max_shift=3.0),
    )

    errors = _pair_errors(result, simulation.rotations)
    assert np.median(errors) < 10.0
    assert np.percentile(errors, 90) < 20.0

    assert result.orientations.shape == (count, 3, 3)
    np.testing.assert_allclose(
        np.linalg.det(result.orientations), 1.0, atol=1e-9
    )
    assert result.shifts.shape == (count, 3)
    assert result.structure.shape == subtomograms.data.shape[1:]
    assert result.kept.sum() > 0.8 * len(result.left)


def test_synchronizing_beats_the_pairs_it_is_given(
    rng: np.random.Generator,
) -> None:
    """The reason for the whole architecture.

    Each pairwise measurement is noisy; the orientations that agree best
    with all of them at once are better than any one of them.
    """
    count = 20
    simulation, subtomograms = _subtomograms(rng, count, size=32, tilt=60.0)
    result = run(
        subtomograms,
        Design(partners=10, bandwidth=16, candidates=8, max_shift=3.0),
    )

    raw = np.degrees(
        [
            relative_angle(
                measured,
                simulation.rotations[i] @ simulation.rotations[j].T,
            )
            for measured, i, j in zip(
                result.measurements, result.left, result.right, strict=True
            )
        ]
    )
    synchronized = np.degrees(
        [
            relative_angle(
                result.orientations[i] @ result.orientations[j].T,
                simulation.rotations[i] @ simulation.rotations[j].T,
            )
            for i, j in zip(result.left, result.right, strict=True)
        ]
    )
    assert np.median(synchronized) < np.median(raw)


def test_an_unconnected_graph_is_refused(
    rng: np.random.Generator,
) -> None:
    _, subtomograms = _subtomograms(rng, 4, size=16, tilt=60.0)
    with pytest.raises(ValueError, match='partner'):
        run(subtomograms, Design(partners=0))


def test_a_design_refuses_nonsense() -> None:
    with pytest.raises(ValueError, match='partner'):
        Design(partners=0)
    with pytest.raises(ValueError, match='inside the box'):
        Design(lowest=0.7)


def test_too_few_particles_is_refused(rng: np.random.Generator) -> None:
    simulation, _ = _subtomograms(rng, 2, size=16, tilt=60.0)
    with pytest.raises(ValueError, match='at least two'):
        Subtomograms(
            data=simulation.data[:1],
            weight=simulation.weight[:1],
            pixel_size=1.0,
            box=16,
        )


def test_mismatched_weights_are_refused(rng: np.random.Generator) -> None:
    simulation, _ = _subtomograms(rng, 3, size=16, tilt=60.0)
    with pytest.raises(ValueError, match='do not go with'):
        Subtomograms(
            data=simulation.data,
            weight=simulation.weight[:, :8, :8, :8],
            pixel_size=1.0,
            box=16,
        )


# -------------------------------------------------------------- the loader


def _relion_project(
    directory: Path, rng: np.random.Generator, count: int
) -> tuple[Path, np.ndarray]:
    """A RELION extraction job, as it would be left on disk."""
    import pandas as pd
    import starfile

    from frmony.io.maps import write_map

    size = 24
    simulation = simulate(rng, count, size=size, tilt=60.0)
    wedge = wedge_mask(size, 60.0)
    half = np.fft.ifftshift(wedge)[:, :, : size // 2 + 1]
    stored = np.concatenate((half, np.ones_like(half)), axis=0)

    names, weights = [], []
    for index in range(count):
        volume = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(simulation.data[index]))
        ).real
        data_path = directory / f'p{index:03d}_data.mrc'
        weight_path = directory / f'p{index:03d}_weights.mrc'
        write_map(data_path, volume, 4.0)
        write_map(weight_path, stored, 4.0)
        names.append(data_path.name)
        weights.append(weight_path.name)

    table = pd.DataFrame(
        {
            'rlnImageName': names,
            'rlnCtfImage': weights,
            'rlnOpticsGroup': [1] * count,
            'rlnCenteredCoordinateXAngst': rng.normal(size=count),
        }
    )
    optics = pd.DataFrame({'rlnOpticsGroup': [1], 'rlnImagePixelSize': [4.0]})
    path = directory / 'particles.star'
    starfile.write({'optics': optics, 'particles': table}, path, overwrite=True)
    return path, simulation.rotations


def test_a_relion_project_loads(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    path, _ = _relion_project(tmp_path, rng, 5)
    from frmony.io.registry import read_source

    source = read_source(path)
    assert source.weights is not None
    subtomograms = load_subtomograms(
        source.data,
        weight_paths=source.weights,
        pixel_size=source.pixel_size or 1.0,
        band=0.4,
    )
    assert len(subtomograms) == 5
    assert subtomograms.box == 24
    assert subtomograms.band <= 24
    assert subtomograms.scale >= 1.0


def test_the_loader_refuses_a_set_with_no_weight(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    path, _ = _relion_project(tmp_path, rng, 3)
    from frmony.io.registry import read_source

    source = read_source(path)
    with pytest.raises(ValueError, match='no weight'):
        load_subtomograms(source.data, pixel_size=4.0)
    with pytest.raises(ValueError, match='not both'):
        load_subtomograms(source.data, weight_paths=source.weights, tilt=60.0)
    with pytest.raises(ValueError, match='wedge or a cone'):
        load_subtomograms(source.data, tilt=60.0, cone=60.0)


@pytest.mark.slow
def test_the_command_line_runs(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """The whole thing as someone would actually invoke it."""
    from frmony.cli import main

    path, _ = _relion_project(tmp_path, rng, 12)
    output = tmp_path / 'run'
    status = main(
        [
            'run',
            str(path),
            str(output),
            '--partners',
            '6',
            '--bandwidth',
            '12',
            '--candidates',
            '6',
            '--max-shift',
            '2',
            '--batch',
            '2',
        ]
    )
    assert status == 0
    assert (output / 'average.mrc').exists()
    assert (output / 'support.mrc').exists()
    assert (output / 'synchronized.star').exists()
    assert (output / 'resolution.txt').exists()

    import starfile

    written = starfile.read(output / 'synchronized.star', always_dict=True)
    particles = written['particles']
    assert len(particles) == 12
    for column in ('rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'):
        assert column in particles.columns
    assert 'rlnOriginXAngst' in particles.columns
    # What came in survived.
    assert 'rlnCenteredCoordinateXAngst' in particles.columns
