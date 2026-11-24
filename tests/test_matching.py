"""Peaks, shifts, the constrained score, and aligning a pair end to end."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import simulate, spectrum
from frmony.geometry import euler_to_matrix, relative_angle
from frmony.harmonics.shells import band_radii
from frmony.harmonics.transform import make_basis
from frmony.matching.correlation import expand_particle, rotation_score
from frmony.matching.pairwise import (
    Joint,
    Settings,
    align_pair,
    align_pairs,
    candidate_rotations,
    lattice_spacing,
    shift_lattice,
    truncate,
)
from frmony.matching.peaks import refine_peak, top_peaks
from frmony.matching.rotation import grid_to_euler
from frmony.matching.translation import (
    frequency_grid,
    rotate_spectrum,
    shift_correlation,
    shift_of_index,
)

SIZE = 32
BANDWIDTH = 24


def _basis() -> object:
    return make_basis(
        band_radii(SIZE), BANDWIDTH, SIZE, sampling_bandwidth=2 * BANDWIDTH
    )


# --------------------------------------------------------------------- peaks


def test_top_peaks_finds_and_separates() -> None:
    grid = np.zeros((8, 8, 8), dtype=np.float32)
    grid[2, 3, 4] = 5.0
    grid[2, 3, 5] = 4.5  # adjacent to the first, so suppressed
    grid[6, 1, 1] = 3.0

    indices, values = top_peaks(grid, 2, separation=1)
    assert np.asarray(indices)[0].tolist() == [2, 3, 4]
    assert np.asarray(indices)[1].tolist() == [6, 1, 1]
    np.testing.assert_allclose(np.asarray(values), [5.0, 3.0])


def test_top_peaks_suppression_wraps() -> None:
    """The angular axes of a rotational grid are periodic."""
    grid = np.zeros((8, 8, 8), dtype=np.float32)
    grid[0, 0, 0] = 5.0
    grid[7, 0, 0] = 4.5  # next to the first, the other way round

    indices, _ = top_peaks(grid, 2, separation=1)
    assert np.asarray(indices)[1].tolist() != [7, 0, 0]


def test_refine_peak_beats_the_grid() -> None:
    """A parabola through three samples recovers most of the step."""
    truth = np.array([2.3, 3.7, 4.1])
    axis = np.arange(8)
    squared = sum(
        (axis.reshape([-1 if k == a else 1 for k in range(3)]) - truth[a]) ** 2
        for a in range(3)
    )
    bump = np.exp(-squared / 2.0).astype(np.float32)

    coarse = np.array(np.unravel_index(int(bump.argmax()), bump.shape))
    position, height = refine_peak(bump, coarse)

    assert np.abs(np.asarray(position) - truth).max() < 0.1
    assert np.abs(coarse - truth).max() > 0.2  # the grid alone was worse
    assert float(height) >= float(bump.max())


def test_refine_peak_stays_put_on_a_trough() -> None:
    """An upward curvature is not a peak, so do not extrapolate off it."""
    grid = np.zeros((5, 5, 5), dtype=np.float32)
    grid[2, 2, 2] = -1.0
    position, _ = refine_peak(grid, np.array([2, 2, 2]))
    np.testing.assert_allclose(np.asarray(position), [2.0, 2.0, 2.0])


@pytest.mark.parametrize('count', [0, -1])
def test_top_peaks_refuses_nothing(count: int) -> None:
    with pytest.raises(ValueError, match='peak'):
        top_peaks(np.zeros((4, 4, 4)), count)


def test_peaks_need_a_volume() -> None:
    with pytest.raises(ValueError, match='three-dimensional'):
        top_peaks(np.zeros((4, 4)), 1)
    with pytest.raises(ValueError, match='three-dimensional'):
        refine_peak(np.zeros((4, 4)), np.zeros(3, dtype=int))


# --------------------------------------------------------------- translation


def test_rotate_spectrum_leaves_the_identity_alone(
    rng: np.random.Generator,
) -> None:
    volume = rng.normal(size=(16, 16, 16)).astype(np.float32)
    turned = rotate_spectrum(volume, np.eye(3), frequency_grid(16))
    np.testing.assert_allclose(np.asarray(turned), volume, atol=1e-5)


def test_rotate_spectrum_turns_the_right_way(
    rng: np.random.Generator,
) -> None:
    """A quarter turn about z must move the x axis onto y."""
    size = 17
    volume = np.zeros((size,) * 3, dtype=np.float32)
    centre = size // 2
    volume[centre, centre, centre + 4] = 1.0  # at +x

    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = np.asarray(rotate_spectrum(volume, quarter, frequency_grid(size)))
    peak = np.unravel_index(int(turned.argmax()), turned.shape)
    assert peak == (centre, centre + 4, centre)  # now at +y


@pytest.mark.parametrize('roll', [(0, 0, 0), (2, -3, 4), (-5, 1, 2)])
def test_an_integer_shift_is_recovered(
    rng: np.random.Generator, roll: tuple[int, int, int]
) -> None:
    size = 24
    axis = np.arange(size) - size // 2
    squared = sum(
        axis.reshape([-1 if k == a else 1 for k in range(3)]) ** 2
        for a in range(3)
    )
    volume = np.exp(-squared / 8.0).astype(np.float32)
    volume = volume + 0.4 * np.roll(volume, (3, -2, 5), axis=(0, 1, 2))

    moved = np.roll(volume, roll, axis=(0, 1, 2))
    weight = np.ones((size,) * 3, dtype=np.float32)
    values, _ = shift_correlation(
        spectrum(volume).astype(np.complex64),
        weight,
        spectrum(moved).astype(np.complex64),
        weight,
        np.eye(3),
        frequency_grid(size),
    )
    values = np.asarray(values)
    peak = np.unravel_index(int(values.argmax()), values.shape)
    found = np.asarray(shift_of_index(np.array(peak), size))

    # The shift brings the second volume back onto the first, so it is the
    # opposite of the roll, and in (x, y, z) rather than [z, y, x].
    expected = -np.array(roll)[::-1]
    np.testing.assert_array_equal(found, expected)
    assert values.max() == pytest.approx(1.0, abs=1e-4)


def test_the_shift_score_is_bounded(rng: np.random.Generator) -> None:
    size = 16
    left = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    right = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    weight = rng.uniform(0.0, 1.0, size=(size,) * 3).astype(np.float32)

    values, _ = shift_correlation(
        left.astype(np.complex64),
        weight,
        right.astype(np.complex64),
        weight,
        np.eye(3),
        frequency_grid(size),
    )
    assert np.abs(np.asarray(values)).max() <= 1.0 + 1e-5


def test_no_shared_frequency_gives_a_number_not_a_nan() -> None:
    """A wide wedge and an unlucky rotation can leave nothing in common."""
    size = 12
    data = np.ones((size,) * 3, dtype=np.complex64)
    weight = np.zeros((size,) * 3, dtype=np.float32)

    values, denominator = shift_correlation(
        data, weight, data, weight, np.eye(3), frequency_grid(size)
    )
    assert np.isfinite(np.asarray(values)).all()
    assert float(denominator) > 0.0


# ---------------------------------------------------------------- the score


def test_the_score_is_bounded(rng: np.random.Generator) -> None:
    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    basis = _basis()
    left = expand_particle(simulation.data[0], simulation.weight[0], basis)
    right = expand_particle(simulation.data[1], simulation.weight[1], basis)

    grid = np.asarray(rotation_score(left, right, BANDWIDTH))
    assert np.isfinite(grid).all()
    assert np.abs(grid).max() <= 1.0 + 1e-3


def test_the_score_ignores_the_contrast_of_a_particle(
    rng: np.random.Generator,
) -> None:
    """Scaling one particle must not move the answer.

    Tomograms differ in contrast, and a score that noticed would rank pairs
    by how bright they are rather than by how well they match.
    """
    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    basis = _basis()
    left = expand_particle(simulation.data[0], simulation.weight[0], basis)
    right = expand_particle(simulation.data[1], simulation.weight[1], basis)
    louder = expand_particle(
        7.5 * simulation.data[1], simulation.weight[1], basis
    )

    plain = np.asarray(rotation_score(left, right, BANDWIDTH))
    scaled = np.asarray(rotation_score(left, louder, BANDWIDTH))
    np.testing.assert_allclose(plain, scaled, atol=2e-3)


def test_a_particle_matches_itself_at_the_identity(
    rng: np.random.Generator,
) -> None:
    simulation = simulate(rng, 1, size=SIZE, tilt=60.0)
    basis = _basis()
    own = expand_particle(simulation.data[0], simulation.weight[0], basis)

    grid = np.asarray(rotation_score(own, own, BANDWIDTH))
    peak = np.unravel_index(int(grid.argmax()), grid.shape)

    # Asserted on the rotation, not the index: where the tilt is zero only
    # the sum of the other two angles is determined, so the identity has a
    # whole ray of indices standing for it.
    found = euler_to_matrix(*grid_to_euler(np.array(peak), BANDWIDTH))
    assert np.degrees(relative_angle(np.eye(3), found)) < 1e-6

    # High, but not exactly one. The three correlations the score is built
    # from band-limit three functions of different bandwidths -- |D| W,
    # |D|^2 W and W -- so the normalization is not quite exact. It is a
    # ratio of truncated quantities, and what matters is that it peaks in
    # the right place and stays inside its bounds.
    assert 0.7 < grid[peak] <= 1.0


# ------------------------------------------------------------------- a pair


def test_a_pair_of_subtomograms_is_aligned(rng: np.random.Generator) -> None:
    """The whole engine, on simulated subtomograms with a missing wedge.

    The wedge is fixed in the tomogram and so does not turn with the
    particle, which is the difficulty the constrained score exists for.
    """
    count = 5
    simulation = simulate(rng, count, size=SIZE, tilt=60.0)
    basis = _basis()
    expansions = [
        expand_particle(simulation.data[i], simulation.weight[i], basis)
        for i in range(count)
    ]
    settings = Settings(bandwidth=BANDWIDTH, candidates=10, max_shift=3.0)

    errors = []
    for first in range(count):
        for second in range(first + 1, count):
            found = align_pair(
                simulation.data[first],
                simulation.weight[first],
                simulation.data[second],
                simulation.weight[second],
                expansions[first],
                expansions[second],
                settings,
            )
            truth = simulation.rotations[first] @ simulation.rotations[second].T
            errors.append(
                np.degrees(relative_angle(truth, np.asarray(found.rotation)))
            )
            assert 0.5 < float(found.score) <= 1.0

    errors = np.array(errors)
    # Measured on this simulation with a sixty degree tilt range. These are
    # measurements a synchronization then averages over, not final
    # orientations, so the median is what matters most.
    assert np.median(errors) < 8.0
    assert np.percentile(errors, 80) < 12.0


def test_phases_find_what_amplitudes_are_too_noisy_to(
    rng: np.random.Generator,
) -> None:
    """Under noise, the first stage has to keep the phases.

    A product of two spectra has an expectation that goes as the signal
    squared. A product of two amplitudes of mostly noise goes as its
    fourth power, so it falls away far sooner. This is the simulation of
    what the first real dataset showed, where the amplitudes aligned
    three pairs in a thousand and the phases three in ten.

    Measured here, with noise at twice the signal, on two seeds: the
    amplitudes had a median error of 118 and 136 degrees, which is what
    rotations drawn at random give, and the phases 13 and 21.
    """
    count = 6
    simulation = simulate(rng, count, size=SIZE, tilt=60.0, noise=2.0)
    basis = _basis()
    settings = Settings(bandwidth=BANDWIDTH, candidates=10, max_shift=3.0)

    medians = {}
    for phases in (False, True):
        expansions = [
            expand_particle(
                simulation.data[i], simulation.weight[i], basis, phases
            )
            for i in range(count)
        ]
        errors = []
        for first in range(count):
            for second in range(first + 1, count):
                found = align_pair(
                    simulation.data[first],
                    simulation.weight[first],
                    simulation.data[second],
                    simulation.weight[second],
                    expansions[first],
                    expansions[second],
                    settings,
                )
                truth = (
                    simulation.rotations[first] @ simulation.rotations[second].T
                )
                errors.append(
                    np.degrees(
                        relative_angle(truth, np.asarray(found.rotation))
                    )
                )
        medians[phases] = float(np.median(errors))

    assert medians[True] < 30.0
    assert medians[False] > 60.0


def test_the_candidates_contain_the_answer(rng: np.random.Generator) -> None:
    """Keeping several rotations is what makes the engine work.

    Amplitudes alone discard the phases, so the best rotation of the first
    stage is often not the right one, but the right one is among the few
    best.
    """
    simulation = simulate(rng, 4, size=SIZE, tilt=60.0)
    basis = _basis()
    expansions = [
        expand_particle(simulation.data[i], simulation.weight[i], basis)
        for i in range(4)
    ]
    settings = Settings(bandwidth=BANDWIDTH, candidates=10)

    for first, second in ((0, 1), (0, 2), (1, 3), (2, 3)):
        matrices, _ = candidate_rotations(
            expansions[first], expansions[second], settings
        )
        truth = simulation.rotations[first] @ simulation.rotations[second].T
        errors = np.degrees(
            [relative_angle(truth, np.asarray(m)) for m in matrices]
        )
        # Better than the step of the grid, which interpolation earns.
        assert errors.min() < 180.0 / BANDWIDTH


def test_the_candidates_are_distinct(rng: np.random.Generator) -> None:
    """The grid holds every rotation twice, and returning both halves the
    number of rotations actually tried."""
    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    basis = _basis()
    left = expand_particle(simulation.data[0], simulation.weight[0], basis)
    right = expand_particle(simulation.data[1], simulation.weight[1], basis)

    matrices = np.asarray(
        candidate_rotations(
            left, right, Settings(bandwidth=BANDWIDTH, candidates=8)
        )[0]
    )
    spread = np.degrees(
        [
            relative_angle(matrices[i], matrices[j])
            for i in range(len(matrices))
            for j in range(i + 1, len(matrices))
        ]
    )
    assert spread.min() > 1.0


def test_a_shift_is_recovered_alongside_the_rotation(
    rng: np.random.Generator,
) -> None:
    simulation = simulate(rng, 3, size=SIZE, tilt=60.0, shift=2.0)
    basis = _basis()
    expansions = [
        expand_particle(simulation.data[i], simulation.weight[i], basis)
        for i in range(3)
    ]
    settings = Settings(bandwidth=BANDWIDTH, candidates=10, max_shift=6.0)

    found = align_pair(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[1],
        simulation.weight[1],
        expansions[0],
        expansions[1],
        settings,
    )
    assert np.all(np.abs(np.asarray(found.shift)) <= 6.0)
    assert float(found.score) > 0.5


def test_many_pairs_align_in_a_batch(rng: np.random.Generator) -> None:
    count = 4
    simulation = simulate(rng, count, size=SIZE, tilt=60.0)
    basis = _basis()
    expansions = [
        expand_particle(simulation.data[i], simulation.weight[i], basis)
        for i in range(count)
    ]
    import jax

    batched = jax.tree.map(lambda *leaves: np.stack(leaves), *expansions)
    left = np.array([0, 0, 1])
    right = np.array([1, 2, 2])

    found = align_pairs(
        simulation.data,
        simulation.weight,
        batched,
        left,
        right,
        Settings(bandwidth=BANDWIDTH, candidates=6, max_shift=3.0),
        batch=2,
    )
    assert np.asarray(found.rotation).shape == (3, 3, 3)
    assert np.asarray(found.shift).shape == (3, 3)

    one = align_pair(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[1],
        simulation.weight[1],
        expansions[0],
        expansions[1],
        Settings(bandwidth=BANDWIDTH, candidates=6, max_shift=3.0),
    )
    np.testing.assert_allclose(
        np.asarray(found.rotation)[0], np.asarray(one.rotation), atol=1e-5
    )


def test_settings_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match='bandwidth'):
        Settings(bandwidth=1)
    with pytest.raises(ValueError, match='candidate'):
        Settings(candidates=0)
    with pytest.raises(ValueError, match='shift'):
        Settings(max_shift=-1.0)


def test_the_alignment_satisfies_the_relation(
    rng: np.random.Generator,
) -> None:
    """What comes out is what the synchronization expects of a pair.

    Checked as a statement about the volumes: turning and shifting the
    second by what was found should reproduce the first.
    """
    from conftest import rotate_volume

    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    basis = _basis()
    expansions = [
        expand_particle(simulation.data[i], simulation.weight[i], basis)
        for i in range(2)
    ]
    found = align_pair(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[1],
        simulation.weight[1],
        expansions[0],
        expansions[1],
        Settings(bandwidth=BANDWIDTH, candidates=10, max_shift=3.0),
    )

    first = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(simulation.data[0]))
    ).real
    second = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(simulation.data[1]))
    ).real
    brought = rotate_volume(second, np.asarray(found.rotation))

    straight = np.corrcoef(first.ravel(), second.ravel())[0, 1]
    turned = np.corrcoef(first.ravel(), brought.ravel())[0, 1]
    assert turned > straight
    assert turned > 0.5


# ------------------------------------------------ rotations and shifts at once


def test_a_lattice_of_shifts_fills_its_ball() -> None:
    reach, spacing = 5.0, 1.5
    lattice = shift_lattice(reach, spacing)

    np.testing.assert_array_equal(lattice[0], 0.0)
    assert np.linalg.norm(lattice, axis=1).max() <= reach + 1e-9
    assert len(np.unique(np.round(lattice, 6), axis=0)) == len(lattice)

    # A body-centred lattice of edge h leaves no point further than
    # sqrt(5) / 4 of h from one of its own. Checked away from the edge of
    # the ball, where the lattice has been cut.
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3))
    points *= (
        (reach - spacing)
        * rng.uniform(0, 1, (500, 1)) ** (1 / 3)
        / np.linalg.norm(points, axis=1, keepdims=True)
    )
    nearest = np.linalg.norm(points[:, None] - lattice[None], axis=-1).min(
        axis=1
    )
    assert nearest.max() <= np.sqrt(5.0) / 4.0 * spacing + 1e-9


def test_no_reach_is_one_trial() -> None:
    np.testing.assert_array_equal(shift_lattice(0.0, 1.5), np.zeros((1, 3)))
    with pytest.raises(ValueError, match='spacing'):
        shift_lattice(3.0, 0.0)


def test_truncating_is_expanding_at_a_lower_bandwidth(
    rng: np.random.Generator,
) -> None:
    # The degrees below the lower bandwidth are the same numbers at
    # either, as long as the spheres are sampled alike, so the joint
    # search can cut what it needs out of the expansions it has.
    from frmony.harmonics.transform import expand

    simulation = simulate(rng, 1, size=SIZE, tilt=60.0)
    radii = band_radii(SIZE)
    high = make_basis(radii, 16, SIZE, sampling_bandwidth=32)
    low = make_basis(radii, 8, SIZE, sampling_bandwidth=32)
    cut = np.asarray(truncate(expand(simulation.data[0], high), 8))
    made = np.asarray(expand(simulation.data[0], low))
    assert cut.shape == made.shape
    np.testing.assert_allclose(cut, made, atol=1e-4 * np.abs(made).max())

    with pytest.raises(ValueError, match='bandwidth'):
        truncate(expand(simulation.data[0], low), 16)


def test_a_displaced_pair_needs_its_shift_searched_too(
    rng: np.random.Generator,
) -> None:
    """Rotations alone cannot see a pair that is off centre.

    With phases the rotational score is the correlation at no shift, and
    two particles three voxels from their centres do not correlate
    there. Searched over a lattice of shifts at once they do. Measured on
    this simulation, with no noise: the ten pairs came out between 12
    and 146 degrees off with rotations alone, a median of 88, and
    between 2 and 13 with the shifts, every shift within a voxel and a
    half.
    """
    count, bandwidth = 5, 16
    simulation = simulate(rng, count, size=SIZE, tilt=60.0, shift=3.0)
    radii = band_radii(SIZE)
    fine = make_basis(radii, bandwidth, SIZE, sampling_bandwidth=2 * bandwidth)
    coarse = make_basis(
        radii, bandwidth // 2, SIZE, sampling_bandwidth=bandwidth
    )
    spacing = lattice_spacing(SIZE, float(radii[-1]))
    joint = Joint(
        coarse=coarse,
        fine=fine,
        lattice=np.asarray(shift_lattice(8.0, spacing), dtype=np.float32),
        spacing=spacing,
    )
    settings = Settings(bandwidth=bandwidth, max_shift=8.0, phases=True)
    offsets = np.arange(SIZE) - SIZE // 2
    distance = np.sqrt(
        offsets[:, None, None] ** 2
        + offsets[None, :, None] ** 2
        + offsets[None, None, :] ** 2
    )
    band = ((distance >= radii[0]) & (distance <= radii[-1] + 0.5)).astype(
        np.float32
    )
    expansions = [
        expand_particle(simulation.data[i], simulation.weight[i], fine, True)
        for i in range(count)
    ]

    errors: dict[bool, list[float]] = {False: [], True: []}
    misses = []
    for first in range(count):
        for second in range(first + 1, count):
            truth = simulation.rotations[first] @ simulation.rotations[second].T
            offset = (
                simulation.shifts[first] - truth @ simulation.shifts[second]
            )
            for together in (False, True):
                found = align_pair(
                    simulation.data[first],
                    simulation.weight[first],
                    simulation.data[second],
                    simulation.weight[second],
                    expansions[first],
                    expansions[second],
                    settings,
                    band,
                    joint if together else None,
                )
                errors[together].append(
                    float(
                        np.degrees(
                            relative_angle(truth, np.asarray(found.rotation))
                        )
                    )
                )
                if together:
                    misses.append(
                        float(np.linalg.norm(np.asarray(found.shift) - offset))
                    )

    assert np.median(errors[False]) > 45.0
    assert np.max(errors[True]) < 20.0
    assert np.max(misses) < 2.5


# ------------------------------------------------------------------ mirrors


def test_the_mirror_of_an_expansion_is_the_expansion_of_the_mirror(
    rng: np.random.Generator,
) -> None:
    """Inverting a real volume conjugates its spectrum. `mirrored` makes
    the expansions of that from the expansions of the particle, without
    expanding anything again. Measured: within 1.4e-7."""
    from frmony.matching.correlation import mirrored

    data = (
        rng.normal(size=(SIZE,) * 3) + 1j * rng.normal(size=(SIZE,) * 3)
    ).astype(np.complex64)
    weight = rng.uniform(size=(SIZE,) * 3).astype(np.float32)
    basis = _basis()
    made = mirrored(expand_particle(data, weight, basis, True))
    direct = expand_particle(np.conj(data), weight, basis, True)
    for found, wanted in zip(
        (made.amplitude, made.energy, made.weight),
        (direct.amplitude, direct.energy, direct.weight),
        strict=True,
    ):
        scale = float(np.abs(np.asarray(wanted)).max())
        np.testing.assert_allclose(
            np.asarray(found), np.asarray(wanted), atol=1e-5 * scale
        )


def test_a_mirrored_pair_is_measured_as_a_mirror(
    rng: np.random.Generator,
) -> None:
    """The second particle inverted through its centre. With mirrors the
    pair comes back as minus the rotation between them, a matrix of
    determinant minus one; without, it cannot."""
    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    mirror = np.conj(simulation.data[1])
    basis = _basis()
    truth = -(simulation.rotations[0] @ simulation.rotations[1].T)
    for mirrors in (True, False):
        settings = Settings(
            bandwidth=BANDWIDTH,
            max_shift=1.0,
            phases=True,
            mirrors=mirrors,
        )
        found = align_pair(
            simulation.data[0],
            simulation.weight[0],
            mirror,
            simulation.weight[1],
            expand_particle(
                simulation.data[0], simulation.weight[0], basis, True
            ),
            expand_particle(mirror, simulation.weight[1], basis, True),
            settings,
        )
        rotation = np.asarray(found.rotation)
        if mirrors:
            assert np.linalg.det(rotation) < 0.0
            assert np.degrees(relative_angle(-rotation, -truth)) < 10.0
        else:
            assert np.linalg.det(rotation) > 0.0
