"""Orientations and centres from pairwise measurements."""

from __future__ import annotations

import numpy as np
import pytest

from frmony.geometry import exponential_map, random_rotations, relative_angle
from frmony.graph import build_pairs
from frmony.symmetry import point_group
from frmony.synchronization.power_method import (
    collapse,
    polar_rows,
    synchronize,
)
from frmony.synchronization.rotations import (
    propagate_along_tree,
    synchronize_rotations,
)
from frmony.synchronization.translations import synchronize_shifts


def _symmetric_agreement(
    found: np.ndarray, truth: np.ndarray, group: np.ndarray, step: int = 1
) -> float:
    """How far the pairs agree [deg], allowing symmetry.

    With a point group the orientation of each particle is recovered only
    up to a member of it, independently of every other particle, so what
    has to be a group element is ``truth_i.T @ found_i @ found_j.T @
    truth_j``. Comparing the relative orientations directly, which is right
    without symmetry, is not right with it.
    """
    count = len(found)
    worst = 0.0
    for first in range(0, count, step):
        for second in range(first + 1, count, step):
            residual = (
                truth[first].T @ found[first] @ found[second].T @ truth[second]
            )
            closest = max(
                float(np.trace(residual @ element.T)) for element in group
            )
            worst = max(
                worst,
                float(
                    np.degrees(
                        np.arccos(np.clip(0.5 * (closest - 1.0), -1.0, 1.0))
                    )
                ),
            )
    return worst


def _centres_agree(
    found: np.ndarray, truth: np.ndarray, orientations: np.ndarray
) -> float:
    """How far the centres are from the truth, up to moving the structure.

    The freedom is ``t_i -> t_i - Q_i @ d``, so the best ``d`` is fitted
    out before comparing.
    """
    offset = np.linalg.lstsq(
        np.concatenate(orientations, axis=0),
        np.concatenate(truth - found, axis=0),
        rcond=None,
    )[0]
    return float(
        np.abs(
            (truth - found) - np.einsum('nij,j->ni', orientations, offset)
        ).max()
    )


def _agreement(found: np.ndarray, truth: np.ndarray) -> float:
    """How far the relative orientations are from the true ones [deg].

    Orientations are only ever known up to one global rotation, so what is
    compared is what each pair says, not the orientations themselves.
    """
    count = len(found)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    return float(
        np.max(
            [
                np.degrees(
                    relative_angle(found[i] @ found[j].T, truth[i] @ truth[j].T)
                )
                for i, j in pairs
            ]
        )
    )


def _measurements(
    rng: np.random.Generator,
    count: int,
    partners: int,
    noise: float = 0.0,
    symmetry: str = 'C1',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = random_rotations(rng, count)
    # The partners asked for throughout this file are two and a half
    # times what they once were. `build_pairs` used to hand out that
    # many more than it was asked for, and the tolerances below were
    # measured on those graphs, so asking for what they really had keeps
    # each test measuring what it measured.
    left, right = build_pairs(count, partners, rng)
    group = point_group(symmetry)
    chosen = group[rng.integers(0, len(group), len(left))]
    measured = truth[left] @ chosen @ np.swapaxes(truth[right], -2, -1)

    if noise > 0.0:
        angles = rng.normal(0.0, noise, size=(len(left), 3))
        from frmony.geometry import exponential_map

        measured = measured @ exponential_map(np.radians(angles))
    return truth, left, right, measured


# ------------------------------------------------------------ power method


def test_polar_rows_orthonormalizes(rng: np.random.Generator) -> None:
    blocks = rng.normal(size=(16, 3, 7))
    projected = polar_rows(blocks)
    gram = np.einsum('nir,njr->nij', projected, projected)
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(3), gram.shape), atol=1e-12
    )


def test_synchronize_recovers_the_orientations(
    rng: np.random.Generator,
) -> None:
    truth, left, right, measured = _measurements(rng, 60, 20)
    weights = rng.uniform(0.5, 2.0, len(left))

    found, iterations = synchronize(60, left, right, measured, weights)
    assert 1 < iterations < 512
    assert _agreement(found, truth) < 0.01


def test_synchronize_survives_noise(rng: np.random.Generator) -> None:
    truth, left, right, measured = _measurements(rng, 80, 24, noise=5.0)
    weights = np.ones(len(left))

    found, _ = synchronize(80, left, right, measured, weights)
    assert _agreement(found, truth) < 8.0


def test_the_result_is_a_rotation_not_a_reflection(
    rng: np.random.Generator,
) -> None:
    """The closest orthogonal matrix is not the closest rotation.

    Letting a reflection through would mirror the structure, and nothing
    downstream would notice.
    """
    _, left, right, measured = _measurements(rng, 40, 15, noise=3.0)
    found, _ = synchronize(40, left, right, measured, np.ones(len(left)))
    np.testing.assert_allclose(np.linalg.det(found), 1.0, atol=1e-9)


def test_collapse_is_not_fooled_by_degenerate_singular_values(
    rng: np.random.Generator,
) -> None:
    """The trap that cost an afternoon.

    At the solution every lifted block is the same rotation times one
    common matrix, and all three singular values of the stack are equal.
    Taking an arbitrary basis of that degenerate subspace leaves a nearly
    singular matrix in the way, and the orientations come out scrambled
    while the objective sits at its maximum.
    """
    count = 50
    truth = random_rotations(rng, count)
    for rank in (3, 5, 7):
        lifted = np.zeros((count, 3, rank))
        # Any common matrix with orthonormal rows is a valid solution.
        common = polar_rows(rng.normal(size=(1, 3, rank)))[0]
        lifted[:] = truth @ common
        assert _agreement(collapse(lifted), truth) < 1e-4


def test_a_weight_of_zero_ignores_a_pair(rng: np.random.Generator) -> None:
    truth, left, right, measured = _measurements(rng, 40, 15)
    weights = np.ones(len(left))

    spoiled = measured.copy()
    spoiled[0] = random_rotations(rng, 1)[0]
    weights[0] = 0.0

    found, _ = synchronize(40, left, right, spoiled, weights)
    assert _agreement(found, truth) < 0.01


def test_synchronize_refuses_nonsense(rng: np.random.Generator) -> None:
    _, left, right, measured = _measurements(rng, 10, 8)
    weights = np.ones(len(left))
    with pytest.raises(ValueError, match='each pair'):
        synchronize(10, left, right, measured, weights[:-1])
    with pytest.raises(ValueError, match='not negative'):
        synchronize(10, left, right, measured, -weights)
    with pytest.raises(ValueError, match='3x3'):
        synchronize(10, left, right, measured[:, :2], weights)
    with pytest.raises(ValueError, match='rank'):
        synchronize(10, left, right, measured, weights, rank=2)


def test_no_pairs_keeps_the_start(rng: np.random.Generator) -> None:
    start = random_rotations(rng, 5)
    found, iterations = synchronize(
        5,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 3, 3)),
        np.zeros(0),
        start=start,
    )
    assert iterations == 0
    assert _agreement(found, start) < 1e-4


# ------------------------------------------------------------------- tree


def test_the_tree_satisfies_its_own_edges(rng: np.random.Generator) -> None:
    """A tree has no cycles, so nothing can disagree along it."""
    truth, left, right, measured = _measurements(rng, 50, 12)
    weights = rng.uniform(0.1, 1.0, len(left))

    found = propagate_along_tree(50, left, right, measured, weights)
    assert _agreement(found, truth) < 0.01


def test_the_tree_is_a_good_start(rng: np.random.Generator) -> None:
    truth, left, right, measured = _measurements(rng, 60, 15, noise=6.0)
    weights = np.ones(len(left))

    start = propagate_along_tree(60, left, right, measured, weights)
    found, iterations = synchronize(
        60, left, right, measured, weights, start=start
    )
    assert _agreement(found, truth) < 15.0
    assert iterations < 512


# -------------------------------------------------------------- the driver


def test_the_driver_recovers_the_orientations(
    rng: np.random.Generator,
) -> None:
    truth, left, right, measured = _measurements(rng, 60, 20, noise=4.0)
    result = synchronize_rotations(
        60, left, right, measured, np.ones(len(left))
    )
    assert _agreement(result.orientations, truth) < 10.0
    assert result.kept.all()
    assert np.degrees(np.median(result.residuals)) < 8.0


@pytest.mark.parametrize('symmetry', ['C2', 'C4', 'D2'])
def test_a_symmetric_particle_still_synchronizes(
    rng: np.random.Generator, symmetry: str
) -> None:
    """Each measurement is on an arbitrary member of its coset.

    Handed to the synchronization unchanged, these average over the orbit
    instead of settling on it. Snapping them back first is what makes the
    problem solvable at all.
    """
    count = 60
    truth, left, right, measured = _measurements(
        rng, count, 24, noise=2.0, symmetry=symmetry
    )
    result = synchronize_rotations(
        count,
        left,
        right,
        measured,
        np.ones(len(left)),
        symmetry=symmetry,
        rounds=6,
    )
    worst = _symmetric_agreement(
        result.orientations, truth, point_group(symmetry)
    )
    assert worst < 15.0


def test_ignoring_symmetry_fails(rng: np.random.Generator) -> None:
    """The case that makes the symmetry handling worth having."""
    count = 60
    truth, left, right, measured = _measurements(rng, count, 24, symmetry='C4')
    group = point_group('C4')

    naive = synchronize_rotations(
        count,
        left,
        right,
        measured,
        np.ones(len(left)),
        symmetry='C1',
        reject_above=None,
    )
    aware = synchronize_rotations(
        count, left, right, measured, np.ones(len(left)), symmetry='C4'
    )

    assert _symmetric_agreement(aware.orientations, truth, group, 7) < 5.0
    assert _symmetric_agreement(naive.orientations, truth, group, 7) > 20.0


def test_outliers_are_dropped(rng: np.random.Generator) -> None:
    count = 60
    truth, left, right, measured = _measurements(rng, count, 24, noise=2.0)
    spoiled = measured.copy()
    broken = rng.permutation(len(left))[: len(left) // 8]
    spoiled[broken] = random_rotations(rng, len(broken))

    result = synchronize_rotations(
        count, left, right, spoiled, np.ones(len(left)), rounds=4
    )
    assert not result.kept[broken].all(), 'the bad pairs should be noticed'
    assert _agreement(result.orientations, truth) < 15.0


def test_the_driver_refuses_nonsense(rng: np.random.Generator) -> None:
    _, left, right, measured = _measurements(rng, 10, 8)
    with pytest.raises(ValueError, match='two particles'):
        synchronize_rotations(1, left, right, measured, np.ones(len(left)))
    with pytest.raises(ValueError, match='one pair'):
        synchronize_rotations(
            10,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 3, 3)),
            np.zeros(0),
        )


# ------------------------------------------------------------------ shifts


def test_shifts_are_recovered(rng: np.random.Generator) -> None:
    count = 50
    truth = random_rotations(rng, count)
    centres = rng.normal(0.0, 3.0, size=(count, 3))
    centres -= centres.mean(axis=0)
    left, right = build_pairs(count, 15, rng)

    relative = truth[left] @ np.swapaxes(truth[right], -2, -1)
    measured = centres[left] - np.einsum('pij,pj->pi', relative, centres[right])

    found, residuals = synchronize_shifts(count, left, right, measured, truth)
    assert residuals.max() < 1e-7
    # Up to a move of the structure, which no measurement can see.
    assert _centres_agree(found, centres, truth) < 1e-6


def test_shifts_fix_the_gauge(rng: np.random.Generator) -> None:
    """Moving the structure leaves every measurement alone, so the answer
    needs a convention to be unique.

    The freedom is not a constant shift of the centres. Taking the
    structure to ``S(y - d)`` sends ``t_i`` to ``t_i - Q_i @ d``, which is
    different for every particle, and that is what the gauge removes.
    """
    count = 30
    truth = random_rotations(rng, count)
    centres = rng.normal(0.0, 2.0, size=(count, 3))
    left, right = build_pairs(count, 12, rng)
    relative = truth[left] @ np.swapaxes(truth[right], -2, -1)
    measured = centres[left] - np.einsum('pij,pj->pi', relative, centres[right])

    found, residuals = synchronize_shifts(count, left, right, measured, truth)
    assert residuals.max() < 1e-6

    # The convention: the structure sits at the middle of its own box.
    np.testing.assert_allclose(
        np.einsum('nji,nj->i', truth, found), 0.0, atol=1e-6
    )

    # And what is left over is exactly a move of the structure.
    offset = np.linalg.lstsq(
        np.concatenate(truth, axis=0),
        np.concatenate(centres - found, axis=0),
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        centres - found, np.einsum('nij,j->ni', truth, offset), atol=1e-6
    )


def test_shifts_survive_noise(rng: np.random.Generator) -> None:
    count = 60
    truth = random_rotations(rng, count)
    centres = rng.normal(0.0, 3.0, size=(count, 3))
    centres -= centres.mean(axis=0)
    left, right = build_pairs(count, 20, rng)
    relative = truth[left] @ np.swapaxes(truth[right], -2, -1)
    measured = centres[left] - np.einsum('pij,pj->pi', relative, centres[right])
    measured = measured + rng.normal(0.0, 0.3, size=measured.shape)

    found, residuals = synchronize_shifts(count, left, right, measured, truth)
    assert np.abs(found - centres).max() < 1.0
    assert np.median(residuals) < 1.0


def test_weights_tilt_the_fit(rng: np.random.Generator) -> None:
    count = 40
    truth = random_rotations(rng, count)
    centres = rng.normal(0.0, 2.0, size=(count, 3))
    centres -= centres.mean(axis=0)
    left, right = build_pairs(count, 15, rng)
    relative = truth[left] @ np.swapaxes(truth[right], -2, -1)
    measured = centres[left] - np.einsum('pij,pj->pi', relative, centres[right])

    spoiled = measured.copy()
    spoiled[::7] += 20.0
    weights = np.ones(len(left))
    weights[::7] = 0.0

    found, _ = synchronize_shifts(count, left, right, spoiled, truth, weights)
    assert _centres_agree(found, centres, truth) < 1e-6


def test_shift_synchronization_refuses_nonsense(
    rng: np.random.Generator,
) -> None:
    count = 10
    truth = random_rotations(rng, count)
    left, right = build_pairs(count, 8, rng)
    shifts = np.zeros((len(left), 3))

    with pytest.raises(ValueError, match='each pair'):
        synchronize_shifts(count, left, right, shifts[:-1], truth)
    with pytest.raises(ValueError, match='three components'):
        synchronize_shifts(count, left, right, shifts[:, :2], truth)
    with pytest.raises(ValueError, match='one for each particle'):
        synchronize_shifts(count, left, right, shifts, truth[:-1])
    with pytest.raises(ValueError, match='not negative'):
        synchronize_shifts(
            count, left, right, shifts, truth, -np.ones(len(left))
        )


def test_everything_together(rng: np.random.Generator) -> None:
    """Rotations then shifts, as the pipeline runs them."""
    count = 50
    truth = random_rotations(rng, count)
    centres = rng.normal(0.0, 2.0, size=(count, 3))
    centres -= centres.mean(axis=0)
    left, right = build_pairs(count, 20, rng)

    relative = truth[left] @ np.swapaxes(truth[right], -2, -1)
    from frmony.geometry import exponential_map

    wobble = exponential_map(
        np.radians(rng.normal(0.0, 3.0, size=(len(left), 3)))
    )
    measured_rotation = relative @ wobble
    measured_shift = centres[left] - np.einsum(
        'pij,pj->pi', relative, centres[right]
    )

    result = synchronize_rotations(
        count, left, right, measured_rotation, np.ones(len(left))
    )
    found, residuals = synchronize_shifts(
        count,
        left[result.kept],
        right[result.kept],
        measured_shift[result.kept],
        result.orientations,
    )
    assert _agreement(result.orientations, truth) < 10.0
    assert np.median(residuals) < 1.0
    np.testing.assert_allclose(
        np.einsum('nji,nj->i', result.orientations, found), 0.0, atol=1e-6
    )


# ------------------------------------------------- weighing pairs by residual


def _particle_errors(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Each orientation's error [deg], once the one global rotation that
    orientations are known up to is taken out."""
    gram = np.einsum('nji,njk->ik', truth, found)
    u, _, vt = np.linalg.svd(gram)
    turn = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return np.degrees(relative_angle(found, truth @ turn))


def test_a_mixture_of_residuals_is_taken_apart(
    rng: np.random.Generator,
) -> None:
    """Right pairs off by a few degrees, wrong ones off at random: the fit
    finds how many of each and how far the right ones are off. Measured:
    a share of 0.31 for 0.30, and a width of 3.01 degrees for 3."""
    from frmony.synchronization.rotations import inlier_probabilities

    count = 4000
    right = rng.random(count) < 0.3
    width = np.radians(3.0)
    small = np.linalg.norm(rng.normal(size=(count, 3)) * width, axis=1)
    noise = random_rotations(rng, count)
    large = np.arccos(
        np.clip((np.trace(noise, axis1=1, axis2=2) - 1) / 2, -1, 1)
    )
    residuals = np.where(right, small, large)

    posterior, fitted, share = inlier_probabilities(residuals)
    assert share == pytest.approx(0.3, abs=0.03)
    assert np.degrees(fitted) == pytest.approx(3.0, abs=0.3)
    assert posterior[right].mean() > 0.95
    assert posterior[~right].mean() < 0.05


def test_a_perfect_measurement_is_right() -> None:
    # Both densities vanish as r**2 at nought. Compared as they are, a
    # residual of exactly nought is nought over nought.
    from frmony.synchronization.rotations import inlier_probabilities

    residuals = np.concatenate((np.zeros(50), np.linspace(1.0, 3.0, 50)))
    posterior, _, _ = inlier_probabilities(residuals)
    assert np.all(posterior[:50] > 0.99)
    assert np.all(np.isfinite(posterior))


def test_a_group_brings_wrong_pairs_closer() -> None:
    # Modulo C4 a random rotation is never more than its nearest element
    # away, so a wrong pair looks less wrong, and the density says so.
    # Measured: a mean of 2.21 radians with no group and 1.64 with C4.
    from frmony.synchronization.rotations import outlier_density

    angles = np.linspace(0.0, np.pi, 2001)
    plain = outlier_density(angles)
    four = outlier_density(angles, point_group('C4'))
    for density in (plain, four):
        assert np.trapezoid(density, angles) == pytest.approx(1.0, abs=0.01)
    mean = lambda d: np.trapezoid(angles * d, angles)  # noqa: E731
    assert mean(four) < mean(plain) - 0.3


def test_weighing_beats_a_threshold_when_most_pairs_are_wrong(
    rng: np.random.Generator,
) -> None:
    """Seven pairs in ten wrong, the rest off by five degrees.

    Measured: a median error of 6.1 degrees, and 11.3 at the ninetieth
    percentile, dropping what is off by more than ninety degrees; 0.85
    and 1.4 weighing every pair by its residual, which put the right
    pairs at 0.999 and the wrong ones at 0.001.
    """
    count = 300
    truth = random_rotations(rng, count)
    left, right = build_pairs(count, 120, rng)
    relative = truth[left] @ np.swapaxes(truth[right], -2, -1)
    good = rng.random(len(left)) < 0.3
    noise = exponential_map(
        rng.normal(size=(len(left), 3)) * np.radians(5.0) / np.sqrt(3.0)
    )
    measured = np.where(
        good[:, None, None], relative @ noise, random_rotations(rng, len(left))
    )

    threshold = synchronize_rotations(
        count,
        left,
        right,
        measured,
        np.ones(len(left)),
        robust=False,
        reject_above=np.radians(90.0),
        rounds=6,
    )
    weighed = synchronize_rotations(
        count, left, right, measured, np.ones(len(left))
    )
    errors = {
        name: _particle_errors(result.orientations, truth)
        for name, result in (('threshold', threshold), ('weighed', weighed))
    }
    assert np.median(errors['weighed']) < 2.0
    assert np.percentile(errors['weighed'], 90) < 3.0
    assert np.median(errors['threshold']) > 3.0 * np.median(errors['weighed'])
    assert weighed.inliers is not None
    assert weighed.inliers[good].mean() > 0.9
    assert weighed.inliers[~good].mean() < 0.1
