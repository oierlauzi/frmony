"""Polishing an alignment against the score itself."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import phantom, rotate_volume, simulate, spectrum
from syncsta.geometry import random_rotations, relative_angle
from syncsta.harmonics.shells import band_radii
from syncsta.harmonics.transform import make_basis
from syncsta.matching.correlation import expand_particle
from syncsta.matching.pairwise import Settings, align_pair
from syncsta.matching.refine import (
    exponential_map,
    refine_alignment,
    score_at,
)
from syncsta.matching.translation import frequency_grid

SIZE = 32
BANDWIDTH = 16


def _aligned(simulation: object, first: int, second: int):
    basis = make_basis(
        band_radii(SIZE), BANDWIDTH, SIZE, sampling_bandwidth=2 * BANDWIDTH
    )
    data, weight = simulation.data, simulation.weight  # type: ignore[attr-defined]
    left = expand_particle(data[first], weight[first], basis)
    right = expand_particle(data[second], weight[second], basis)
    return align_pair(
        data[first],
        weight[first],
        data[second],
        weight[second],
        left,
        right,
        Settings(bandwidth=BANDWIDTH, candidates=8, max_shift=3.0),
    )


def test_the_exponential_map_is_a_rotation() -> None:
    rng = np.random.default_rng(0)
    for vector in [np.zeros(3), *rng.normal(size=(8, 3))]:
        matrix = np.asarray(exponential_map(np.asarray(vector)))
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-5)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-5)


def test_the_gradient_is_finite_at_the_origin() -> None:
    """Every refinement starts here, so this is not an edge case.

    The derivative of a norm is zero over zero at the origin, and a NaN
    there spreads through a `where` from the branch that was not taken.
    Working from the squared norm instead is what keeps it finite.
    """
    import jax

    gradient = jax.jacobian(exponential_map)(np.zeros(3))
    assert np.isfinite(np.asarray(gradient)).all()


def test_the_gradient_matches_a_finite_difference(
    rng: np.random.Generator,
) -> None:
    import jax
    import jax.numpy as jnp

    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    found = _aligned(simulation, 0, 1)
    grid = jnp.asarray(frequency_grid(SIZE))
    start = jnp.asarray(found.rotation, jnp.float32)
    offset = jnp.asarray(found.shift, jnp.float32)

    def score(parameters):
        return score_at(
            simulation.data[0],
            simulation.weight[0],
            simulation.data[1],
            simulation.weight[1],
            start @ exponential_map(parameters[:3]),
            offset + parameters[3:],
            grid,
        )

    begin = jnp.zeros(6, jnp.float32)
    analytic = np.asarray(jax.grad(score)(begin))
    assert np.isfinite(analytic).all()

    step = 2e-3
    numeric = np.array(
        [
            float(
                score(begin.at[axis].add(step))
                - score(begin.at[axis].add(-step))
            )
            / (2 * step)
            for axis in range(3)
        ]
    )
    np.testing.assert_allclose(analytic[:3], numeric, atol=5e-3)


def test_refining_never_lowers_the_score(rng: np.random.Generator) -> None:
    """The guarantee the caller gets, whatever the optimizer does."""
    count = 4
    simulation = simulate(rng, count, size=SIZE, tilt=60.0)
    for first in range(count):
        for second in range(first + 1, count):
            found = _aligned(simulation, first, second)
            _, _, score = refine_alignment(
                simulation.data[first],
                simulation.weight[first],
                simulation.data[second],
                simulation.weight[second],
                found.rotation,
                found.shift,
            )
            assert float(score) >= float(found.score) - 1e-6


def test_refining_sharpens_a_clean_alignment(
    rng: np.random.Generator,
) -> None:
    """Without a missing wedge it does what a refinement should."""
    count = 5
    structure = phantom(SIZE, 7, rng)
    rotations = random_rotations(rng, count)
    weight = np.ones((SIZE,) * 3, dtype=np.float32)

    data = []
    for rotation in rotations:
        values = spectrum(rotate_volume(structure, rotation))
        values[SIZE // 2, SIZE // 2, SIZE // 2] = 0.0
        data.append(values.astype(np.complex64))
    data = np.array(data)
    weights = np.broadcast_to(weight, data.shape).astype(np.float32)

    basis = make_basis(
        band_radii(SIZE), BANDWIDTH, SIZE, sampling_bandwidth=2 * BANDWIDTH
    )
    expansions = [
        expand_particle(data[i], weights[i], basis) for i in range(count)
    ]
    settings = Settings(bandwidth=BANDWIDTH, candidates=8, max_shift=3.0)

    before, after = [], []
    for first in range(count):
        for second in range(first + 1, count):
            found = align_pair(
                data[first],
                weights[first],
                data[second],
                weights[second],
                expansions[first],
                expansions[second],
                settings,
            )
            truth = rotations[first] @ rotations[second].T
            before.append(
                np.degrees(relative_angle(truth, np.asarray(found.rotation)))
            )
            turn, _, _ = refine_alignment(
                data[first],
                weights[first],
                data[second],
                weights[second],
                found.rotation,
                found.shift,
            )
            after.append(np.degrees(relative_angle(truth, np.asarray(turn))))

    assert np.median(after) < np.median(before)
    assert max(after) < max(before)


def test_refining_a_perfect_alignment_leaves_it(
    rng: np.random.Generator,
) -> None:
    """A particle against itself is already at the top of the score."""
    simulation = simulate(rng, 1, size=SIZE, tilt=60.0)
    turn, shift, score = refine_alignment(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[0],
        simulation.weight[0],
        np.eye(3),
        np.zeros(3),
    )
    assert np.degrees(relative_angle(np.eye(3), np.asarray(turn))) < 1.0
    assert np.abs(np.asarray(shift)).max() < 0.5
    assert float(score) > 0.9


def test_the_score_agrees_with_the_shift_search(
    rng: np.random.Generator,
) -> None:
    """`score_at` is the same quantity the grid search maximizes."""
    import jax.numpy as jnp

    from syncsta.matching.translation import shift_correlation

    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    grid = jnp.asarray(frequency_grid(SIZE))
    rotation = random_rotations(rng, 1)[0]

    values, _ = shift_correlation(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[1],
        simulation.weight[1],
        rotation,
        grid,
    )
    # Index [z, y, x] of no shift at all.
    middle = SIZE // 2
    direct = score_at(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[1],
        simulation.weight[1],
        jnp.asarray(rotation),
        jnp.zeros(3),
        grid,
    )
    assert float(direct) == pytest.approx(
        float(np.asarray(values)[middle, middle, middle]), abs=1e-4
    )


def test_a_wedge_moves_the_peak_away_from_the_truth(
    rng: np.random.Generator,
) -> None:
    """Why this is off by default.

    The optimizer works: the score rises. But with a wide wedge missing,
    the top of the score is not quite at the true orientation, so there is
    somewhere better to go than the truth and the refinement goes there.
    """
    simulation = simulate(rng, 2, size=SIZE, tilt=60.0)
    import jax.numpy as jnp

    grid = jnp.asarray(frequency_grid(SIZE))
    truth = simulation.rotations[0] @ simulation.rotations[1].T

    at_truth = float(
        score_at(
            simulation.data[0],
            simulation.weight[0],
            simulation.data[1],
            simulation.weight[1],
            jnp.asarray(truth, jnp.float32),
            jnp.zeros(3),
            grid,
        )
    )
    found = _aligned(simulation, 0, 1)
    _, _, refined = refine_alignment(
        simulation.data[0],
        simulation.weight[0],
        simulation.data[1],
        simulation.weight[1],
        found.rotation,
        found.shift,
    )
    assert float(refined) >= at_truth - 1e-3


def test_the_pipeline_can_turn_it_on(rng: np.random.Generator) -> None:
    from syncsta.particles import Subtomograms
    from syncsta.pipeline import Design, run

    simulation = simulate(rng, 8, size=SIZE, tilt=60.0)
    subtomograms = Subtomograms(
        data=simulation.data,
        weight=simulation.weight,
        pixel_size=4.0,
        box=SIZE,
    )
    design = Design(
        partners=4, bandwidth=12, candidates=6, max_shift=2.0, refine=True
    )
    result = run(subtomograms, design)
    assert result.orientations.shape == (8, 3, 3)
    assert np.isfinite(result.scores).all()
