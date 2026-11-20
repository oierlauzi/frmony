"""The five-degree-of-freedom matching of Kovacs et al. (2003).

What is checked here is the part that carries the mathematics: the
two-centre integrals, which are the whole cost of the method, and the
five-dimensional series built on them. The integrals have an exact
identity to be held to -- at a distance of zero they must collapse onto the
ordinary radial overlap -- and that is what pins them down.
"""

from __future__ import annotations

import numpy as np
import pytest

from frmony.harmonics.legendre import orders
from frmony.harmonics.transform import expand, make_basis
from frmony.harmonics.wigner import wigner_d
from frmony.matching.frm5d import (
    _wigner_zero,
    correlation_5d,
    cylindrical_grid,
    memory_for,
    series_5d,
    two_centre_integrals,
)
from frmony.matching.rotation import radial_overlap

BANDWIDTH = 4
SIZE = 48


def _smooth(size: int, twist: float) -> np.ndarray:
    """A volume whose coefficients vary slowly with radius."""
    axis = np.arange(size) - size // 2
    squared = (
        axis[:, None, None] ** 2
        + axis[None, :, None] ** 2
        + axis[None, None, :] ** 2
    )
    return np.exp(-squared / 120.0) * (
        1.0 + 0.5 * np.sin(axis[:, None, None] / twist)
    )


def _expansions(step: float = 0.25):
    radii = np.arange(5.0, 18.0, step)
    basis = make_basis(radii, BANDWIDTH, SIZE, sampling_bandwidth=2 * BANDWIDTH)
    left = np.asarray(expand(_smooth(SIZE, 3.0), basis))
    right = np.asarray(expand(_smooth(SIZE, 4.0), basis))
    return radii, np.asarray(basis.radial), left, right


# ------------------------------------------------- the Wigner zero column


def test_a_wigner_function_with_a_zero_index_is_a_legendre_one() -> None:
    """Which is why the method needs nothing new for them.

    ``d^l_{n0}(b) = sqrt(4 pi / (2l + 1)) * Ybar_l^n(cos b)``, so the
    table built for the expansion serves here too.
    """
    colatitude = np.array([0.0, 0.3, 1.0, 2.2, 2.9, np.pi])
    table = _wigner_zero(BANDWIDTH, np.cos(colatitude))
    index = orders(BANDWIDTH)

    for point, angle in enumerate(colatitude):
        for degree in range(BANDWIDTH):
            exact = wigner_d(degree, float(angle))
            for column, order in enumerate(index):
                if abs(int(order)) > degree:
                    assert table[point, degree, column] == 0.0
                else:
                    assert table[point, degree, column] == pytest.approx(
                        exact[int(order) + degree, degree], abs=1e-12
                    )


def test_the_zero_column_is_orthogonal_over_the_colatitude() -> None:
    """The orthogonality the collapse at zero distance rests on."""
    angle = np.linspace(0.0, np.pi, 4000)
    table = _wigner_zero(BANDWIDTH, np.cos(angle))
    index = orders(BANDWIDTH)

    for column, order in enumerate(index):
        for degree in range(BANDWIDTH):
            if abs(int(order)) > degree:
                continue
            integral = np.trapezoid(
                table[:, degree, column] ** 2 * np.sin(angle), angle
            )
            assert integral * (2 * degree + 1) / 2 == pytest.approx(
                1.0, abs=1e-6
            )


# -------------------------------------------------- the two-centre integrals


def test_the_integrals_collapse_at_zero_distance() -> None:
    """The identity that pins the whole construction down.

    At a distance of zero the second object is not moved, the two radii
    and colatitudes coincide, and the Wigner functions integrate to
    ``2 / (2l + 1)`` times a Kronecker delta. What is left is the ordinary
    radial overlap, the same for every ``n``.

    The conjugate appears because the 2003 paper conjugates the first
    object while the 2002 one conjugates the second.
    """
    radii, weight, left, right = _expansions()
    overlap = np.conjugate(np.asarray(radial_overlap(left, right)))
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 0.0, samples=128)
    integrals = np.asarray(
        two_centre_integrals(left, right, radii, weight, 0.0, grid, BANDWIDTH)
    )

    index = orders(BANDWIDTH)
    ratios = []
    for column, order in enumerate(index):
        for degree in range(BANDWIDTH):
            if abs(int(order)) > degree:
                continue
            for first in range(2 * BANDWIDTH):
                for second in range(2 * BANDWIDTH):
                    if (
                        abs(int(index[first])) > degree
                        or abs(int(index[second])) > degree
                    ):
                        continue
                    reference = overlap[degree, first, second]
                    if abs(reference) < 1e-9:
                        continue
                    ratios.append(
                        integrals[column, degree, first, degree, second]
                        / reference
                    )

    ratios = np.array(ratios)
    assert len(ratios) > 50
    # The quadrature is a finite grid over a half plane and the radial
    # direction is interpolated between shells, so this is the accuracy of
    # the integration rather than of the identity.
    assert np.mean(ratios.real) == pytest.approx(1.0, abs=0.1)
    assert np.std(ratios.real) < 0.2


def test_the_integrals_vanish_off_the_diagonal_at_zero_distance() -> None:
    """Different degrees do not mix when nothing has been moved."""
    radii, weight, left, right = _expansions()
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 0.0, samples=128)
    integrals = np.asarray(
        two_centre_integrals(left, right, radii, weight, 0.0, grid, BANDWIDTH)
    )

    off = integrals.copy()
    degrees = np.arange(BANDWIDTH)
    off[:, degrees, :, degrees, :] = 0.0
    assert np.abs(off).max() < 0.05 * np.abs(integrals).max()


def test_the_integrals_respect_the_order_of_a_degree() -> None:
    """Nothing with an order a degree cannot carry may be non-zero."""
    radii, weight, left, right = _expansions(step=1.0)
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 2.0, samples=48)
    integrals = np.asarray(
        two_centre_integrals(left, right, radii, weight, 1.5, grid, BANDWIDTH)
    )
    index = orders(BANDWIDTH)

    for column, order in enumerate(index):
        for degree in range(BANDWIDTH):
            if abs(int(order)) > degree:
                assert np.abs(integrals[column, degree]).max() < 1e-6, (
                    'a degree carried an order it cannot have'
                )


def test_moving_the_second_object_changes_the_integrals() -> None:
    radii, weight, left, right = _expansions(step=1.0)
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 4.0, samples=48)
    still = np.asarray(
        two_centre_integrals(left, right, radii, weight, 0.0, grid, BANDWIDTH)
    )
    moved = np.asarray(
        two_centre_integrals(left, right, radii, weight, 3.0, grid, BANDWIDTH)
    )
    assert np.abs(moved - still).max() > 0.05 * np.abs(still).max()


def test_a_degenerate_band_is_refused() -> None:
    with pytest.raises(ValueError, match='not a band'):
        cylindrical_grid(5.0, 5.0, 0.0)
    with pytest.raises(ValueError, match='not a band'):
        cylindrical_grid(9.0, 3.0, 0.0)


# ------------------------------------------------------------- the series


def test_the_series_has_five_angles() -> None:
    radii, weight, left, right = _expansions(step=1.0)
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 0.0, samples=32)
    integrals = two_centre_integrals(
        left, right, radii, weight, 0.0, grid, BANDWIDTH
    )
    series = np.asarray(series_5d(integrals, BANDWIDTH))
    assert series.shape == (2 * BANDWIDTH,) * 5
    assert np.isfinite(series).all()


def test_the_correlation_is_real_and_finite() -> None:
    radii, weight, left, right = _expansions(step=1.0)
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 0.0, samples=32)
    series = series_5d(
        two_centre_integrals(left, right, radii, weight, 0.0, grid, BANDWIDTH),
        BANDWIDTH,
    )
    values = np.asarray(correlation_5d(series))

    assert values.shape == (2 * BANDWIDTH,) * 5
    assert values.dtype.kind == 'f'
    assert np.isfinite(values).all()
    assert np.abs(values).max() > 0.0


def test_the_series_is_linear() -> None:
    radii, weight, left, right = _expansions(step=1.0)
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), 0.0, samples=32)
    first = two_centre_integrals(
        left, right, radii, weight, 0.0, grid, BANDWIDTH
    )
    doubled = np.asarray(series_5d(2.0 * np.asarray(first), BANDWIDTH))
    single = np.asarray(series_5d(first, BANDWIDTH))
    np.testing.assert_allclose(doubled, 2.0 * single, rtol=1e-4, atol=1e-6)


# -------------------------------------------------------------- the cost


@pytest.mark.parametrize(
    ('bandwidth', 'expected'),
    [(4, 32768 * 8), (8, 1048576 * 8), (16, 33554432 * 8)],
)
def test_the_cost_is_what_it_says(bandwidth: int, expected: int) -> None:
    """A quarter of a gigabyte at bandwidth 16, per value of the distance.

    This is the number that decides the method is not the default, so it
    is worth stating rather than discovering.
    """
    assert memory_for(bandwidth) == expected


def test_the_cost_grows_as_the_fifth_power() -> None:
    assert memory_for(8) == 32 * memory_for(4)
    assert memory_for(16) == 32 * memory_for(8)


def _blobs(rng: np.random.Generator, size: int) -> np.ndarray:
    axis = np.arange(size) - size // 2
    z, y, x = np.meshgrid(axis, axis, axis, indexing='ij')
    out = np.zeros((size,) * 3)
    for _ in range(5):
        centre = rng.normal(size=3)
        centre *= 0.22 * size * rng.uniform(0.4, 1.0) / np.linalg.norm(centre)
        out += rng.uniform(0.5, 1.5) * np.exp(
            -(
                (x - centre[0]) ** 2
                + (y - centre[1]) ** 2
                + (z - centre[2]) ** 2
            )
            / (2 * 1.6**2)
        )
    return out


def _five_dimensional(
    first: np.ndarray, second: np.ndarray, bandwidth: int, distance: float
) -> np.ndarray:
    from frmony.harmonics.transform import expand, make_basis
    from frmony.matching.frm5d import (
        correlation_5d,
        cylindrical_grid,
        series_5d,
        two_centre_integrals,
    )

    size = first.shape[-1]
    radii = np.arange(0.5, 0.5 * size - 1.0, 1.0)
    basis = make_basis(radii, bandwidth, size, sampling_bandwidth=2 * bandwidth)
    grid = cylindrical_grid(float(radii[0]), float(radii[-1]), distance)
    integrals = two_centre_integrals(
        np.asarray(expand(first.astype(np.float32), basis)),
        np.asarray(expand(second.astype(np.float32), basis)),
        radii,
        np.asarray(basis.radial),
        distance,
        grid,
        bandwidth,
    )
    return np.asarray(correlation_5d(series_5d(integrals, bandwidth)))


def _direct(
    first: np.ndarray,
    second: np.ndarray,
    index: np.ndarray,
    bandwidth: int,
    distance: float,
) -> float:
    """The correlation an index stands for, by turning and moving."""
    from scipy.ndimage import shift

    from conftest import rotate_volume
    from frmony.matching.frm5d import grid_to_rotations

    turn_first, turn_second = grid_to_rotations(index, bandwidth)
    moved = shift(
        rotate_volume(second, turn_second),
        (distance, 0.0, 0.0),
        order=3,
        mode='constant',
    )
    return float(np.sum(rotate_volume(first, turn_first) * moved))


def test_the_grid_is_the_correlation_it_says_it_is() -> None:
    """Held against the sum itself, at indices drawn at random.

    This is the test the series went without. Its integrals were checked,
    at a distance of zero, and the series built on them was not, and was
    wrong. Measured with the series as it stands: a correlation of
    0.9999 with the direct sum and a ratio of 0.97, the three in a
    hundred being the spheres a volume is read off.
    """
    rng = np.random.default_rng(0)
    size, bandwidth, distance = 24, 8, 3.0
    first, second = _blobs(rng, size), _blobs(rng, size)
    values = _five_dimensional(first, second, bandwidth, distance)

    indices = rng.integers(0, 2 * bandwidth, size=(24, 5))
    found = values[tuple(indices.T)]
    wanted = np.array(
        [
            _direct(first, second, index, bandwidth, distance)
            for index in indices
        ]
    )
    assert np.corrcoef(found, wanted)[0, 1] > 0.995
    assert np.dot(found, wanted) / np.dot(wanted, wanted) == pytest.approx(
        1.0, abs=0.08
    )


def test_the_peak_puts_a_copy_back_on_its_original() -> None:
    """A wrong reading of the index cannot pass this by accident.

    The second volume is the first, turned and moved by a known amount,
    so the grid has a sharp peak, and the rotations its index stands for
    have to superimpose the two. Under the series that stood here before,
    no reading of the index got past an overlap of 0.91.
    """
    from scipy.ndimage import shift

    from conftest import rotate_volume
    from frmony.matching.frm5d import grid_to_rotations

    rng = np.random.default_rng(1)
    size, bandwidth, distance = 24, 8, 3.0
    first = _blobs(rng, size)
    planted = rng.integers(1, 2 * bandwidth, size=5)
    turn_first, turn_second = grid_to_rotations(planted, bandwidth)
    # f(Q1^-1 x) = g(Q2^-1 (x - rho z)): turn f by Q1, move it back by
    # rho, and turn what comes out by the inverse of Q2.
    moved = shift(
        rotate_volume(first, turn_first),
        (-distance, 0.0, 0.0),
        order=3,
        mode='constant',
    )
    second = rotate_volume(moved, turn_second.T)

    values = _five_dimensional(first, second, bandwidth, distance)
    peak = np.array(np.unravel_index(int(values.argmax()), values.shape))
    overlap = _direct(first, second, peak, bandwidth, distance) / np.sqrt(
        np.sum(first**2) * np.sum(second**2)
    )
    assert overlap > 0.97
