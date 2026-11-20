"""Fast rotational matching, against a direct quadrature of the integral.

The precedent this package grew out of had no such test, and left four
unresolved questions about which Euler angle a grid index stood for. Every
claim the module makes is checked here against a correlation integrated
directly over the sphere, which shares no code with the transform.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special as sp

from syncsta.geometry import euler_to_matrix, random_rotations, relative_angle
from syncsta.harmonics.legendre import gauss_legendre, orders
from syncsta.harmonics.transform import expand_shells, make_basis
from syncsta.matching.rotation import (
    correlation_grid,
    grid_angles,
    grid_to_euler,
    radial_overlap,
    rotational_correlation,
)

BANDWIDTH = 6


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def _real_coefficients(rng: np.random.Generator) -> np.ndarray:
    """Coefficients of a real, band-limited function on the sphere."""
    index = orders(BANDWIDTH)
    coefficients = np.zeros((BANDWIDTH, 2 * BANDWIDTH), dtype=np.complex128)
    for degree in range(BANDWIDTH):
        for column, order in enumerate(index):
            if 0 <= int(order) <= degree:
                coefficients[degree, column] = rng.normal() + 1j * rng.normal()
    # A real function has c[l, -m] = (-1)**m conj(c[l, m]).
    for degree in range(BANDWIDTH):
        for column, order in enumerate(index):
            order = int(order)
            if order < 0 and abs(order) <= degree:
                mirror = int(np.flatnonzero(index == -order)[0])
                coefficients[degree, column] = ((-1) ** abs(order)) * np.conj(
                    coefficients[degree, mirror]
                )
    return coefficients


def _evaluate(
    coefficients: np.ndarray, colatitude: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    """The function those coefficients stand for, built with scipy."""
    shape = np.broadcast(colatitude, azimuth).shape
    values = np.zeros(shape, dtype=np.complex128)
    for degree in range(BANDWIDTH):
        for column, order in enumerate(orders(BANDWIDTH)):
            if abs(int(order)) > degree:
                continue
            values = values + coefficients[degree, column] * sp.sph_harm_y(
                degree, int(order), colatitude, azimuth
            )
    return values


class _Pair:
    """Two band-limited spheres, their expansions, and the exact answer."""

    def __init__(self, rng: np.random.Generator) -> None:
        nodes, azimuths = BANDWIDTH, 2 * BANDWIDTH
        cosine, self.weights = gauss_legendre(nodes)
        self.colatitude = np.arccos(cosine)
        self.azimuth = 2.0 * np.pi * np.arange(azimuths) / azimuths
        sine = np.sin(self.colatitude)
        self.directions = np.stack(
            (
                sine[:, None] * np.cos(self.azimuth)[None, :],
                sine[:, None] * np.sin(self.azimuth)[None, :],
                np.broadcast_to(cosine[:, None], (nodes, azimuths)),
            ),
            axis=-1,
        )
        self.azimuths = azimuths

        self.left_c = _real_coefficients(rng)
        self.right_c = _real_coefficients(rng)
        grid = (self.colatitude[:, None], self.azimuth[None, :])
        self.left = _evaluate(self.left_c, *grid)
        self.right = _evaluate(self.right_c, *grid)

        self.basis = make_basis([1.0], BANDWIDTH, size=16)
        self.scale = float(self.basis.radial[0]) ** 2
        self.left_h = np.asarray(expand_shells(self.left[None], self.basis))
        self.right_h = np.asarray(expand_shells(self.right[None], self.basis))

    def brute(self, rotation: np.ndarray) -> float:
        """The correlation at one rotation, integrated over the sphere.

        Shares nothing with the transform but the quadrature nodes.
        """
        turned = np.einsum('ji,abj->abi', rotation, self.directions)
        colatitude = np.arccos(np.clip(turned[..., 2], -1.0, 1.0))
        azimuth = np.arctan2(turned[..., 1], turned[..., 0])
        other = _evaluate(self.right_c, colatitude, azimuth)
        return float(
            np.real(
                np.sum(
                    self.weights[:, None]
                    * (2.0 * np.pi / self.azimuths)
                    * self.left
                    * np.conjugate(other)
                )
            )
        )

    def fast(self) -> np.ndarray:
        found = np.asarray(
            rotational_correlation(self.left_h, self.right_h, BANDWIDTH)
        )
        return found / self.scale


def test_the_grid_index_means_what_it_says(rng: np.random.Generator) -> None:
    """Every entry of the grid is the correlation at the rotation that
    `grid_to_euler` names, to the precision of the dtype.

    This is the test the precedent never had, and it is what pins the chain
    of conventions from the Fourier variables to RELION's Euler angles.
    """
    pair = _Pair(rng)
    grid = pair.fast()
    size = 2 * BANDWIDTH
    assert grid.shape == (size, size, size)

    indices = rng.integers(0, size, size=(40, 3))
    for index in indices:
        rot, tilt, psi = grid_to_euler(index, BANDWIDTH)
        expected = pair.brute(euler_to_matrix(rot, tilt, psi))
        assert grid[tuple(index)] == pytest.approx(expected, abs=2e-5)


def test_the_whole_grid_is_the_correlation(rng: np.random.Generator) -> None:
    """Not just sampled entries: every one of them at once."""
    pair = _Pair(rng)
    grid = pair.fast()
    size = 2 * BANDWIDTH

    index = np.stack(
        np.meshgrid(*(np.arange(size),) * 3, indexing='ij'), axis=-1
    ).reshape(-1, 3)
    # A tenth of the grid, which is still a few hundred rotations.
    index = index[rng.permutation(len(index))[:200]]
    expected = np.array(
        [
            pair.brute(euler_to_matrix(*grid_to_euler(entry, BANDWIDTH)))
            for entry in index
        ]
    )
    found = grid[tuple(index.T)]
    np.testing.assert_allclose(found, expected, atol=2e-5)


def test_the_correlation_peaks_at_the_true_rotation(
    rng: np.random.Generator,
) -> None:
    """Correlating a sphere with a rotated copy of itself must find the
    rotation, to within the step of the grid."""
    pair = _Pair(rng)
    truth = random_rotations(rng, 1)[0]

    turned = np.einsum('ji,abj->abi', truth, pair.directions)
    colatitude = np.arccos(np.clip(turned[..., 2], -1.0, 1.0))
    azimuth = np.arctan2(turned[..., 1], turned[..., 0])
    rotated = _evaluate(pair.left_c, colatitude, azimuth)
    rotated_h = np.asarray(expand_shells(rotated[None], pair.basis))

    grid = np.asarray(rotational_correlation(pair.left_h, rotated_h, BANDWIDTH))
    peak = np.unravel_index(int(np.argmax(grid)), grid.shape)
    found = euler_to_matrix(*grid_to_euler(np.array(peak), BANDWIDTH))

    # The peak is the turn that brings the second sphere onto the first, so
    # for a sphere turned by `truth` it is the inverse of `truth`.
    step = np.degrees(np.pi / BANDWIDTH)
    assert np.degrees(relative_angle(truth.T, found)) < step


def test_the_peak_undoes_the_turn(rng: np.random.Generator) -> None:
    """The peak, applied to the second sphere, reproduces the first.

    Stated as a property of the functions rather than of the matrices, so
    it holds whatever convention the matrices are written in.
    """
    pair = _Pair(rng)
    truth = random_rotations(rng, 1)[0]

    def turn(rotation: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        moved = np.einsum('ji,abj->abi', rotation, pair.directions)
        return _evaluate(
            coefficients,
            np.arccos(np.clip(moved[..., 2], -1.0, 1.0)),
            np.arctan2(moved[..., 1], moved[..., 0]),
        )

    rotated = turn(truth, pair.left_c)
    rotated_h = np.asarray(expand_shells(rotated[None], pair.basis))
    grid = np.asarray(rotational_correlation(pair.left_h, rotated_h, BANDWIDTH))
    peak = np.unravel_index(int(np.argmax(grid)), grid.shape)
    found = euler_to_matrix(*grid_to_euler(np.array(peak), BANDWIDTH))

    # Undoing the peak on the turned sphere should give back the original.
    undone = turn(truth @ found, pair.left_c)
    error = np.abs(undone - pair.left).max() / np.abs(pair.left).max()
    assert error < 0.5, 'the peak should roughly undo the turn'


def test_the_correlation_is_real(rng: np.random.Generator) -> None:
    """Two real spheres correlate to a real number at every rotation."""
    pair = _Pair(rng)
    overlap = radial_overlap(pair.left_h, pair.right_h)
    values = np.asarray(correlation_grid(overlap, BANDWIDTH))
    assert values.dtype.kind == 'f'
    assert np.isfinite(values).all()


def test_self_correlation_peaks_at_the_identity(
    rng: np.random.Generator,
) -> None:
    pair = _Pair(rng)
    grid = np.asarray(
        rotational_correlation(pair.left_h, pair.left_h, BANDWIDTH)
    )
    peak = np.unravel_index(int(np.argmax(grid)), grid.shape)
    found = euler_to_matrix(*grid_to_euler(np.array(peak), BANDWIDTH))
    assert np.degrees(relative_angle(np.eye(3), found)) < 1e-6


def test_the_grid_covers_every_rotation_twice(
    rng: np.random.Generator,
) -> None:
    """(rot, tilt, psi) and (rot + pi, -tilt, psi + pi) are one rotation,
    so an index and its reflection in the second axis agree.

    Peak finding leans on this to search half the axis.
    """
    pair = _Pair(rng)
    grid = pair.fast()
    size = 2 * BANDWIDTH

    for _ in range(20):
        n, h, m = (int(v) for v in rng.integers(0, size, 3))
        mirror = ((n + BANDWIDTH) % size, (-h) % size, (m + BANDWIDTH) % size)
        assert grid[n, h, m] == pytest.approx(grid[mirror], abs=2e-5)


def test_radial_overlap_refuses_mismatched_expansions() -> None:
    left = np.zeros((2, 4, 8), dtype=np.complex64)
    right = np.zeros((3, 4, 8), dtype=np.complex64)
    with pytest.raises(ValueError, match='same expansion'):
        radial_overlap(left, right)


def test_grid_to_euler_refuses_a_bad_index() -> None:
    with pytest.raises(ValueError, match='three components'):
        grid_to_euler(np.zeros((4, 2)), BANDWIDTH)


def test_grid_angles_step_evenly() -> None:
    angles = grid_angles(BANDWIDTH)
    assert len(angles) == 2 * BANDWIDTH
    np.testing.assert_allclose(np.diff(angles), np.pi / BANDWIDTH)
