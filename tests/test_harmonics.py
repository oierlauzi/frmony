"""Spherical harmonics, the quadrature, and the quarter-turn identity.

Nothing downstream can be trusted before these pass: the rotational
correlation is a contraction of these tables, so an error here would show up
only as orientations that are quietly a little wrong.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest
import scipy.special as sp

from syncsta.harmonics.legendre import (
    gauss_legendre,
    legendre_table,
    orders,
)
from syncsta.harmonics.shells import shell_directions, shell_grid
from syncsta.harmonics.transform import (
    Basis,
    expand,
    expand_shells,
    make_basis,
)
from syncsta.harmonics.wigner import (
    half_turn_table,
    quarter_turn_identity,
    wigner_d,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(31415)


# --------------------------------------------------------------- quadrature


@pytest.mark.parametrize('nodes', [1, 2, 5, 16, 33])
def test_gauss_legendre_is_exact_to_its_degree(nodes: int) -> None:
    """A quadrature of n nodes integrates degree 2n - 1 exactly, and that
    is why the colatitude needs no more nodes than the bandwidth."""
    points, weights = gauss_legendre(nodes)
    assert points.shape == weights.shape == (nodes,)

    for degree in range(2 * nodes):
        # The integral of x**d over [-1, 1].
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1.0)
        assert np.sum(weights * points**degree) == pytest.approx(
            exact, abs=1e-12
        )


def test_a_uniform_grid_would_not_be_exact() -> None:
    """The grid this package does not use, for the record.

    A uniform grid in theta with a sin(theta) weight, which is what the
    precedent implementation used, is not exact at any number of nodes.
    """
    for nodes in (8, 16, 32):
        theta = np.linspace(0.0, math.pi, nodes)
        weights = np.sin(theta) * (math.pi / (nodes - 1))
        # The integral of cos(theta)**2 over the sphere's colatitude.
        naive = np.sum(weights * np.cos(theta) ** 2)
        assert abs(naive - 2.0 / 3.0) > 1e-6


@pytest.mark.parametrize('nodes', [0, -2])
def test_an_empty_quadrature_is_refused(nodes: int) -> None:
    with pytest.raises(ValueError, match='node'):
        gauss_legendre(nodes)


# ----------------------------------------------------------------- Legendre


def test_legendre_matches_scipy() -> None:
    bandwidth, nodes = 10, 14
    table = legendre_table(bandwidth, nodes)
    cosine, _ = gauss_legendre(nodes)
    colatitude = np.arccos(cosine)
    azimuth = np.zeros_like(colatitude)

    for degree in range(bandwidth):
        for column, order in enumerate(orders(bandwidth)):
            if abs(int(order)) > degree:
                assert np.all(table[:, degree, column] == 0.0)
                continue
            # At phi = 0 a spherical harmonic is its Legendre part.
            expected = sp.sph_harm_y(
                degree, int(order), colatitude, azimuth
            ).real
            np.testing.assert_allclose(
                table[:, degree, column], expected, atol=1e-13
            )


def test_legendre_is_orthonormal_under_the_quadrature() -> None:
    """The property the whole transform depends on."""
    bandwidth, nodes = 12, 12
    table = legendre_table(bandwidth, nodes)
    _, weights = gauss_legendre(nodes)

    for column, order in enumerate(orders(bandwidth)):
        magnitude = abs(int(order))
        if magnitude >= bandwidth:
            continue
        block = table[:, magnitude:, column]
        gram = np.einsum('ta,t,tb->ab', block, 2.0 * np.pi * weights, block)
        np.testing.assert_allclose(gram, np.eye(len(gram)), atol=1e-11)


def test_legendre_stays_bounded_at_a_high_bandwidth() -> None:
    """Normalizing inside the recurrence is what stops it overflowing."""
    bandwidth = 64
    table = legendre_table(bandwidth, bandwidth)
    assert np.all(np.isfinite(table))

    # The bound a normalised harmonic obeys. The unnormalised recurrence
    # would have overflowed long before this degree.
    degree = np.arange(bandwidth)
    bound = np.sqrt((2.0 * degree + 1.0) / (4.0 * np.pi))
    assert np.all(np.abs(table).max(axis=(0, 2)) <= bound + 1e-12)


@pytest.mark.parametrize('bandwidth', [0, -1])
def test_a_zero_bandwidth_is_refused(bandwidth: int) -> None:
    with pytest.raises(ValueError, match='bandwidth'):
        legendre_table(bandwidth, 4)


# ------------------------------------------------------------------- Wigner


@pytest.mark.parametrize('degree', [0, 1, 2, 5, 12])
def test_wigner_d_is_orthogonal(degree: int) -> None:
    for angle in (0.0, 0.4, math.pi / 2, 2.2, math.pi):
        matrix = wigner_d(degree, angle)
        size = 2 * degree + 1
        assert matrix.shape == (size, size)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(size), atol=1e-12)


def test_wigner_d_at_zero_is_the_identity() -> None:
    for degree in (0, 1, 4, 9):
        np.testing.assert_allclose(
            wigner_d(degree, 0.0), np.eye(2 * degree + 1), atol=1e-14
        )


def test_wigner_d_matches_the_closed_form_at_degree_one() -> None:
    """Checked against the textbook matrix, ordered by ascending m."""
    for angle in (0.0, 0.3, 1.2, math.pi):
        cosine, sine = math.cos(angle), math.sin(angle)
        root = math.sqrt(2.0)
        descending = np.array(
            [
                [(1 + cosine) / 2, -sine / root, (1 - cosine) / 2],
                [sine / root, cosine, -sine / root],
                [(1 - cosine) / 2, sine / root, (1 + cosine) / 2],
            ]
        )
        np.testing.assert_allclose(
            wigner_d(1, angle), descending[::-1, ::-1], atol=1e-13
        )


def test_wigner_d_composes() -> None:
    for degree in (1, 3, 6):
        left, right = 0.4, 1.1
        np.testing.assert_allclose(
            wigner_d(degree, left) @ wigner_d(degree, right),
            wigner_d(degree, left + right),
            atol=1e-12,
        )


def test_the_quarter_turn_identity_holds() -> None:
    """The identity fast rotational matching is built on.

    It turns the one Euler angle that is not already an exponential into
    one, which is what makes the correlation a Fourier series in all three
    angles and so computable by a single inverse FFT.
    """
    for degree in range(10):
        for angle in (0.0, 0.4, 1.9, 2.7, math.pi):
            np.testing.assert_allclose(
                quarter_turn_identity(degree, angle),
                wigner_d(degree, angle),
                atol=1e-11,
            )


def test_half_turn_table_is_laid_out_for_the_fft() -> None:
    bandwidth = 7
    table = half_turn_table(bandwidth)
    assert table.shape == (bandwidth, 2 * bandwidth, 2 * bandwidth)

    index = orders(bandwidth)
    for degree in range(bandwidth):
        block = wigner_d(degree, math.pi / 2)
        for row, left in enumerate(index):
            for column, right in enumerate(index):
                if abs(int(left)) > degree or abs(int(right)) > degree:
                    assert table[degree, row, column] == 0.0
                else:
                    assert table[degree, row, column] == pytest.approx(
                        block[int(left) + degree, int(right) + degree]
                    )


def test_a_negative_degree_is_refused() -> None:
    with pytest.raises(ValueError, match='degree'):
        wigner_d(-1, 0.0)


# ---------------------------------------------------------------- transform


def _sphere(
    coefficients: np.ndarray, bandwidth: int, nodes: int, azimuths: int
) -> np.ndarray:
    """A band-limited function on the sampling grid, built with scipy."""
    cosine, _ = gauss_legendre(nodes)
    colatitude = np.arccos(cosine)[:, None]
    azimuth = (2.0 * np.pi * np.arange(azimuths) / azimuths)[None, :]

    values = np.zeros((nodes, azimuths), dtype=np.complex128)
    for degree in range(bandwidth):
        for column, order in enumerate(orders(bandwidth)):
            if abs(int(order)) > degree:
                continue
            values += coefficients[degree, column] * sp.sph_harm_y(
                degree, int(order), colatitude, azimuth
            )
    return values


def test_the_transform_recovers_known_coefficients(
    rng: np.random.Generator,
) -> None:
    """Build a band-limited sphere from coefficients and get them back.

    The quadrature is exact for such a function, so this is not a tolerance
    on an approximation: it is the transform being the transform.
    """
    bandwidth = 8
    basis = make_basis([1.0], bandwidth=bandwidth, size=16)
    nodes, azimuths = bandwidth, 2 * bandwidth

    truth = np.zeros((bandwidth, 2 * bandwidth), dtype=np.complex128)
    for degree in range(bandwidth):
        for column, order in enumerate(orders(bandwidth)):
            if abs(int(order)) <= degree:
                truth[degree, column] = rng.normal() + 1j * rng.normal()

    shells = _sphere(truth, bandwidth, nodes, azimuths)[None]
    found = np.asarray(expand_shells(shells, basis), dtype=np.complex128)
    found = found[0] / float(basis.radial[0])

    np.testing.assert_allclose(found, truth, atol=1e-6)


def test_the_transform_is_linear(rng: np.random.Generator) -> None:
    basis = make_basis([2.0, 4.0], bandwidth=6, size=24)
    left = rng.normal(size=(2, 6, 12))
    right = rng.normal(size=(2, 6, 12))

    np.testing.assert_allclose(
        np.asarray(expand_shells(left + 3.0 * right, basis)),
        np.asarray(expand_shells(left, basis))
        + 3.0 * np.asarray(expand_shells(right, basis)),
        atol=1e-5,
    )


def test_a_radial_volume_has_only_its_lowest_degree() -> None:
    """A volume that does not change with direction is degree zero alone.

    This is the test of the whole chain, interpolation included: nothing
    but l = 0 may survive.
    """
    size = 48
    axis = np.arange(size) - size // 2
    radius = np.sqrt(
        axis[:, None, None] ** 2
        + axis[None, :, None] ** 2
        + axis[None, None, :] ** 2
    )
    volume = np.exp(-0.01 * radius**2)

    basis = make_basis(np.arange(1.0, 13.0), bandwidth=8, size=size)
    coefficients = np.asarray(expand(volume, basis))

    lowest = np.abs(coefficients[:, 0, 0]).max()
    rest = np.abs(coefficients[:, 1:, :]).max()
    assert lowest > 0.1
    assert rest < 1e-3 * lowest


def test_the_expansion_is_jittable() -> None:
    basis = make_basis([3.0, 6.0], bandwidth=5, size=20)
    volume = np.zeros((20, 20, 20))
    volume[10, 10, 10] = 1.0

    eager = np.asarray(expand(volume, basis))
    compiled = np.asarray(jax.jit(expand)(volume, basis))
    np.testing.assert_allclose(eager, compiled, atol=1e-6)


def test_complex_volumes_are_read_in_both_parts(
    rng: np.random.Generator,
) -> None:
    """Fourier volumes are complex, and interpolation has to respect that."""
    size = 16
    volume = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    basis = make_basis([2.0, 5.0], bandwidth=4, size=size)

    both = np.asarray(expand(volume, basis))
    real = np.asarray(expand(volume.real, basis))
    imaginary = np.asarray(expand(volume.imag, basis))
    np.testing.assert_allclose(both, real + 1j * imaginary, atol=1e-5)


def test_oversampling_keeps_the_same_orders(
    rng: np.random.Generator,
) -> None:
    """Sampling a sphere finer must not move its coefficients.

    The orders wanted sit at different columns of a longer FFT, and the
    column for the Nyquist order has no degree to go with it, so this is
    really a test that the slicing lines the two layouts up.
    """
    bandwidth = 4
    truth = np.zeros((bandwidth, 2 * bandwidth), dtype=np.complex128)
    for degree in range(bandwidth):
        for column, order in enumerate(orders(bandwidth)):
            if abs(int(order)) <= degree:
                truth[degree, column] = rng.normal() + 1j * rng.normal()

    plain = make_basis([1.0], bandwidth, size=16)
    rich = make_basis([1.0], bandwidth, size=16, sampling_bandwidth=11)

    for basis, nodes in ((plain, bandwidth), (rich, 11)):
        shells = _sphere(truth, bandwidth, nodes, 2 * nodes)[None]
        found = np.asarray(expand_shells(shells, basis), dtype=np.complex128)
        # JAX computes in single precision unless told otherwise, so
        # this is the dtype's floor rather than the method's.
        np.testing.assert_allclose(
            found[0] / float(basis.radial[0]), truth, atol=1e-6
        )


def test_the_grid_lands_on_the_centre() -> None:
    grid = shell_grid((0.0,), nodes=4, azimuths=8, size=32)
    np.testing.assert_allclose(grid, 16.0)


def test_shell_directions_are_unit_vectors() -> None:
    directions = shell_directions(8, 16)
    assert directions.shape == (8, 16, 3)
    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=-1), 1.0, atol=1e-14
    )


def test_a_basis_refuses_nonsense() -> None:
    with pytest.raises(ValueError, match='bandwidth'):
        make_basis([1.0], bandwidth=0, size=8)
    with pytest.raises(ValueError, match='bandwidth'):
        make_basis([1.0], bandwidth=8, size=8, sampling_bandwidth=4)
    with pytest.raises(ValueError, match='radius'):
        make_basis([-1.0], bandwidth=4, size=8)
    with pytest.raises(ValueError, match='sequence'):
        make_basis([[1.0, 2.0]], bandwidth=4, size=8)
    with pytest.raises(ValueError, match='bandwidth'):
        Basis(grid=None, colatitude=None, radial=None, bandwidth=0, azimuths=2)  # type: ignore[arg-type]
