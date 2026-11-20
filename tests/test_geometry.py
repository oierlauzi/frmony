"""The conventions of orientations, and the maps between their forms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from syncsta.geometry import (
    euler_to_matrix,
    exponential_map,
    logarithmic_map,
    matrix_to_euler,
    project_to_rotation,
    random_rotations,
    relative_angle,
    rotation_angle,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261120)


def test_euler_to_matrix_gives_rotations(rng: np.random.Generator) -> None:
    angles = rng.uniform(-math.pi, math.pi, size=(32, 3))
    matrices = euler_to_matrix(angles[:, 0], angles[:, 1], angles[:, 2])

    assert matrices.shape == (32, 3, 3)
    identity = np.einsum('nij,nkj->nik', matrices, matrices)
    expected = np.broadcast_to(np.eye(3), identity.shape)
    np.testing.assert_allclose(identity, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-12)


def test_euler_to_matrix_follows_relion() -> None:
    """The matrix is Rz(psi) @ Ry(tilt) @ Rz(rot), as RELION writes it.

    This is the one convention the whole package hangs from, so it is
    checked against the composition written out rather than against itself.
    """

    def about_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def about_y(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    rot, tilt, psi = 0.3, 1.1, -0.7
    expected = about_z(psi) @ about_y(tilt) @ about_z(rot)
    np.testing.assert_allclose(
        euler_to_matrix(rot, tilt, psi), expected, atol=1e-12
    )


def test_euler_round_trips(rng: np.random.Generator) -> None:
    matrices = random_rotations(rng, 256)
    rot, tilt, psi = matrix_to_euler(matrices)

    np.testing.assert_allclose(
        euler_to_matrix(rot, tilt, psi), matrices, atol=1e-9
    )
    assert np.all(tilt >= -1e-12)
    assert np.all(tilt <= math.pi + 1e-12)


@pytest.mark.parametrize('tilt', [0.0, math.pi])
def test_euler_round_trips_at_the_poles(tilt: float) -> None:
    """Where rot and psi merge, the matrix still has to come back."""
    for turn in (-2.0, -0.4, 0.0, 0.9, 3.0):
        matrix = euler_to_matrix(0.0, tilt, turn)
        rot, new_tilt, psi = matrix_to_euler(matrix)
        np.testing.assert_allclose(
            euler_to_matrix(rot, new_tilt, psi), matrix, atol=1e-9
        )


def test_rotation_angle_is_the_turn() -> None:
    for angle in (0.0, 0.3, 1.0, 2.5, math.pi):
        matrix = euler_to_matrix(0.0, angle, 0.0)
        assert rotation_angle(matrix) == pytest.approx(angle, abs=1e-9)


def test_rotation_angle_survives_conjugation(rng: np.random.Generator) -> None:
    """The property that lets symmetry be detected without any orientation."""
    rotations = random_rotations(rng, 64)
    frames = random_rotations(rng, 64)
    conjugated = frames @ rotations @ np.swapaxes(frames, -2, -1)

    np.testing.assert_allclose(
        rotation_angle(conjugated), rotation_angle(rotations), atol=1e-9
    )


def test_relative_angle_is_symmetric(rng: np.random.Generator) -> None:
    left = random_rotations(rng, 64)
    right = random_rotations(rng, 64)

    np.testing.assert_allclose(
        relative_angle(left, right), relative_angle(right, left), atol=1e-9
    )
    np.testing.assert_allclose(relative_angle(left, left), 0.0, atol=1e-7)


def test_project_to_rotation_keeps_rotations(rng: np.random.Generator) -> None:
    rotations = random_rotations(rng, 64)
    np.testing.assert_allclose(
        project_to_rotation(rotations), rotations, atol=1e-12
    )


def test_project_to_rotation_refuses_reflections(
    rng: np.random.Generator,
) -> None:
    """A reflection must come back as a rotation, not be passed through.

    The polar factor of a plain SVD would keep the reflection. Getting this
    wrong gives a synchronization over O(3) that silently mirrors half the
    particles.
    """
    reflected = random_rotations(rng, 64).copy()
    reflected[:, :, 2] *= -1.0
    assert np.all(np.linalg.det(reflected) < 0.0)

    projected = project_to_rotation(reflected)
    np.testing.assert_allclose(np.linalg.det(projected), 1.0, atol=1e-9)


def test_project_to_rotation_denoises(rng: np.random.Generator) -> None:
    rotations = random_rotations(rng, 64)
    noisy = rotations + 0.01 * rng.normal(size=rotations.shape)
    recovered = project_to_rotation(noisy)

    np.testing.assert_allclose(np.linalg.det(recovered), 1.0, atol=1e-9)
    assert np.all(relative_angle(rotations, recovered) < 0.05)


def test_exponential_map_round_trips(rng: np.random.Generator) -> None:
    rotations = random_rotations(rng, 256)
    vectors = logarithmic_map(rotations)

    assert np.all(np.linalg.norm(vectors, axis=-1) <= math.pi + 1e-9)
    np.testing.assert_allclose(exponential_map(vectors), rotations, atol=1e-8)


def test_exponential_map_is_finite_at_zero() -> None:
    """A gradient taken through the refinement passes through here."""
    tiny = np.array([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, -1e-8, 1e-8]])
    matrices = exponential_map(tiny)

    assert np.all(np.isfinite(matrices))
    np.testing.assert_allclose(matrices[0], np.eye(3), atol=1e-15)


def test_logarithmic_map_handles_half_turns() -> None:
    """At pi the antisymmetric part vanishes and the axis hides elsewhere."""
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.3, -0.7, 0.2],
        ]
    )
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    rotations = exponential_map(math.pi * axes)

    recovered = exponential_map(logarithmic_map(rotations))
    np.testing.assert_allclose(recovered, rotations, atol=1e-6)


def test_random_rotations_cover_the_sphere(rng: np.random.Generator) -> None:
    """Uniform on SO(3) means the viewing axis is uniform on the sphere."""
    rotations = random_rotations(rng, 20000)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-9)

    axis = rotations[:, 2, :]
    np.testing.assert_allclose(axis.mean(axis=0), 0.0, atol=0.03)
    # cos(theta) is uniform on [-1, 1] for a uniform direction.
    assert np.std(axis[:, 2]) == pytest.approx(1.0 / math.sqrt(3.0), abs=0.02)
