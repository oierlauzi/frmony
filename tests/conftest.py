"""Simulated subtomograms, for tests that need something to align."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261120)


def phantom(size: int, blobs: int, rng: np.random.Generator) -> np.ndarray:
    """A lumpy, asymmetric structure: Gaussian blobs in a ball.

    Asymmetric on purpose, so that a rotational search has one answer and
    not several.
    """
    axis = np.arange(size) - size // 2
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)

    volume = np.zeros((size,) * 3)
    radius = 0.28 * size
    for _ in range(blobs):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        centre = direction * radius * rng.uniform(0.3, 1.0)
        width = rng.uniform(0.05, 0.10) * size
        squared = np.sum((grid - centre[::-1]) ** 2, axis=-1)
        volume += rng.uniform(0.5, 1.5) * np.exp(-squared / (2 * width**2))
    return volume


def wedge_mask(size: int, tilt: float) -> np.ndarray:
    """A single-tilt missing wedge, as a weight in Fourier space.

    One where a tilt series over ``[-tilt, tilt]`` measures, zero in the
    wedge it never reaches. Centred, indexed ``[z, y, x]``.
    """
    axis = np.arange(size) - size // 2
    z = axis[:, None, None]
    x = axis[None, None, :]
    return (np.abs(z) <= np.abs(x) * np.tan(np.radians(tilt))).astype(
        np.float32
    ) * np.ones((size, size, size), dtype=np.float32)


def spectrum(volume: np.ndarray) -> np.ndarray:
    """The centred Fourier transform of a centred volume."""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))


def rotate_volume(volume: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Turn a volume about its centre, by trilinear interpolation.

    ``out(x) = volume(R^-1 x)``, with ``x`` measured from the centre and
    ordered ``(x, y, z)`` while the array is indexed ``[z, y, x]``.
    """
    from scipy.ndimage import affine_transform

    size = volume.shape[-1]
    centre = np.array([size // 2] * 3, dtype=np.float64)
    # In index order the matrix is the rotation with its axes reversed.
    matrix = rotation[::-1, ::-1].T
    offset = centre - matrix @ centre
    # Cubic, not linear: a linear rotation blurs by an amount that
    # depends on the angle, which would show up as an error of the
    # matching rather than of the simulation.
    return affine_transform(
        volume, matrix, offset=offset, order=3, mode='constant', cval=0.0
    )


@dataclass(frozen=True)
class Simulation:
    """Subtomograms of one structure, at known orientations."""

    data: np.ndarray
    """Centred spectra, ``(particles, size, size, size)``."""
    weight: np.ndarray
    """Fourier weights, the same shape, real."""
    rotations: np.ndarray
    """The true orientation of each particle, ``(particles, 3, 3)``."""
    shifts: np.ndarray
    """The true centre of each particle, ``(particles, 3)``, ``(x, y, z)``."""
    structure: np.ndarray
    """The phantom they are all views of."""


def simulate(
    rng: np.random.Generator,
    count: int,
    size: int = 32,
    tilt: float = 60.0,
    noise: float = 0.0,
    shift: float = 0.0,
    blobs: int = 7,
) -> Simulation:
    """Build subtomograms of one structure at random orientations.

    Each is the structure turned, shifted, and then robbed of the
    frequencies its own missing wedge never measured. The wedge is fixed in
    the frame of the tomogram, so it does *not* turn with the particle,
    which is the whole difficulty the score has to handle.
    """
    from syncsta.geometry import random_rotations

    structure = phantom(size, blobs, rng)
    wedge = wedge_mask(size, tilt)
    rotations = random_rotations(rng, count)
    shifts = (
        rng.uniform(-shift, shift, size=(count, 3))
        if shift > 0
        else np.zeros((count, 3))
    )

    data = np.empty((count, size, size, size), dtype=np.complex64)
    for index in range(count):
        turned = rotate_volume(structure, rotations[index])
        if shift > 0.0:
            moved = np.roll(
                turned,
                np.round(shifts[index][::-1]).astype(int),
                axis=(0, 1, 2),
            )
            shifts[index] = np.round(shifts[index])
        else:
            moved = turned
        if noise > 0.0:
            moved = moved + rng.normal(0.0, noise * moved.std(), moved.shape)
        values = spectrum(moved) * wedge
        values[size // 2, size // 2, size // 2] = 0.0
        data[index] = values.astype(np.complex64)

    weight = np.broadcast_to(wedge, data.shape).astype(np.float32)
    return Simulation(
        data=data,
        weight=np.ascontiguousarray(weight),
        rotations=rotations,
        shifts=shifts,
        structure=structure,
    )
