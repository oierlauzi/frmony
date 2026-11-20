"""The whole thing, end to end.

Pairs of subtomograms are aligned to one another; orthogonal group
synchronization finds the orientations that agree best with all of those
measurements at once; the centres follow by least squares; and the aligned
particles are averaged in Fourier space, where each supplies the
frequencies the others never measured.

Nothing is aligned against a growing reference at any point, so there is no
reference to be biased by.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import jax
import numpy as np
from numpy.typing import NDArray

from .averaging import average, fourier_shell_correlation, resolution_at
from .graph import build_pairs, is_connected
from .harmonics.shells import band_radii
from .harmonics.transform import make_basis
from .matching.correlation import expand_particle
from .matching.pairwise import Settings, align_pairs
from .matching.refine import refine_alignment
from .particles import Subtomograms
from .symmetry import detect_symmetry
from .synchronization.rotations import synchronize_rotations
from .synchronization.translations import synchronize_shifts

__all__ = ['Design', 'Result', 'run']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Design:
    """Everything the pipeline can be told."""

    partners: int = 16
    """Roughly how many others each particle is aligned to. Clamped to one
    fewer than the number of particles, at which point every pair is
    aligned."""
    bandwidth: int = 32
    """Degrees kept in the spherical expansion. The grid of rotations has a
    step of ``180 / bandwidth`` degrees, which interpolation improves on."""
    candidates: int = 10
    """Rotations carried from the first stage of a pairwise alignment into
    the second."""
    max_shift: float = 6.0
    """Largest shift considered, in voxels of the cropped band."""
    lowest: float = 0.125
    """Innermost sphere of the expansion, as a fraction of the box. The
    frequencies below it carry a particle's mass rather than its
    orientation; see `syncsta.harmonics.band_radii`."""
    symmetry: str = 'C1'
    """Point group of the particle."""
    rounds: int = 4
    """Times to snap, synchronize and reject."""
    reject_above: float | None = 90.0
    """Drop pairs whose residual exceeds this [deg], measured modulo the
    symmetry. None keeps every pair."""
    seed: int = 0
    """Which pairs get chosen."""
    batch: int = 4
    """Pairs aligned at once. Larger is faster until the device runs out of
    room."""
    refine: bool = False
    """Polish each pair against the score itself after the grid search.
    Off by default; `syncsta.matching.refine` says why."""
    detect: bool = True
    """Report the symmetry the measurements suggest, whatever was asked
    for."""

    def __post_init__(self) -> None:
        if self.partners < 1:
            raise ValueError('Each particle needs at least one partner.')
        if not 0.0 <= self.lowest < 0.5:
            raise ValueError('The innermost sphere is inside the box.')


@dataclass(frozen=True)
class Result:
    """What a run found."""

    orientations: NDArray[np.float64]
    """``(particles, 3, 3)``."""
    shifts: NDArray[np.float64]
    """``(particles, 3)`` in voxels of the original box, ``(x, y, z)``."""
    left: NDArray[np.int64]
    """First particle of each pair aligned."""
    right: NDArray[np.int64]
    """Second particle of each pair."""
    scores: NDArray[np.float64]
    """What each pair scored."""
    measurements: NDArray[np.float64]
    """The relative orientation each pair measured, before any
    synchronization, ``(pairs, 3, 3)``. Kept because comparing it with
    what the synchronization settled on is the clearest diagnostic there
    is."""
    measured_shifts: NDArray[np.float64]
    """The shift each pair measured, ``(pairs, 3)``, in voxels of the
    cropped band."""
    residuals: NDArray[np.float64]
    """How far each pair ended up from agreeing [rad], modulo symmetry."""
    kept: NDArray[np.bool_]
    """Which pairs survived rejection."""
    structure: NDArray[np.complex64]
    """The average, as a centred spectrum."""
    support: NDArray[np.float32]
    """How much weight stands behind each of its frequencies."""
    frequency: NDArray[np.float64]
    """Shells of the resolution estimate, as fractions of the sampling
    rate."""
    correlation: NDArray[np.float64]
    """Shell correlation between two halves of the data."""
    resolution: float
    """Where that correlation falls below 0.143 [A]."""
    detected: str = 'C1'
    """The point group the triangle residuals suggest."""
    scores_by_symmetry: dict[str, float] = field(default_factory=dict)
    """How much of those residuals each candidate group explains."""


def run(subtomograms: Subtomograms, design: Design | None = None) -> Result:
    """Align, synchronize and average a set of subtomograms.

    Args:
        subtomograms: What `syncsta.particles.load_subtomograms` built.
        design: What to do. The defaults if not given.

    Returns:
        A `Result`.

    Raises:
        ValueError: If the pairs chosen leave the particles in more than
            one piece, which would give each piece its own unrelated
            frame.
    """
    design = design or Design()
    count = len(subtomograms)
    generator = np.random.default_rng(design.seed)

    left, right = build_pairs(count, design.partners, generator)
    if not is_connected(count, left, right):
        raise ValueError(
            'The pairs chosen leave the particles in more than one piece, '
            'so there is nothing relating one piece to another. Ask for '
            'more partners.'
        )
    _LOGGER.info('Aligning %d pairs of %d particles.', len(left), count)

    basis = make_basis(
        band_radii(subtomograms.band, lowest=design.lowest),
        design.bandwidth,
        subtomograms.band,
        sampling_bandwidth=2 * design.bandwidth,
    )
    started = time.perf_counter()
    expansions = jax.tree.map(
        lambda *leaves: np.stack(leaves),
        *[
            expand_particle(
                subtomograms.data[index], subtomograms.weight[index], basis
            )
            for index in range(count)
        ],
    )
    _LOGGER.info(
        'Expanded %d particles in %.1f s.', count, time.perf_counter() - started
    )

    settings = Settings(
        bandwidth=design.bandwidth,
        candidates=design.candidates,
        max_shift=design.max_shift,
    )
    started = time.perf_counter()
    aligned = align_pairs(
        subtomograms.data,
        subtomograms.weight,
        expansions,
        left,
        right,
        settings,
        batch=design.batch,
    )
    elapsed = time.perf_counter() - started
    _LOGGER.info(
        'Aligned %d pairs in %.1f s, %.0f ms each.',
        len(left),
        elapsed,
        1e3 * elapsed / max(len(left), 1),
    )

    measurements = np.asarray(aligned.rotation, dtype=np.float64)
    measured_shifts = np.asarray(aligned.shift, dtype=np.float64)
    scores = np.asarray(aligned.score, dtype=np.float64)

    if design.refine:
        started = time.perf_counter()
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            turn, offset, value = refine_alignment(
                subtomograms.data[first],
                subtomograms.weight[first],
                subtomograms.data[second],
                subtomograms.weight[second],
                measurements[index],
                measured_shifts[index],
            )
            measurements[index] = np.asarray(turn, dtype=np.float64)
            measured_shifts[index] = np.asarray(offset, dtype=np.float64)
            scores[index] = float(value)
        _LOGGER.info(
            'Refined %d pairs in %.1f s.',
            len(left),
            time.perf_counter() - started,
        )
    # A correlation runs from minus one to one; a weight does not.
    weights = np.clip(0.5 * (1.0 + scores), 0.0, 1.0)

    detected = 'C1'
    by_symmetry: dict[str, float] = {}
    if design.detect:
        detected, by_symmetry = detect_symmetry(
            measurements, left, right, generator
        )
        # The detector reads angles, not axes, so it cannot tell a
        # dihedral group from the cyclic one inside it. Warning on a
        # different name would cry wolf. What is worth saying is that the
        # group asked for explains the residuals badly.
        asked = by_symmetry.get(design.symmetry)
        best = max(by_symmetry.values(), default=0.0)
        if asked is not None and asked < best - 0.25:
            _LOGGER.warning(
                'The pairs look more like %s (%.2f) than %s (%.2f). If the '
                'symmetry is wrong the orientations will be too.',
                detected,
                best,
                design.symmetry,
                asked,
            )

    synchronized = synchronize_rotations(
        count,
        left,
        right,
        measurements,
        weights,
        symmetry=design.symmetry,
        rounds=design.rounds,
        reject_above=(
            None
            if design.reject_above is None
            else float(np.radians(design.reject_above))
        ),
    )
    orientations = synchronized.orientations

    kept = synchronized.kept
    centres, _ = synchronize_shifts(
        count,
        left[kept],
        right[kept],
        measured_shifts[kept],
        orientations,
        weights[kept],
    )

    structure, support = average(
        subtomograms.data, subtomograms.weight, orientations, centres
    )
    halves = []
    for half in (np.arange(count) % 2 == 0, np.arange(count) % 2 == 1):
        part, _ = average(
            subtomograms.data[half],
            subtomograms.weight[half],
            orientations[half],
            centres[half],
        )
        halves.append(np.asarray(part))
    frequency, correlation = fourier_shell_correlation(halves[0], halves[1])
    # The band was cropped, so a shell's frequency is a fraction of the
    # cropped sampling rate; the pixel size that goes with it is coarser
    # than the one the particles were extracted at by the same factor.
    resolution = resolution_at(
        frequency, correlation, subtomograms.pixel_size * subtomograms.scale
    )
    _LOGGER.info('Resolution by half-set correlation: %.1f A.', resolution)

    return Result(
        orientations=orientations,
        shifts=centres * subtomograms.scale,
        left=left,
        right=right,
        scores=scores,
        measurements=measurements,
        measured_shifts=measured_shifts,
        residuals=synchronized.residuals,
        kept=kept,
        structure=np.asarray(structure),
        support=np.asarray(support),
        frequency=frequency,
        correlation=correlation,
        resolution=resolution,
        detected=detected,
        scores_by_symmetry=by_symmetry,
    )
