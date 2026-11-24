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

import numpy as np
from numpy.typing import NDArray

from ._transfers import to_host
from .averaging import fourier_shell_correlation, resolution_at
from .blocks import (
    Aligned,
    align_in_blocks,
    average_in_blocks,
    block_size,
    spectra_on_device,
)
from .engine import Engine, Search
from .graph import bridges, build_pairs, is_connected, pieces
from .matching.refine import refine_alignment
from .particles import Subtomograms, recentred_on_host
from .symmetry import detect_symmetry
from .synchronization.rotations import synchronize_rotations
from .synchronization.translations import synchronize_shifts

__all__ = ['Design', 'Result', 'memory_in_bytes', 'run']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Design(Search):
    """Everything the ab-initio pipeline can be told: how pairs are
    searched, which `Search` says, and which pairs there are and how they
    are put together."""

    partners: int = 16
    """Roughly how many others each particle is aligned to. Clamped to one
    fewer than the number of particles, at which point every pair is
    aligned."""
    symmetry: str = 'C1'
    """Point group of the particle."""
    rounds: int = 8
    """Times to snap, synchronize and weigh the pairs.

    Where most pairs are wrong, the weights take a while to settle. On
    the second picking of EMPIAR-10304 at bin 4, four rounds put 83
    particles in a hundred within fifteen degrees, eight put 87, and
    sixteen and thirty-two no more. On the first picking four were
    already enough. A round of a hundred thousand pairs is under a
    second."""
    reject_above: float | None = None
    """Also drop pairs whose residual exceeds this [deg], measured modulo
    the symmetry. None keeps every pair, and each is weighed by how likely
    it is to be right; see
    `frmony.synchronization.rotations.inlier_probabilities`."""
    seed: int = 0
    """Which pairs get chosen."""
    refine: bool = False
    """Polish each pair against the score itself after the grid search.
    Off by default; `frmony.matching.refine` says why."""
    passes: int = 1
    """Times to align, synchronize and bring the particles back to the
    centres found. One pass leaves them where they were picked. A pair
    of particles that are off centre is harder to align than a pair that
    is not, so the later passes align more of them: on EMPIAR-10304,
    with particles thirty angstroms off, 59 in a hundred came out within
    fifteen degrees after one pass and 73 after three."""
    detect: bool = True
    """Report the symmetry the measurements suggest, whatever was asked
    for."""

    average: bool = True
    """Average the particles over the band analysed, once they are
    aligned, and say how far its halves agree. A caller that averages them
    otherwise, as the command line does at the size they were extracted
    at, has no use for it."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.partners < 1:
            raise ValueError('Each particle needs at least one partner.')
        if self.passes < 1:
            raise ValueError('There is at least one pass.')


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
    structure: NDArray[np.complex64] | None
    """The average over the band analysed, as a centred spectrum, if one
    was asked for; see `Design.average`."""
    support: NDArray[np.float32] | None
    """How much weight stands behind each of its frequencies."""
    frequency: NDArray[np.float64] | None
    """Shells of the resolution estimate, as fractions of the sampling
    rate."""
    correlation: NDArray[np.float64] | None
    """Shell correlation between two halves of the data."""
    resolution: float
    """Where that correlation falls below 0.143 [A], and not a number if
    nothing was averaged."""
    detected: str = 'C1'
    """The point group the triangle residuals suggest. Only worth reading
    when `symmetry_doubted` is set: where nothing argues against the
    group asked for, the groups score within noise of one another and the
    name is whichever came out on top."""
    symmetry_doubted: bool = False
    """Whether the pairs argue against the symmetry asked for."""
    confidence: NDArray[np.float64] | None = None
    """For each particle, the share of its pairs that look right, and
    nought for a particle outside the piece the trusted pairs hold
    together, whose orientation means nothing. On EMPIAR-10304 it told
    picks of a ribosome from picks of something else with an area under
    the curve of 0.91 to 0.93."""
    band_limit: float = float('inf')
    """The finest detail the analysed band holds [A]. A resolution of
    infinity means the halves agreed all the way to it."""
    scores_by_symmetry: dict[str, float] = field(default_factory=dict)
    """How much of those residuals each candidate group explains."""


def run(subtomograms: Subtomograms, design: Design | None = None) -> Result:
    """Align, synchronize and average a set of subtomograms.

    Args:
        subtomograms: What `frmony.particles.load_subtomograms` built.
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
    if count < 2:
        raise ValueError(
            f'Aligning pairs needs at least two particles, not {count}.'
        )
    generator = np.random.default_rng(design.seed)

    left, right = build_pairs(count, design.partners, generator)
    if not is_connected(count, left, right):
        raise ValueError(
            'The pairs chosen leave the particles in more than one piece, '
            'so there is nothing relating one piece to another. Ask for '
            'more partners.'
        )
    _LOGGER.info('Aligning %d pairs of %d particles.', len(left), count)

    engine = Engine.build(subtomograms.band, design)
    size = block_size(count, engine.per_particle(), memory_in_bytes(design))
    applied = np.zeros((count, 3))
    for number in range(design.passes):
        # The particles, brought back to where they are believed to be,
        # stay on the host. The device is handed them a block at a time,
        # or all at once if they fit; see `frmony.blocks`.
        data, weight = recentred_on_host(
            subtomograms,
            applied if number else None,
            design.batch,
            noise=design.noise_spectrum,
        )
        # The first pass has to reach as far as the particles may be off.
        # After it they have been brought back to where the pairs put
        # them, and one step of the lattice is enough.
        reach = design.max_shift if number == 0 else engine.spacing
        aligned = _align_pass(
            engine, data, weight, size, left, right, reach, number
        )
        measurements = aligned.rotation.astype(np.float64)
        measured_shifts = aligned.shift.astype(np.float64)
        scores = aligned.score.astype(np.float64)
        if design.refine:
            _refine(
                subtomograms,
                left,
                right,
                measurements,
                measured_shifts,
                scores,
            )
        # A correlation runs from minus one to one; a weight does not.
        weights = np.clip(0.5 * (1.0 + scores), 0.0, 1.0)

        detected, by_symmetry, doubted = _detect_symmetry(
            design, measurements, left, right, generator
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
            improper=design.mirrors,
        )
        orientations = synchronized.orientations
        kept = synchronized.kept
        held = _report_trust(count, left[kept], right[kept])

        # The centres are weighed as the orientations were: by how likely
        # each pair is to be right, as well as by its score.
        trusted = (
            weights
            if synchronized.inliers is None
            else weights * synchronized.inliers
        )
        centres, _ = synchronize_shifts(
            count,
            left[kept],
            right[kept],
            measured_shifts[kept],
            orientations,
            trusted[kept],
        )
        if number < design.passes - 1:
            # Bring every particle back to where the pairs put it, and go
            # again: a pair whose particles are centred is a pair the
            # rotational search can see.
            applied = applied + centres
            _LOGGER.info(
                'Pass %d moved the centres by a median of %.2f voxels of the '
                'band.',
                number + 1,
                float(np.median(np.linalg.norm(centres, axis=1))),
            )

    # The band was cropped, so a shell's frequency is a fraction of the
    # cropped sampling rate; the pixel size that goes with it is coarser
    # than the one the particles were extracted at by the same factor.
    sampled = subtomograms.pixel_size * subtomograms.scale
    averaged = (
        _average(count, size, data, weight, orientations, centres, sampled)
        if design.average
        else None
    )

    return Result(
        orientations=orientations,
        shifts=(applied + centres) * subtomograms.scale,
        left=left,
        right=right,
        scores=scores,
        measurements=measurements,
        measured_shifts=measured_shifts,
        residuals=synchronized.residuals,
        kept=kept,
        structure=None if averaged is None else averaged.structure,
        support=None if averaged is None else averaged.support,
        frequency=None if averaged is None else averaged.frequency,
        correlation=None if averaged is None else averaged.correlation,
        resolution=float('nan') if averaged is None else averaged.resolution,
        detected=detected,
        scores_by_symmetry=by_symmetry,
        symmetry_doubted=doubted,
        confidence=_confidence(count, left, right, synchronized.inliers, held),
        band_limit=2.0 * sampled,
    )


@dataclass(frozen=True)
class _Averaged:
    structure: NDArray[np.complex64]
    support: NDArray[np.float32]
    frequency: NDArray[np.float64]
    correlation: NDArray[np.float64]
    resolution: float


def _average(
    count: int,
    size: int,
    data: NDArray[np.complex64],
    weight: NDArray[np.float32],
    orientations: NDArray[np.float64],
    centres: NDArray[np.float64],
    sampled: float,
) -> _Averaged:
    """The average over the band, and how far its halves agree."""
    averaged = average_in_blocks(
        count,
        size,
        spectra_on_device(data, weight, size),
        orientations,
        centres,
    )
    frequency, correlation = fourier_shell_correlation(*averaged.halves)
    resolution = resolution_at(frequency, correlation, sampled)
    _report_resolution(resolution, 2.0 * sampled)
    return _Averaged(
        structure=averaged.structure,
        support=averaged.support,
        frequency=frequency,
        correlation=correlation,
        resolution=resolution,
    )


def memory_in_bytes(search: Search) -> float | None:
    """The device memory a search was told to plan for, in bytes."""
    return None if search.memory is None else search.memory * 2**30


def _align_pass(
    engine: Engine,
    data: NDArray[np.complex64],
    weight: NDArray[np.float32],
    size: int,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    reach: float,
    number: int,
) -> Aligned:
    """Align every pair once, a block of particles at a time."""
    joint = engine.joint(reach)
    if joint is not None:
        _LOGGER.info(
            'Pass %d: %d trial shifts, %.2f voxels of the band apart.',
            number + 1,
            joint.lattice.shape[0],
            engine.spacing,
        )
    started = time.perf_counter()
    aligned = align_in_blocks(
        len(data),
        size,
        engine.particles(data, weight, size),
        left,
        right,
        engine.aligner(joint),
        engine.batch,
    )
    elapsed = time.perf_counter() - started
    _LOGGER.info(
        'Aligned %d pairs in %.1f s, %.2f ms each.',
        len(left),
        elapsed,
        1e3 * elapsed / max(len(left), 1),
    )
    return aligned


def _refine(
    subtomograms: Subtomograms,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    measurements: NDArray[np.float64],
    measured_shifts: NDArray[np.float64],
    scores: NDArray[np.float64],
) -> None:
    """Polish every pair against the score itself, in place."""
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
        measurements[index] = to_host(turn, dtype=np.float64)
        measured_shifts[index] = to_host(offset, dtype=np.float64)
        scores[index] = float(value)
    _LOGGER.info(
        'Refined %d pairs in %.1f s.',
        len(left),
        time.perf_counter() - started,
    )


def _detect_symmetry(
    design: Design,
    measurements: NDArray[np.float64],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    generator: np.random.Generator,
) -> tuple[str, dict[str, float], bool]:
    """The group the pairs look like, its scores, and whether they argue
    against the one asked for."""
    if not design.detect:
        return 'C1', {}, False
    detected, by_symmetry = detect_symmetry(
        measurements, left, right, generator
    )
    # The detector reads angles, not axes, so it cannot tell a dihedral
    # group from the cyclic one inside it. Warning on a different name
    # would cry wolf. What is worth saying is that the group asked for
    # explains the residuals badly.
    asked = by_symmetry.get(design.symmetry)
    best = max(by_symmetry.values(), default=0.0)
    doubted = asked is not None and asked < best - 0.25
    if doubted:
        _LOGGER.warning(
            'The pairs look more like %s (%.2f) than %s (%.2f). If the '
            'symmetry is wrong the orientations will be too.',
            detected,
            best,
            design.symmetry,
            asked,
        )
    return detected, by_symmetry, doubted


def _report_trust(
    count: int, left: NDArray[np.int64], right: NDArray[np.int64]
) -> NDArray[np.bool_]:
    """Say how the trusted pairs hold the particles together, and return
    which particles are in the piece they hold most of."""
    piece = pieces(count, left, right)
    sizes = np.bincount(piece)
    # A piece of its own frame only matters if it is big enough to show
    # in the average; a hundredth of the particles is not.
    matters = max(2, int(np.ceil(0.01 * count)))
    if len(sizes) > 1:
        _LOGGER.info(
            'The trusted pairs hold %d of the %d particles together. The '
            'other %d stand apart, %d of them alone, the rest in pieces of '
            'at most %d: none of their pairs looks right, which is what a '
            'pick of something else looks like, and their orientations '
            'mean nothing.',
            int(sizes[0]),
            count,
            count - int(sizes[0]),
            int(np.sum(sizes == 1)),
            int(sizes[1]),
        )
    if len(sizes) > 1 and sizes[1] >= matters:
        _LOGGER.warning(
            '%d particles hang together apart from the %d of the largest '
            'piece. Their pairs agree with one another but nothing relates '
            'them to the rest, so they sit in a frame of their own and blur '
            'the average. They may be a second kind of particle. More '
            'partners would tell.',
            int(sizes[1]),
            int(sizes[0]),
        )
    weak = bridges(count, left, right, smallest=matters)
    if len(weak):
        _LOGGER.warning(
            '%d of the trusted pairs are each the only thing holding a '
            'group of at least %d particles to the rest. What those groups '
            'say about each other rests on one measurement apiece, which '
            'nothing can contradict, so it may be wrong without any '
            'residual showing it. Ask for more partners.',
            len(weak),
            matters,
        )
    held: NDArray[np.bool_] = piece == 0
    return held


def _confidence(
    count: int,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    inliers: NDArray[np.float64] | None,
    held: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """The share of each particle's pairs that look right, and nought
    outside the piece the trusted pairs hold together."""
    trust = np.ones(len(left)) if inliers is None else inliers
    summed = np.bincount(left, trust, count) + np.bincount(right, trust, count)
    touched = np.bincount(left, minlength=count) + np.bincount(
        right, minlength=count
    )
    confidence: NDArray[np.float64] = np.where(
        held, summed / np.maximum(touched, 1), 0.0
    )
    return confidence


def _report_resolution(resolution: float, band_limit: float) -> None:
    if np.isfinite(resolution):
        _LOGGER.info('Resolution by half-set correlation: %.1f A.', resolution)
    else:
        _LOGGER.info(
            'The halves agree to the edge of the band analysed, %.1f A, '
            'which is as far as this can see. They were aligned together, '
            'so it is a bound on how well they agree, not a gold standard. '
            '--band looks further.',
            band_limit,
        )
