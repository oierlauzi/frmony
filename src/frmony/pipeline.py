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

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._transfers import to_device, to_host
from .averaging import fourier_shell_correlation, resolution_at
from .blocks import (
    align_in_blocks,
    average_in_blocks,
    block_size,
    pad_to,
    per_particle,
)
from .graph import bridges, build_pairs, is_connected, pieces
from .harmonics.shells import band_radii
from .harmonics.transform import Basis, make_basis
from .matching.correlation import Expansion, expand_particle
from .matching.pairwise import (
    Joint,
    Particles,
    Settings,
    align_chunk,
    lattice_spacing,
    shift_lattice,
)
from .matching.refine import refine_alignment
from .particles import Subtomograms, recentred_on_host
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
    """How far apart the centres of two particles may be, in voxels of
    the cropped band. With the phases kept, rotations and shifts are
    searched at once out to this distance, and the further it reaches
    the more a pair has to stand out, so it should be what the picking
    warrants and no more: on centred particles a reach of nothing
    aligned 40 pairs in a hundred and a reach of five voxels 22."""
    lowest: float = 0.125
    """Innermost sphere of the expansion, as a fraction of the box. The
    frequencies below it carry a particle's mass rather than its
    orientation; see `frmony.harmonics.band_radii`."""
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
    batch: int = 8
    """Pairs aligned at once, and particles expanded at once.

    Measured on a 32 GB device, with particles in a band of 38 and the
    rotations and shifts searched together: 1.6 milliseconds a pair at
    two, 1.4 at four and at eight, 2.4 at sixteen, 3.7 at thirty-two and
    4.4 at a hundred and twenty-eight, compilation included. More at
    once is slower, not faster: each pair in flight holds its own grids
    of every trial shift, and past eight they no longer fit where the
    device works fastest. It was thirty-two, reasoned from a device
    waiting on dispatch and not measured.

    Turn it down if a device runs out of memory."""
    memory: float | None = None
    """Device memory to plan for [GB]. None takes what the device says it
    has. Particles that do not fit are handed to it a block at a time;
    see `frmony.blocks`."""
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
    noise_spectrum: bool = False
    """Divide each particle by the noise spectrum of its acquisition; see
    `frmony.particles.noise_spectra`.

    Off. Measured on pairs put on their true centres it helped, and in a
    whole run, where the shifts are searched, it did harm on both
    datasets it was tried on: on the first picking of EMPIAR-10304,
    displaced by thirty angstroms, 84 particles in a hundred within
    fifteen degrees after one pass without it and 54 with it, and on
    the second picking at bin 4 a set that aligned without it did not
    align at all with it. The spectrum it divides by is the total power,
    signal included, so it takes most off the strong low frequencies,
    which are the ones a shift that is not yet known spoils least."""
    phases: bool = True
    """Keep the phases in the first, rotational stage. They find rotations
    that the amplitudes are too noisy to: on EMPIAR-10304 the amplitudes
    aligned three pairs in a thousand, which is chance. The price is that
    a phased score only sees a pair at the shift it is at, which is why
    rotations and shifts are then searched together; see `max_shift`.
    Turn it off only for clean volumes, where amplitudes are enough and
    need no shift searched."""
    detect: bool = True
    """Report the symmetry the measurements suggest, whatever was asked
    for."""

    def __post_init__(self) -> None:
        if self.partners < 1:
            raise ValueError('Each particle needs at least one partner.')
        if self.passes < 1:
            raise ValueError('There is at least one pass.')
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


@functools.partial(jax.jit, static_argnames=('phases', 'batch'))
def _expand_stretch(
    data: jax.Array,
    measured: jax.Array,
    basis: Basis,
    phases: bool,
    batch: int,
) -> Expansion:
    """The expansions of a stretch of the particles.

    The basis is an argument and not something closed over. Closed over,
    its tables are folded into the compiled program and uploaded again
    each time one is built, by the runtime and under no name.
    """
    expanded: Expansion = jax.lax.map(
        lambda particle: expand_particle(
            particle[0], particle[1], basis, phases
        ),
        (data, measured),
        batch_size=batch,
    )
    return expanded


def _particles_on_device(
    data: NDArray[np.complex64],
    weight: NDArray[np.float32],
    size: int,
    basis: Basis,
    phases: bool,
    batch: int,
) -> Callable[[int, int], Particles]:
    """What hands a block of particles to the device, ready to align.

    Each call dispatches and returns at once: the block travels and is
    expanded while the device finishes what it was given before.
    """

    def make(start: int, stop: int) -> Particles:
        spectra = to_device(pad_to(data[start:stop], size))
        # Two particles are compared over the frequencies both measured,
        # and not in proportion to how well. A weight that follows the
        # transfer is largest where every tilt overlaps, on the tilt axis
        # and at the lowest frequencies, which is also where what
        # surrounds the particle is strongest, and that is not noise the
        # weight knows about. Counting those frequencies by their weight
        # once more took the pairs aligned correctly on EMPIAR-10304 from
        # eighteen in a hundred to five. The average does use the weight,
        # as it must to undo the transfer.
        measured = (to_device(pad_to(weight[start:stop], size)) > 0.0).astype(
            jnp.float32
        )
        return Particles(
            data=spectra,
            weight=measured,
            expansions=_expand_stretch(spectra, measured, basis, phases, batch),
        )

    return make


def _spectra_on_device(
    data: NDArray[np.complex64], weight: NDArray[np.float32], size: int
) -> Callable[[int, int], tuple[jax.Array, jax.Array]]:
    """What hands a block of spectra and weights to the device, to average."""

    def make(start: int, stop: int) -> tuple[jax.Array, jax.Array]:
        return (
            to_device(pad_to(data[start:stop], size)),
            to_device(pad_to(weight[start:stop], size)),
        )

    return make


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
    generator = np.random.default_rng(design.seed)

    left, right = build_pairs(count, design.partners, generator)
    if not is_connected(count, left, right):
        raise ValueError(
            'The pairs chosen leave the particles in more than one piece, '
            'so there is nothing relating one piece to another. Ask for '
            'more partners.'
        )
    _LOGGER.info('Aligning %d pairs of %d particles.', len(left), count)

    radii = band_radii(subtomograms.band, lowest=design.lowest)
    basis = make_basis(
        radii,
        design.bandwidth,
        subtomograms.band,
        sampling_bandwidth=2 * design.bandwidth,
    )
    settings = Settings(
        bandwidth=design.bandwidth,
        candidates=design.candidates,
        max_shift=design.max_shift,
        phases=design.phases,
    )
    # Rotations and shifts are searched at once when the phases are kept
    # and the particles may be further off than the search over shifts
    # reaches by itself. The lattice is searched at half the bandwidth.
    spacing = lattice_spacing(subtomograms.band, float(radii[-1]))
    together = design.phases and design.max_shift >= spacing
    coarse = make_basis(
        radii,
        max(design.bandwidth // 2, 2),
        subtomograms.band,
        sampling_bandwidth=2 * max(design.bandwidth // 2, 2),
    )
    # The shifts are searched over the spheres the rotations were. Built
    # on the device: it is the shape of one particle, and a copy of that
    # shape is how a particle crossing on its own would look.
    offsets = jnp.arange(subtomograms.band) - subtomograms.band // 2
    distance = jnp.sqrt(
        offsets[:, None, None] ** 2
        + offsets[None, :, None] ** 2
        + offsets[None, None, :] ** 2
    )
    in_band = (
        (distance >= float(radii[0])) & (distance <= float(radii[-1]) + 0.5)
    ).astype(jnp.float32)
    batch = max(design.batch, 1)
    size = block_size(
        count,
        per_particle(subtomograms.band, len(radii), design.bandwidth),
        None if design.memory is None else design.memory * 2**30,
    )
    applied = np.zeros((count, 3))
    for number in range(design.passes):
        # The particles, brought back to where they are believed to be,
        # stay on the host. The device is handed them a block at a time,
        # or all at once if they fit; see `frmony.blocks`.
        started = time.perf_counter()
        data, weight = recentred_on_host(
            subtomograms,
            applied if number else None,
            design.batch,
            noise=design.noise_spectrum,
        )
        _LOGGER.info(
            'Prepared %d particles in %.1f s.',
            count,
            time.perf_counter() - started,
        )

        joint = None
        if together:
            # The first pass has to reach as far as the particles may be
            # off. After it they have been brought back to where the pairs
            # put them, and one step of the lattice is enough.
            reach = design.max_shift if number == 0 else spacing
            joint = Joint(
                coarse=coarse,
                fine=basis,
                lattice=to_device(shift_lattice(reach, spacing), jnp.float32),
                spacing=spacing,
            )
            _LOGGER.info(
                'Pass %d: %d trial shifts, %.2f voxels of the band apart.',
                number + 1,
                joint.lattice.shape[0],
                spacing,
            )
        started = time.perf_counter()
        aligned = align_in_blocks(
            count,
            size,
            _particles_on_device(
                data, weight, size, basis, design.phases, batch
            ),
            left,
            right,
            functools.partial(
                align_chunk,
                settings=settings,
                band=in_band,
                joint=joint,
                batch=batch,
            ),
            batch,
        )
        elapsed = time.perf_counter() - started
        _LOGGER.info(
            'Aligned %d pairs in %.1f s, %.2f ms each.',
            len(left),
            elapsed,
            1e3 * elapsed / max(len(left), 1),
        )

        measurements = aligned.rotation.astype(np.float64)
        measured_shifts = aligned.shift.astype(np.float64)
        scores = aligned.score.astype(np.float64)

        if design.refine:
            started = time.perf_counter()
            for index, (first, second) in enumerate(
                zip(left, right, strict=True)
            ):
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
        # A correlation runs from minus one to one; a weight does not.
        weights = np.clip(0.5 * (1.0 + scores), 0.0, 1.0)

        detected = 'C1'
        doubted = False
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
        piece = pieces(count, left[kept], right[kept])
        held = piece == 0
        sizes = np.bincount(piece)
        # A piece of its own frame only matters if it is big enough to
        # show in the average; a hundredth of the particles is not.
        matters = max(2, int(np.ceil(0.01 * count)))
        if len(sizes) > 1:
            _LOGGER.info(
                'The trusted pairs hold %d of the %d particles together. The '
                'other %d stand apart, %d of them alone, the rest in pieces '
                'of at most %d: none of their pairs looks right, which is '
                'what a pick of something else looks like, and their '
                'orientations mean nothing.',
                int(sizes[0]),
                count,
                count - int(sizes[0]),
                int(np.sum(sizes == 1)),
                int(sizes[1]),
            )
        if len(sizes) > 1 and sizes[1] >= matters:
            _LOGGER.warning(
                '%d particles hang together apart from the %d of the largest '
                'piece. Their pairs agree with one another but nothing '
                'relates them to the rest, so they sit in a frame of their '
                'own and blur the average. They may be a second kind of '
                'particle. More partners would tell.',
                int(sizes[1]),
                int(sizes[0]),
            )
        weak = bridges(count, left[kept], right[kept], smallest=matters)
        if len(weak):
            _LOGGER.warning(
                '%d of the trusted pairs are each the only thing holding a '
                'group of at least %d particles to the rest. What those '
                'groups say about each other rests on one measurement '
                'apiece, which nothing can contradict, so it may be wrong '
                'without any residual showing it. Ask for more partners.',
                len(weak),
                matters,
            )

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

    averaged = average_in_blocks(
        count,
        size,
        _spectra_on_device(data, weight, size),
        orientations,
        centres,
    )
    structure, support, halves = (
        averaged.structure,
        averaged.support,
        averaged.halves,
    )
    frequency, correlation = fourier_shell_correlation(halves[0], halves[1])
    # The band was cropped, so a shell's frequency is a fraction of the
    # cropped sampling rate; the pixel size that goes with it is coarser
    # than the one the particles were extracted at by the same factor.
    sampled = subtomograms.pixel_size * subtomograms.scale
    resolution = resolution_at(frequency, correlation, sampled)
    band_limit = 2.0 * sampled
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

    # How much of each particle's evidence looks right.
    trust = (
        np.ones(len(left))
        if synchronized.inliers is None
        else synchronized.inliers
    )
    summed = np.bincount(left, trust, count) + np.bincount(right, trust, count)
    touched = np.bincount(left, minlength=count) + np.bincount(
        right, minlength=count
    )
    confidence = np.where(held, summed / np.maximum(touched, 1), 0.0)

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
        structure=structure,
        support=support,
        frequency=frequency,
        correlation=correlation,
        resolution=resolution,
        detected=detected,
        scores_by_symmetry=by_symmetry,
        symmetry_doubted=doubted,
        confidence=confidence,
        band_limit=band_limit,
    )
