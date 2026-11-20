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
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._transfers import to_device, to_host
from .averaging import average, fourier_shell_correlation, resolution_at
from .graph import bridges, build_pairs, is_connected
from .harmonics.shells import band_radii
from .harmonics.transform import Basis, make_basis
from .matching.correlation import Expansion, expand_particle
from .matching.pairwise import (
    Joint,
    Settings,
    align_pairs,
    lattice_spacing,
    shift_lattice,
)
from .matching.refine import refine_alignment
from .particles import Subtomograms, recentred
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
    rounds: int = 4
    """Times to snap, synchronize and reject."""
    reject_above: float | None = 90.0
    """Drop pairs whose residual exceeds this [deg], measured modulo the
    symmetry. None keeps every pair."""
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
    phases: bool = False
    """Keep the phases in the first, rotational stage. It then needs the
    particles roughly centred, and finds rotations that the amplitudes
    are too noisy to; see `frmony.matching.expand_particle`."""
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
    """The point group the triangle residuals suggest."""
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
    applied = np.zeros((count, 3))
    for number in range(design.passes):
        # The one crossing the particles make. Everything after this reads
        # them from the device, so a thousand particles cost one copy and not
        # a thousand. Expanding them in a Python loop instead, which is what
        # this used to do, sent each particle over and brought its
        # coefficients back, and then sent those over again to align with.
        started = time.perf_counter()
        data, weight = recentred(
            subtomograms, applied if number else None, design.batch
        )
        # Two particles are compared over the frequencies both measured, and
        # not in proportion to how well. A weight that follows the transfer
        # is largest where every tilt overlaps, on the tilt axis and at the
        # lowest frequencies, which is also where what surrounds the particle
        # is strongest, and that is not noise the weight knows about. Counting
        # those frequencies by their weight once more took the pairs aligned
        # correctly on EMPIAR-10304 from eighteen in a hundred to five. The
        # average below does use the weight, as it must to undo the transfer.
        measured = (weight > 0.0).astype(weight.dtype)
        # The weight itself is only wanted again for the average. Holding
        # it here through the alignment is a quarter of what the
        # particles take, for nothing.
        del weight
        # A stretch of the particles at a time. One map over all of
        # them holds every expansion twice while it puts them together,
        # which is the largest thing on the device.
        reach_at_once = max(design.batch, 1) * 32
        pieces = [
            _expand_stretch(
                data[start : start + reach_at_once],
                measured[start : start + reach_at_once],
                basis,
                design.phases,
                max(design.batch, 1),
            )
            for start in range(0, count, reach_at_once)
        ]
        # Put together one leaf at a time, letting go of its pieces
        # before the next, so that only a third is ever held twice.
        layout = jax.tree.structure(pieces[0])
        scattered = [
            list(parts)
            for parts in zip(
                *(jax.tree.leaves(piece) for piece in pieces), strict=True
            )
        ]
        del pieces
        joined = []
        for parts in scattered:
            joined.append(jnp.concatenate(parts))
            parts.clear()
        expansions = jax.tree.unflatten(layout, joined)
        del joined
        # The timing above means nothing without this: JAX dispatches
        # asynchronously, so the loop returns before the work is done.
        jax.tree.map(lambda leaf: leaf.block_until_ready(), expansions)
        _LOGGER.info(
            'Expanded %d particles in %.1f s.',
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
        aligned = align_pairs(
            data,
            measured,
            expansions,
            left,
            right,
            settings,
            batch=design.batch,
            band=in_band,
            joint=joint,
        )
        elapsed = time.perf_counter() - started
        _LOGGER.info(
            'Aligned %d pairs in %.1f s, %.2f ms each.',
            len(left),
            elapsed,
            1e3 * elapsed / max(len(left), 1),
        )

        measurements = to_host(aligned.rotation, dtype=np.float64)
        measured_shifts = to_host(aligned.shift, dtype=np.float64)
        scores = to_host(aligned.score, dtype=np.float64)

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
        if not is_connected(count, left[kept], right[kept]):
            _LOGGER.warning(
                'Rejecting outliers left the particles in more than one piece. '
                'Each piece has its own frame and nothing relates them, so the '
                'average is not meaningful. Ask for more partners, or raise '
                'the rejection threshold.'
            )
        else:
            weak = bridges(count, left[kept], right[kept])
            if len(weak):
                _LOGGER.warning(
                    '%d of the surviving pairs are the only thing holding two '
                    'groups of particles together. Everything those groups say '
                    'about each other rests on one measurement apiece, which '
                    'nothing can contradict, so it may be wrong without any '
                    'residual showing it. Ask for more partners.',
                    len(weak),
                )

        centres, _ = synchronize_shifts(
            count,
            left[kept],
            right[kept],
            measured_shifts[kept],
            orientations,
            weights[kept],
        )
        if number < design.passes - 1:
            # Bring every particle back to where the pairs put it, and go
            # again: a pair whose particles are centred is a pair the
            # rotational search can see.
            applied = applied + centres
            # Let go of this pass before the next is built. Held over,
            # the particles and their expansions are on the device twice.
            del data, measured, expansions, aligned
            _LOGGER.info(
                'Pass %d moved the centres by a median of %.2f voxels of the '
                'band.',
                number + 1,
                float(np.median(np.linalg.norm(centres, axis=1))),
            )

    weight = to_device(subtomograms.weight)
    structure, support = average(data, weight, orientations, centres)
    halves = []
    for half in (np.arange(count) % 2 == 0, np.arange(count) % 2 == 1):
        part, _ = average(
            data[half], weight[half], orientations[half], centres[half]
        )
        halves.append(to_host(part))
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
        shifts=(applied + centres) * subtomograms.scale,
        left=left,
        right=right,
        scores=scores,
        measurements=measurements,
        measured_shifts=measured_shifts,
        residuals=synchronized.residuals,
        kept=kept,
        structure=to_host(structure),
        support=to_host(support),
        frequency=frequency,
        correlation=correlation,
        resolution=resolution,
        detected=detected,
        scores_by_symmetry=by_symmetry,
    )
