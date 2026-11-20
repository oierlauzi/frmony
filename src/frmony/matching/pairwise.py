"""Aligning one subtomogram to another, in two stages.

The rotation is found first, on amplitudes, which do not move when the
particle does. The best few rotations are kept rather than the best one,
because amplitudes discard the phases and so discriminate less than the
full score: the right rotation is reliably among the best few even when it
is not the best one. Then each candidate gets a single transform, which
gives the correlation at every shift at once and scores the candidate in
full, phases included.

What comes out is a rotation and a shift satisfying the relation
`frmony.geometry` is written in,

    V_i(x) = V_j(A.T @ (x - a)),

which is what the synchronization expects of a pair.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._progress import progress
from .._transfers import to_device
from ..harmonics.shells import sample_shells
from ..harmonics.transform import Basis, expand_shells
from .correlation import Expansion, rotation_score
from .peaks import refine_peak, top_peaks
from .rotation import correlation_grid, radial_overlap
from .translation import frequency_grid, shift_correlation

__all__ = [
    'Alignment',
    'Joint',
    'Settings',
    'align_pair',
    'align_pairs',
    'candidate_rotations',
    'lattice_spacing',
    'shift_lattice',
    'truncate',
]


@dataclass(frozen=True)
class Settings:
    """What to search, and how finely."""

    bandwidth: int = 32
    """Degrees kept in the spherical expansion. The grid of rotations has a
    step of ``180 / bandwidth`` degrees, which interpolation then improves
    on."""
    candidates: int = 10
    """Rotations carried from the first stage into the second."""
    separation: int = 1
    """How far apart those candidates must be, in samples of the grid."""
    max_shift: float = 8.0
    """Largest shift considered, in voxels of the working volume."""
    epsilon: float = 1e-12
    """Guards the divisions where two particles overlap in nothing."""
    phases: bool = False
    """Whether the expansions kept the phases. If they did, the first
    stage is the constrained correlation itself and its best rotation is
    taken as it stands; the second stage then only finds the shift.

    The second stage used to choose among the candidates whatever the
    first had ranked them by. That is right for amplitudes, which it
    outranks. With phases it is the poorer judge: it turns the second
    particle by interpolating its spectrum, and on EMPIAR-10304 letting
    it choose took the pairs aligned from 40 in a hundred to 29."""

    def __post_init__(self) -> None:
        if self.bandwidth < 2:
            raise ValueError(
                f'A bandwidth of {self.bandwidth} is too small to resolve a '
                'rotation.'
            )
        if self.candidates < 1:
            raise ValueError('At least one candidate rotation is needed.')
        if not self.max_shift >= 0.0:
            raise ValueError('The largest shift is not negative.')


@dataclass(frozen=True)
class Alignment:
    """What aligning a pair found."""

    rotation: jax.Array
    """``(3, 3)``, taking the second particle onto the first."""
    shift: jax.Array
    """``(3,)`` in voxels, ordered ``(x, y, z)``."""
    score: jax.Array
    """The constrained correlation there, in ``[-1, 1]``."""


jax.tree_util.register_dataclass(
    Alignment, data_fields=['rotation', 'shift', 'score'], meta_fields=[]
)


@dataclass(frozen=True)
class Joint:
    """What searching rotations and shifts at once needs beyond a pair.

    A rotational search with phases only sees a pair that is already
    centred, to about ten angstroms on EMPIAR-10304. Two particles picked
    by hand or by a template are not. Looking for the rotation first and
    the shift after, in turns, does not get there either: with the shift
    wrong the rotational grid holds nothing, so the first rotation is
    arbitrary and so is the shift that follows from it. What does is
    searching both at once, which is what the five-dimensional method of
    Kovacs et al. (2003) is for, and `frmony.matching.frm5d` measures 22
    to 24 pairs in a hundred aligned on pairs forty angstroms apart,
    where rotations alone give 6 and the alternation 9 to 12.

    This is the same search reached another way. A shift of a pair is a
    shift of its first particle, ``a = t_i - A t_j`` being measured in
    its frame, so each trial shift is one more expansion of the first
    particle and one more rotational grid. It stays in Fourier space, at
    whatever bandwidth is wanted, and reads nothing off real-space
    spheres. It aligned as many of those pairs, 23 in a hundred, in a
    thousandth of the time.

    The lattice is searched at half the bandwidth, which is enough to
    say which shift a pair is at, and the rotation is then found at that
    shift at the full one.
    """

    coarse: Basis
    """The basis the lattice is searched on, at half the bandwidth."""
    fine: Basis
    """The basis of the expansions, which the rotation is found on."""
    lattice: jax.Array
    """Trial shifts ``(trials, 3)`` in voxels, ordered ``(x, y, z)``."""
    spacing: float
    """How far apart they are, and so how far from one the search over
    shifts has to look."""


jax.tree_util.register_dataclass(
    Joint,
    data_fields=['coarse', 'fine', 'lattice'],
    meta_fields=['spacing'],
)


def lattice_spacing(size: int, outer: float) -> float:
    """How far apart trial shifts may be, in voxels.

    Seven tenths of the shortest wave the search uses, ``size / outer``
    voxels. A shift missed by half of that turns the outermost sphere by
    about a radian, and the spheres inside it by less. Measured rather
    than derived: two thirds of this spacing aligned no more pairs.

    Args:
        size: Extent of the volume along each axis.
        outer: Radius of the outermost sphere searched, in voxels.
    """
    return 0.7 * size / outer


def shift_lattice(reach: float, spacing: float) -> NDArray[np.float64]:
    """Trial shifts filling a ball, on a body-centred cubic lattice.

    Body-centred because it covers a ball with the fewest points for a
    given largest miss.

    Args:
        reach: Radius of the ball, in voxels.
        spacing: Edge of the cubic cell, in voxels.

    Returns:
        ``(trials, 3)``, no shift first.
    """
    if not spacing > 0.0:
        raise ValueError(f'A spacing is positive, not {spacing}.')
    if reach <= 0.0:
        return np.zeros((1, 3))
    cells = int(np.ceil(reach / spacing)) + 1
    axis = np.arange(-cells, cells + 1, dtype=np.float64)
    corners = np.stack(
        np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1
    ).reshape(-1, 3)
    points = np.concatenate((corners, corners + 0.5)) * spacing
    distance = np.linalg.norm(points, axis=1)
    points = points[distance <= reach + 1e-9]
    return points[np.argsort(np.linalg.norm(points, axis=1), kind='stable')]


def truncate(coefficients: jax.Array, bandwidth: int) -> jax.Array:
    """Coefficients at a lower bandwidth, cut out of those at a higher.

    The degrees below the lower bandwidth are the same numbers at either,
    so nothing is recomputed. The orders are laid out as an FFT lays them
    out, and the column of the lower Nyquist order stays empty, as it
    does in an expansion made at that bandwidth.

    Args:
        coefficients: ``(radii, B, 2 * B)``.
        bandwidth: The lower bandwidth, no more than ``B``.

    Returns:
        ``(radii, bandwidth, 2 * bandwidth)``.
    """
    full = coefficients.shape[-2]
    if bandwidth > full:
        raise ValueError(
            f'Coefficients at bandwidth {full} hold none at {bandwidth}.'
        )
    if bandwidth == full:
        return coefficients
    kept = coefficients[..., :bandwidth, :]
    empty = jnp.zeros((*kept.shape[:-1], 1), dtype=kept.dtype)
    return jnp.concatenate(
        (
            kept[..., :bandwidth],
            empty,
            kept[..., 2 * full - bandwidth + 1 :],
        ),
        axis=-1,
    )


def _frequencies(basis: Basis, size: int) -> jax.Array:
    """Where a basis reads a spectrum, in cycles a box, ``(x, y, z)``."""
    return (basis.grid[::-1] - size // 2) / size


def _brought_back(
    samples: jax.Array, frequencies: jax.Array, shift: jax.Array, basis: Basis
) -> jax.Array:
    """The expansion of a particle moved back by a shift.

    A shift is a phase on a spectrum, and the phase is known exactly
    where the spheres are read, so the samples are turned and nothing is
    interpolated again.
    """
    phase = jnp.exp(
        2j
        * jnp.pi
        * (
            frequencies[0] * shift[0]
            + frequencies[1] * shift[1]
            + frequencies[2] * shift[2]
        )
    )
    return expand_shells(samples * phase, basis)


_CHUNK = 16
"""Trial shifts searched at once. Their grids are never all held."""


def _joint_peak(
    measured: jax.Array,
    left: Expansion,
    right: Expansion,
    joint: Joint,
    settings: Settings,
) -> jax.Array:
    """The trial shift at which the pair's best rotation is found."""
    size = measured.shape[-1]
    bandwidth = joint.coarse.bandwidth
    samples = sample_shells(measured, joint.coarse.grid)
    frequencies = _frequencies(joint.coarse, size)
    theirs = truncate(right.amplitude, bandwidth)
    denominator = jnp.sqrt(
        jnp.abs(
            correlation_grid(
                radial_overlap(
                    truncate(left.energy, bandwidth),
                    truncate(right.weight, bandwidth),
                ),
                bandwidth,
            )
        )
        * jnp.abs(
            correlation_grid(
                radial_overlap(
                    truncate(left.weight, bandwidth),
                    truncate(right.energy, bandwidth),
                ),
                bandwidth,
            )
        )
        + settings.epsilon
    )

    trials = joint.lattice.shape[0]
    padded = -(-trials // _CHUNK) * _CHUNK
    # Padded with the first trial, which changes no maximum.
    lattice = jnp.concatenate(
        (
            joint.lattice,
            jnp.broadcast_to(joint.lattice[:1], (padded - trials, 3)),
        )
    ).reshape(-1, _CHUNK, 3)
    numbers = jnp.arange(padded).reshape(-1, _CHUNK) % trials

    def one(shift: jax.Array) -> jax.Array:
        ours = _brought_back(samples, frequencies, shift, joint.coarse)
        return correlation_grid(radial_overlap(ours, theirs), bandwidth)

    def step(
        carry: tuple[jax.Array, jax.Array],
        chunk: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        best, which = carry
        shifts, number = chunk
        grids = jax.vmap(one)(shifts)
        top = jnp.max(grids, axis=0)
        owner = number[jnp.argmax(grids, axis=0)]
        better = top > best
        return (
            jnp.where(better, top, best),
            jnp.where(better, owner, which),
        ), None

    shape = (2 * bandwidth,) * 3
    start = (
        jnp.full(shape, -jnp.inf, dtype=jnp.float32),
        jnp.zeros(shape, dtype=numbers.dtype),
    )
    (best, which), _ = jax.lax.scan(step, start, (lattice, numbers))
    score = best / denominator
    searchable = score.at[:, bandwidth + 1 :, :].set(-jnp.inf)
    index = jnp.stack(jnp.unravel_index(jnp.argmax(searchable), shape))
    return joint.lattice[which[index[0], index[1], index[2]]]


def candidate_rotations(
    left: Expansion, right: Expansion, settings: Settings
) -> tuple[jax.Array, jax.Array]:
    """The best few rotations of the first stage, interpolated.

    Returns:
        Matrices ``(candidates, 3, 3)`` and the score of each.
    """
    grid = rotation_score(left, right, settings.bandwidth, settings.epsilon)

    # The grid covers every rotation twice: (rot, tilt, psi) and
    # (rot + pi, -tilt, psi + pi) are the same turn. Searching the whole of
    # it returns each candidate alongside its mirror and so halves how many
    # distinct rotations are actually tried. Half of the second axis covers
    # the group exactly once.
    half = grid.shape[1] // 2
    searchable = grid.at[:, half + 1 :, :].set(-jnp.inf)
    indices, _ = top_peaks(searchable, settings.candidates, settings.separation)

    def polish(index: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Every axis of the grid wraps: they are angles.
        position, height = refine_peak(grid, index, (True, True, True))
        return position, height

    positions, heights = jax.vmap(polish)(indices)
    step = jnp.pi / settings.bandwidth
    angles = -step * positions
    matrices = _euler_to_matrix(angles[:, 2], angles[:, 1], angles[:, 0])
    return matrices, heights


def _euler_to_matrix(
    rot: jax.Array, tilt: jax.Array, psi: jax.Array
) -> jax.Array:
    """`frmony.geometry.euler_to_matrix`, in JAX and under a trace."""
    ca, sa = jnp.cos(rot), jnp.sin(rot)
    cb, sb = jnp.cos(tilt), jnp.sin(tilt)
    cg, sg = jnp.cos(psi), jnp.sin(psi)
    return jnp.stack(
        [
            jnp.stack(
                [cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb], -1
            ),
            jnp.stack(
                [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb], -1
            ),
            jnp.stack([sb * ca, sb * sa, cb], -1),
        ],
        axis=-2,
    )


def _shift_mask(size: int, radius: float) -> jax.Array:
    """Shifts no further from nothing than ``radius``, as a mask."""
    axis = jnp.arange(size) - size // 2
    squared = (
        axis[:, None, None] ** 2
        + axis[None, :, None] ** 2
        + axis[None, None, :] ** 2
    )
    return squared <= radius * radius


def align_pair(
    left_data: ArrayLike,
    left_weight: ArrayLike,
    right_data: ArrayLike,
    right_weight: ArrayLike,
    left: Expansion,
    right: Expansion,
    settings: Settings,
    band: ArrayLike | None = None,
    joint: Joint | None = None,
) -> Alignment:
    """Align the second particle onto the first.

    Args:
        left_data: Centred spectrum of the first particle.
        left_weight: Its Fourier weight.
        right_data: Centred spectrum of the second.
        right_weight: Its weight.
        left: The expansion of the first, from `expand_particle`.
        right: The expansion of the second.
        settings: What to search.
        band: One over the frequencies the rotations were searched on,
            the shape of a spectrum, and nothing elsewhere. The shifts
            are then searched over the same frequencies. Left out, they
            are searched over every frequency, the lowest among them,
            which carry a particle's mass and not its orientation: the
            rotational search has always dropped those, and a shift
            stage that keeps them is led by them. On EMPIAR-10304 it
            turned 49 pairs in a hundred aligned by the first stage
            into 22.

        joint: What a search of rotations and shifts at once needs. Left
            out, the rotations are searched with the pair as it stands,
            which only finds them if it is centred, and the shifts
            within `Settings.max_shift` of nothing.

    Returns:
        The best rotation, shift and score found.
    """
    left_data = to_device(left_data)
    size = left_data.shape[-1]
    grid = to_device(frequency_grid(size))
    centre = jnp.zeros(3, dtype=jnp.float32)
    reach = settings.max_shift

    if joint is not None:
        measured = left_data * to_device(left_weight)
        centre = _joint_peak(measured, left, right, joint, settings)
        # The rotation, at the full bandwidth, with the first particle
        # brought back by the shift the lattice found.
        left = Expansion(
            amplitude=_brought_back(
                sample_shells(measured, joint.fine.grid),
                _frequencies(joint.fine, size),
                centre,
                joint.fine,
            ),
            energy=left.energy,
            weight=left.weight,
        )
        reach = joint.spacing

    offsets = jnp.arange(size, dtype=jnp.float32) - size // 2
    mask = (
        (offsets[:, None, None] - centre[2]) ** 2
        + (offsets[None, :, None] - centre[1]) ** 2
        + (offsets[None, None, :] - centre[0]) ** 2
    ) <= reach * reach

    matrices, _ = candidate_rotations(left, right, settings)
    if settings.phases:
        matrices = matrices[:1]
    if band is not None:
        left_weight = to_device(left_weight) * to_device(band)
        right_weight = to_device(right_weight) * to_device(band)

    def score_one(rotation: jax.Array) -> tuple[jax.Array, jax.Array]:
        values, _ = shift_correlation(
            left_data,
            left_weight,
            right_data,
            right_weight,
            rotation,
            grid,
            settings.epsilon,
        )
        allowed = jnp.where(mask, values, -jnp.inf)
        flat = jnp.argmax(allowed)
        index = jnp.stack(jnp.unravel_index(flat, allowed.shape))
        # The shifts are masked to a ball, so their axes do not wrap.
        position, height = refine_peak(values, index, (False, False, False))
        return position, height

    positions, heights = jax.vmap(score_one)(matrices)
    best = jnp.argmax(heights)
    centred = positions[best] - size // 2
    return Alignment(
        rotation=matrices[best],
        shift=centred[::-1],
        score=heights[best],
    )


def align_pairs(
    data: ArrayLike,
    weight: ArrayLike,
    expansions: Expansion,
    left: ArrayLike,
    right: ArrayLike,
    settings: Settings,
    batch: int = 8,
    band: ArrayLike | None = None,
    joint: Joint | None = None,
) -> Alignment:
    """Align many pairs, a batch at a time.

    Args:
        data: Centred spectra of every particle, ``(particles, n, n, n)``.
        weight: Their Fourier weights, the same shape.
        expansions: Their expansions, batched over the leading axis.
        left: Index of the first particle of each pair.
        right: Index of the second.
        settings: What to search.
        batch: How many pairs to align at once. Larger is faster until the
            device runs out of room; the cost is one correlation grid and
            one working volume per pair in flight.
        band: The frequencies the shifts are searched over; see
            `align_pair`.
        joint: What a search of rotations and shifts at once needs; see
            `Joint`.

    Returns:
        An `Alignment` whose fields are batched over the pairs.
    """
    volumes = to_device(data)
    weights = to_device(weight)
    # The leaves are indexed by a traced index below, so they have to
    # be on the device already; a NumPy leaf would be asked to turn a
    # tracer into an array.
    batched = jax.tree.map(to_device, expansions)
    if band is not None:
        band = to_device(band)
    pairs = np.stack(
        (np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32)),
        axis=1,
    )
    count = len(pairs)
    batch = max(batch, 1)

    # The pairs go to the device a chunk at a time, so that a run can say
    # how far it has got. A chunk is a hundredth of them, a whole number
    # of batches and never fewer than sixty-four, which is large enough
    # that handing them over costs nothing measurable. Every chunk is
    # the same size, the last padded with its own last pair, so the
    # program is compiled once.
    chunk = max(-(-count // 100), 64 * batch)
    chunk = min(-(-chunk // batch) * batch, -(-count // batch) * batch)
    chunks = -(-count // chunk)
    padded = np.concatenate(
        (pairs, np.repeat(pairs[-1:], chunks * chunk - count, axis=0))
    ).reshape(chunks, chunk, 2)

    pieces = []
    for indices in progress(
        padded, 'Aligning pairs', total=chunks, unit='pair', scale=chunk
    ):
        piece = _align_chunk(
            volumes,
            weights,
            batched,
            to_device(indices),
            settings,
            band,
            joint,
            batch,
        )
        # Waited on, or the bar would only say how much was dispatched.
        piece.score.block_until_ready()
        pieces.append(piece)
    aligned: Alignment = jax.tree.map(
        lambda *leaves: jnp.concatenate(leaves)[:count], *pieces
    )
    return aligned


@functools.partial(jax.jit, static_argnames=('settings', 'batch'))
def _align_chunk(
    volumes: jax.Array,
    weights: jax.Array,
    expansions: Expansion,
    pairs: jax.Array,
    settings: Settings,
    band: jax.Array | None,
    joint: Joint | None,
    batch: int,
) -> Alignment:
    """Align a chunk of pairs, ``(pairs, 2)`` indices into the particles.

    Everything is an argument, and nothing is closed over. Closed over,
    the particles would be folded into the program as constants.
    """

    def one(indices: jax.Array) -> Alignment:
        first, second = indices[0], indices[1]
        return align_pair(
            volumes[first],
            weights[first],
            volumes[second],
            weights[second],
            jax.tree.map(lambda leaf: leaf[first], expansions),
            jax.tree.map(lambda leaf: leaf[second], expansions),
            settings,
            band,
            joint,
        )

    aligned: Alignment = jax.lax.map(one, pairs, batch_size=batch)
    return aligned
