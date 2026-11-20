"""Aligning one subtomogram to another, in two stages.

The rotation is found first, on amplitudes, which do not move when the
particle does. The best few rotations are kept rather than the best one,
because amplitudes discard the phases and so discriminate less than the
full score: the right rotation is reliably among the best few even when it
is not the best one. Then each candidate gets a single transform, which
gives the correlation at every shift at once and scores the candidate in
full, phases included.

What comes out is a rotation and a shift satisfying the relation
`syncsta.geometry` is written in,

    V_i(x) = V_j(A.T @ (x - a)),

which is what the synchronization expects of a pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .correlation import Expansion, rotation_score
from .peaks import refine_peak, top_peaks
from .translation import frequency_grid, shift_correlation

__all__ = [
    'Alignment',
    'Settings',
    'align_pair',
    'align_pairs',
    'candidate_rotations',
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
    """`syncsta.geometry.euler_to_matrix`, in JAX and under a trace."""
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

    Returns:
        The best rotation, shift and score found.
    """
    left_data = jnp.asarray(left_data)
    size = left_data.shape[-1]
    grid = jnp.asarray(frequency_grid(size))
    mask = _shift_mask(size, settings.max_shift)

    matrices, _ = candidate_rotations(left, right, settings)

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

    Returns:
        An `Alignment` whose fields are batched over the pairs.
    """
    volumes = jnp.asarray(data)
    weights = jnp.asarray(weight)
    # The leaves are indexed by a traced index below, so they have to
    # be on the device already; a NumPy leaf would be asked to turn a
    # tracer into an array.
    batched = jax.tree.map(jnp.asarray, expansions)
    first_index = jnp.asarray(left)
    second_index = jnp.asarray(right)

    def one(indices: jax.Array) -> Alignment:
        first, second = indices[0], indices[1]
        return align_pair(
            volumes[first],
            weights[first],
            volumes[second],
            weights[second],
            jax.tree.map(lambda leaf: leaf[first], batched),
            jax.tree.map(lambda leaf: leaf[second], batched),
            settings,
        )

    pairs = jnp.stack((first_index, second_index), axis=1)
    aligned: Alignment = jax.lax.map(one, pairs, batch_size=max(batch, 1))
    return aligned
