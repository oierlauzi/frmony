"""Polishing an alignment off the grid.

The two stages of `frmony.matching.pairwise` leave a rotation and a shift
on a grid, improved by fitting a parabola through their neighbours. That is
good to a fraction of the step and no better, because a parabola is not the
score. This refines them against the score itself, by gradient.

It is off by default, and for a reason worth stating plainly. Measured on
simulated subtomograms, with no missing wedge it does what a refinement
should: the median error of a pair fell from 2.3 degrees to 1.0, and the
worst from 5.4 to 1.8. With a sixty degree tilt range the median went the
other way, from 4.8 degrees to 6.4, while the worst still improved, from
17.6 to 14.1.

Nothing is wrong with the optimizer: the score rises every time, which a
test checks. What the second row says is that when a wide wedge is missing,
the peak of the score is not quite at the true orientation, so climbing
harder walks away from the truth. That is a property of the objective and
not of the method used to maximize it, and it is why this is something to
turn on deliberately rather than a better default. Where the pairs are many
and a synchronization is going to average over them, the grid is enough and
the bias is not worth buying.

Rotations are parameterized by the exponential map about the candidate --
``R = R_candidate @ exp([d])`` with ``d`` starting at zero -- rather than by
Euler angles. Euler angles have coordinate singularities at a tilt of zero
and of half a turn, where the parameterization stops being able to express
some directions of movement. The published work closest to this handles
that by arguing the iterates stay away from them; the exponential map does
not have them, and nothing is paid for the difference because JAX
differentiates whatever it is given.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.optimize
from jax.typing import ArrayLike

from .translation import frequency_grid, rotate_spectrum

__all__ = ['exponential_map', 'refine_alignment', 'score_at']


def exponential_map(vector: ArrayLike) -> jax.Array:
    """The rotation of a rotation vector, by Rodrigues' formula.

    Written so that it, and its derivative, stay finite at zero, which is
    where every refinement starts.
    """
    vector = jnp.asarray(vector)
    # The norm itself is what has to be avoided, not just the division by
    # it: the derivative of ||v|| is v / ||v||, which is zero over zero at
    # the origin, and a NaN there spreads through a `where` from the
    # branch that was not taken. Every refinement starts at the origin, so
    # this is not an edge case but the first step of every call. Working
    # from the squared norm, and substituting a harmless value inside the
    # branch not taken, keeps the derivative finite.
    squared = jnp.sum(vector * vector)
    small = squared < 1e-12
    safe = jnp.where(small, 1.0, squared)
    angle = jnp.sqrt(safe)
    sinc = jnp.where(small, 1.0 - squared / 6.0, jnp.sin(angle) / angle)
    versine = jnp.where(
        small, 0.5 - squared / 24.0, (1.0 - jnp.cos(angle)) / safe
    )

    x, y, z = vector[0], vector[1], vector[2]
    zero = jnp.zeros((), dtype=vector.dtype)
    cross = jnp.stack(
        [
            jnp.stack([zero, -z, y]),
            jnp.stack([z, zero, -x]),
            jnp.stack([-y, x, zero]),
        ]
    )
    return (
        jnp.eye(3, dtype=vector.dtype)
        + sinc * cross
        + versine * (cross @ cross)
    )


def score_at(
    left_data: jax.Array,
    left_weight: jax.Array,
    right_data: jax.Array,
    right_weight: jax.Array,
    rotation: jax.Array,
    shift: jax.Array,
    grid: jax.Array,
    epsilon: float = 1e-12,
) -> jax.Array:
    """The constrained score at one rotation and one shift.

    The same quantity `frmony.matching.translation` evaluates at every
    shift at once, written for a single one so that it can be
    differentiated with respect to both.
    """
    size = left_data.shape[-1]
    turned = rotate_spectrum(right_data, rotation, grid)
    turned_weight = jnp.maximum(
        rotate_spectrum(right_weight, rotation, grid), 0.0
    )
    overlap = left_weight * turned_weight

    axis = (jnp.arange(size) - size // 2) / size
    phase = (
        axis[:, None, None] * shift[2]
        + axis[None, :, None] * shift[1]
        + axis[None, None, :] * shift[0]
    )
    ramp = jnp.exp(2j * jnp.pi * phase)

    numerator = jnp.sum(
        overlap * (left_data * jnp.conjugate(turned) * ramp)
    ).real
    first = jnp.sum(overlap * jnp.abs(left_data) ** 2)
    second = jnp.sum(overlap * jnp.abs(turned) ** 2)
    return numerator / jnp.sqrt(jnp.abs(first) * jnp.abs(second) + epsilon)


def refine_alignment(
    left_data: ArrayLike,
    left_weight: ArrayLike,
    right_data: ArrayLike,
    right_weight: ArrayLike,
    rotation: ArrayLike,
    shift: ArrayLike,
    max_iterations: int = 40,
    epsilon: float = 1e-12,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Improve a rotation and a shift against the score itself.

    Args:
        left_data: Centred spectrum of the first particle.
        left_weight: Its Fourier weight.
        right_data: Centred spectrum of the second.
        right_weight: Its weight.
        rotation: The rotation to start from, ``(3, 3)``.
        shift: The shift to start from, ``(3,)``, ordered ``(x, y, z)``.
        max_iterations: How many steps the optimizer may take.
        epsilon: Guards the division where two particles overlap in
            nothing.

    Returns:
        The refined rotation, shift and score. If the optimizer fails to
        improve on what it was given, what it was given comes back, so
        this can never make an alignment worse.
    """
    refined: tuple[jax.Array, jax.Array, jax.Array] = _refine(
        jnp.asarray(left_data),
        jnp.asarray(left_weight),
        jnp.asarray(right_data),
        jnp.asarray(right_weight),
        jnp.asarray(rotation, dtype=jnp.float32),
        jnp.asarray(shift, dtype=jnp.float32),
        jnp.asarray(frequency_grid(jnp.asarray(left_data).shape[-1])),
        max_iterations,
        epsilon,
    )
    return refined


@partial(jax.jit, static_argnums=(7, 8))
def _refine(
    left_data: jax.Array,
    left_weight: jax.Array,
    right_data: jax.Array,
    right_weight: jax.Array,
    start_rotation: jax.Array,
    start_shift: jax.Array,
    grid: jax.Array,
    max_iterations: int,
    epsilon: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`refine_alignment`, compiled once per shape."""

    def negative(parameters: jax.Array) -> jax.Array:
        turn = start_rotation @ exponential_map(parameters[:3])
        return -score_at(
            left_data,
            left_weight,
            right_data,
            right_weight,
            turn,
            start_shift + parameters[3:],
            grid,
            epsilon,
        )

    begin = jnp.zeros(6, dtype=jnp.float32)
    before = -negative(begin)
    found = jax.scipy.optimize.minimize(
        negative, begin, method='BFGS', options={'maxiter': max_iterations}
    )
    after = -found.fun

    # Never hand back something worse than what came in.
    better = after > before
    parameters = jnp.where(better, found.x, begin)
    return (
        start_rotation @ exponential_map(parameters[:3]),
        start_shift + parameters[3:],
        jnp.where(better, after, before),
    )
