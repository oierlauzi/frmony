"""Orientations from pairwise measurements, symmetry and outliers included.

`frmony.synchronization.power_method` solves the clean problem: given
measurements that are all honest attempts at ``Q_i @ Q_j.T``, find the
orientations. Two things stop real measurements from being that.

A particle with point group ``G`` has an orientation defined only up to
``G``, so an alignment measures some member of a coset rather than the
relative orientation itself. `frmony.symmetry` explains why that cannot be
fixed one pair at a time, and gives the step that fixes it once an estimate
exists. Here the two are alternated: snap every measurement onto the coset
member nearest the current estimate, synchronize, repeat. With a trivial
group the snap does nothing, so there is one path and not two.

And some pairs are simply wrong -- two particles that were never the same
thing, or a correlation that peaked in the wrong place. On real
subtomograms most are: of the pairs of EMPIAR-10304, between four and
eight in ten. Each pair is weighed by the probability that it is right,
read off its residual, measured modulo the group, which matters: with C4
symmetry a residual of ninety degrees is a perfect measurement, and a rule
that did not know about the symmetry would discard exactly the pairs that
are right.

That probability comes from a mixture fitted to the residuals themselves
(`inlier_probabilities`). A right pair is off by a small error of some
width; a wrong one is off by as much as a rotation drawn at random would
be, which the group decides and nothing else does. The width and the share
of right pairs are fitted, so there is no threshold to choose. Dropping
pairs above a residual of ninety degrees, which is what this used to do,
kept two thirds of the pairs of EMPIAR-10304 and only six in ten of those
were right; 88 particles in a hundred came out within fifteen degrees of a
refinement. Weighed by the mixture, 97 did, where knowing which pairs were
right would give 99.7.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..graph import spanning_tree
from ..symmetry import (
    estimate_group,
    residual_angles,
    snap_to_symmetry,
    symmetry_order,
)
from .power_method import synchronize

__all__ = [
    'Synchronized',
    'inlier_probabilities',
    'outlier_density',
    'propagate_along_tree',
    'synchronize_rotations',
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Synchronized:
    """Orientations, and what the measurements made of them."""

    orientations: NDArray[np.float64]
    """``(count, 3, 3)``, each taking a particle onto the structure."""
    residuals: NDArray[np.float64]
    """How far each pair ended up from agreeing [rad], modulo symmetry."""
    kept: NDArray[np.bool_]
    """Which pairs are more likely right than not, ``(pairs,)``."""
    iterations: int
    """Steps the last synchronization took."""
    inliers: NDArray[np.float64] | None = None
    """The probability that each pair is right, ``(pairs,)``, if it was
    fitted."""


def propagate_along_tree(
    count: int,
    left: ArrayLike,
    right: ArrayLike,
    measurements: ArrayLike,
    weights: ArrayLike,
) -> NDArray[np.float64]:
    """Orientations that satisfy a spanning tree of the pairs exactly.

    A tree has no cycles, so every one of its edges can be satisfied at
    once by walking out from a root: there is nothing for the measurements
    to disagree about. That makes it a good start in general, and the right
    one when the particle has symmetry, where a cycle is exactly where the
    ambiguity of each measurement first shows up.

    The heaviest edges are chosen, so the orientations are propagated along
    the pairs that were aligned most confidently.

    Args:
        count: How many particles there are.
        left: Index of the first particle of each pair.
        right: Index of the second.
        measurements: Relative orientations ``(pairs, 3, 3)``.
        weights: How much each is trusted, ``(pairs,)``.

    Returns:
        Orientations ``(count, 3, 3)``. A particle in no pair, or in a
        piece of the graph of its own, gets the identity or its piece's
        own arbitrary frame.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    measurements = np.asarray(measurements, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    tree = spanning_tree(count, left, right, weights)
    neighbours: dict[int, list[tuple[int, int, bool]]] = {}
    for edge in tree:
        first, second = int(left[edge]), int(right[edge])
        neighbours.setdefault(first, []).append((second, int(edge), False))
        neighbours.setdefault(second, []).append((first, int(edge), True))

    orientations = np.tile(np.eye(3), (count, 1, 1))
    seen = np.zeros(count, dtype=bool)
    for root in range(count):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            current = stack.pop()
            for other, edge, backwards in neighbours.get(current, ()):
                if seen[other]:
                    continue
                # A_ij is about Q_i @ Q_j.T, so Q_j = A_ij.T @ Q_i going
                # one way and Q_i = A_ij @ Q_j going the other.
                block = measurements[edge]
                orientations[other] = (
                    block @ orientations[current]
                    if backwards
                    else block.T @ orientations[current]
                )
                seen[other] = True
                stack.append(other)
    return orientations


_ANGLES = np.linspace(0.0, np.pi, 1025)
"""Where the densities of a residual are tabulated [rad]."""


@functools.lru_cache(maxsize=32)
def _outlier_table(group_key: bytes, order: int) -> NDArray[np.float64]:
    if order == 1:
        density = (1.0 - np.cos(_ANGLES)) / np.pi
        return density
    group = np.frombuffer(group_key).reshape(-1, 3, 3)
    from ..geometry import random_rotations

    rotations = random_rotations(np.random.default_rng(0), 200_000)
    traces = np.einsum('gij,nij->ng', group, rotations)
    angles = np.arccos(np.clip((traces.max(axis=1) - 1.0) / 2.0, -1.0, 1.0))
    counts, edges = np.histogram(angles, bins=256, range=(0.0, np.pi))
    centres = 0.5 * (edges[1:] + edges[:-1])
    density = np.interp(
        _ANGLES, centres, counts / (len(angles) * (edges[1] - edges[0]))
    )
    return np.asarray(density / np.trapezoid(density, _ANGLES))


def outlier_density(
    residuals: ArrayLike, group: ArrayLike | None = None
) -> NDArray[np.float64]:
    """How likely a residual is for a pair that measured nothing.

    A wrong pair is off by as much as a rotation drawn at random. Measured
    modulo a group, that is the angle to the nearest of its elements,
    which is tabulated once per group by drawing rotations. With no group
    it is ``(1 - cos r) / pi``.

    Args:
        residuals: Angles [rad].
        group: The elements of the group, ``(order, 3, 3)``, in any frame:
            the angle to the nearest of them does not depend on it.

    Returns:
        The density at each residual, per radian.
    """
    elements = np.eye(3)[None] if group is None else np.asarray(group)
    table = _outlier_table(
        np.ascontiguousarray(elements, dtype=np.float64).tobytes(),
        len(elements),
    )
    return np.interp(np.asarray(residuals, dtype=np.float64), _ANGLES, table)


def _outlier_shape(
    residuals: NDArray[np.float64], group: ArrayLike | None
) -> NDArray[np.float64]:
    """`outlier_density` over ``r**2``, which is finite at nought.

    Both densities vanish as ``r**2`` at a residual of nought, so they are
    compared with that factor taken out. Comparing them as they are made
    a perfect measurement, whose residual is exactly nought, into nought
    over nought, and called it wrong.
    """
    elements = np.eye(3)[None] if group is None else np.asarray(group)
    table = _outlier_table(
        np.ascontiguousarray(elements, dtype=np.float64).tobytes(),
        len(elements),
    )
    shape = np.empty_like(table)
    shape[1:] = table[1:] / _ANGLES[1:] ** 2
    # (1 - cos r) / (pi r**2) tends to 1 / 2 pi; a tabulated density is
    # carried to nought from its first few points, where it is flat.
    shape[0] = shape[1:6].mean() if len(elements) > 1 else 1.0 / (2.0 * np.pi)
    if len(elements) == 1:
        small = residuals < 1e-4
        exact = (1.0 - np.cos(residuals)) / (
            np.pi * np.maximum(residuals, 1e-12) ** 2
        )
        return np.where(small, 1.0 / (2.0 * np.pi), exact)
    return np.interp(residuals, _ANGLES, shape)


def inlier_probabilities(
    residuals: ArrayLike,
    group: ArrayLike | None = None,
    iterations: int = 200,
) -> tuple[NDArray[np.float64], float, float]:
    """The probability that each pair is right, from its residual alone.

    A right pair is off by a small error. Taking it as isotropic and
    Gaussian, of a width ``s``, the angle it is off by has the density
    ``r**2 exp(-r**2 / 2 s**2)``. A wrong pair is off as `outlier_density`
    says. The width and the share of right pairs are fitted to the
    residuals by expectation and maximization, so neither is chosen.

    Args:
        residuals: Angles [rad], modulo the group.
        group: The group the residuals are measured modulo.
        iterations: Steps of the fit.

    Returns:
        The probability for each pair, the width [rad] and the share of
        right pairs.
    """
    r = np.asarray(residuals, dtype=np.float64)
    wrong = _outlier_shape(r, group)
    width = max(float(np.median(r)) / 2.0, np.radians(1.0))
    share = 0.5
    posterior = np.full(len(r), 0.5)
    for _ in range(iterations):
        # The density of a right pair, over r**2, normalized on [0, pi].
        bell = _ANGLES**2 * np.exp(-(_ANGLES**2) / (2.0 * width * width))
        right = np.exp(-(r**2) / (2.0 * width * width)) / max(
            float(np.trapezoid(bell, _ANGLES)), 1e-300
        )
        mine = share * right
        posterior = mine / np.maximum(mine + (1.0 - share) * wrong, 1e-300)
        updated = float(np.clip(posterior.mean(), 1e-4, 1.0 - 1e-4))
        # Three degrees of freedom: the mean of r**2 is 3 s**2. The floor
        # is what single precision resolves of an angle near nought.
        width = max(
            float(
                np.sqrt(
                    (posterior * r * r).sum()
                    / (3.0 * max(posterior.sum(), 1e-12))
                )
            ),
            np.radians(0.05),
        )
        if abs(updated - share) < 1e-7:
            share = updated
            break
        share = updated
    return posterior, width, share


def synchronize_rotations(
    count: int,
    left: ArrayLike,
    right: ArrayLike,
    measurements: ArrayLike,
    weights: ArrayLike,
    symmetry: str = 'C1',
    rounds: int = 4,
    reject_above: float | None = None,
    max_iterations: int = 512,
    robust: bool = True,
    improper: bool = False,
) -> Synchronized:
    """Find the orientations the measurements agree on.

    Args:
        count: How many particles there are.
        left: Index of the first particle of each pair.
        right: Index of the second.
        measurements: Relative orientations ``(pairs, 3, 3)``.
        weights: How much each is trusted, ``(pairs,)``, non-negative.
        symmetry: The point group of the particle, as `point_group` names
            them. ``'C1'`` for none.
        rounds: How many times to snap, synchronize and weigh.
        reject_above: Also drop pairs whose residual exceeds this [rad].
            None keeps every pair to be weighed. The residual is measured
            modulo the symmetry, so the threshold means what it says
            whatever the group is.
        max_iterations: Steps allowed in each synchronization.
        robust: Weigh each pair by the probability that it is right; see
            `inlier_probabilities`. Without it every pair kept counts by
            its weight alone.
        improper: Solve over the orthogonal group, so that a particle can
            come out as a mirror image of the rest. The measurements may
            then be mirrors too.

    Returns:
        A `Synchronized`.

    Raises:
        ValueError: If there are no pairs, or fewer than two particles.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    measurements = np.asarray(measurements, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if count < 2:
        raise ValueError(
            f'Synchronizing needs at least two particles, not {count}.'
        )
    if len(left) == 0:
        raise ValueError('Synchronizing needs at least one pair.')

    order = symmetry_order(symmetry)
    kept = np.ones(len(left), dtype=bool)
    inliers = np.ones(len(left))
    orientations = propagate_along_tree(
        count, left, right, measurements, weights
    )
    iterations = 0

    group = np.eye(3)[None]
    for attempt in range(max(rounds, 1)):
        # The group to snap against is not the named one, and it has to be
        # found again every round. A synchronization pins the orientations
        # down only up to one global rotation, and a point group acts in
        # the frame of the particle, so the group present in the estimated
        # frame is the named one conjugated by that rotation -- the same
        # group with its axes somewhere else. Worse, the rotation changes
        # from round to round, so a group read off once goes stale and
        # starts rejecting the pairs that are right. See `frmony.symmetry`.
        group = _frame_group(left, right, measurements, orientations, order)
        if len(group) < order and attempt == 0:
            _LOGGER.warning(
                'Only %d of the %d elements of %s turned up in the '
                'residuals; the symmetry may be weaker than asked for, or '
                'the pairs too few.',
                len(group),
                order,
                symmetry,
            )

        using = kept & (inliers > 1e-6)
        snapped = snap_to_symmetry(
            measurements[using],
            left[using],
            right[using],
            orientations,
            group,
        )
        orientations, iterations = synchronize(
            count,
            left[using],
            right[using],
            snapped,
            weights[using] * inliers[using],
            start=orientations,
            max_iterations=max_iterations,
            improper=improper,
        )
        # The synchronization moved the global frame, so the group has
        # to be found again before the residuals are judged against it. A
        # stale group reads honest pairs as outliers.
        group = _frame_group(left, right, measurements, orientations, order)
        residuals = residual_angles(
            measurements, left, right, orientations, group
        )
        before = kept.copy()
        if reject_above is not None:
            surviving = kept & (residuals <= reject_above)
            if not surviving.any():
                _LOGGER.warning(
                    'Every pair would be rejected at %.0f degrees; keeping '
                    'them all instead.',
                    np.degrees(reject_above),
                )
            else:
                kept = surviving
        if robust:
            previous = inliers
            inliers, width, share = inlier_probabilities(
                residuals[kept], group if order > 1 else None
            )
            inliers = _scatter(inliers, kept)
            _LOGGER.info(
                'Round %d: %.0f%% of the pairs look right, off by %.1f '
                'degrees; median residual %.1f degrees.',
                attempt + 1,
                100.0 * share,
                float(np.degrees(width) * np.sqrt(3.0)),
                float(np.degrees(np.median(residuals[kept]))),
            )
            if np.max(np.abs(inliers - previous)) < 1e-3 and np.array_equal(
                kept, before
            ):
                break
        else:
            dropped = int(before.sum() - kept.sum())
            _LOGGER.info(
                'Round %d: %d pairs kept, %d dropped, median residual %.1f '
                'degrees.',
                attempt + 1,
                int(kept.sum()),
                dropped,
                float(np.degrees(np.median(residuals[kept]))),
            )
            if dropped == 0:
                break

    if improper and np.sum(np.linalg.det(orientations) < 0.0) > count / 2:
        # The orientations are found up to one element of the orthogonal
        # group, a mirror among them: inverting every one of them changes
        # nothing that was measured. Of the two hands, the one most of the
        # particles are in is the one called proper.
        orientations = -orientations
    residuals = residual_angles(measurements, left, right, orientations, group)
    return Synchronized(
        orientations=orientations,
        residuals=residuals,
        kept=kept & (inliers > 0.5) if robust else kept,
        iterations=iterations,
        inliers=inliers if robust else None,
    )


def _scatter(
    values: NDArray[np.float64], where: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Values for the pairs where ``where`` holds, and nothing elsewhere."""
    out = np.zeros(len(where))
    out[where] = values
    return out


def _frame_group(
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    measurements: NDArray[np.float64],
    orientations: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """The point group as it sits in the frame the orientations are in."""
    if order == 1:
        return np.eye(3)[None]
    seen = (
        np.swapaxes(orientations[left], -2, -1)
        @ measurements
        @ orientations[right]
    )
    return estimate_group(seen, order)
