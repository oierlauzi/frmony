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
thing, or a correlation that peaked in the wrong place. They are dropped by
their residual, measured modulo the group, which matters: with C4 symmetry
a residual of ninety degrees is a perfect measurement, and a rule that did
not know about the symmetry would discard exactly the pairs that are right.
"""

from __future__ import annotations

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

__all__ = ['Synchronized', 'propagate_along_tree', 'synchronize_rotations']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Synchronized:
    """Orientations, and what the measurements made of them."""

    orientations: NDArray[np.float64]
    """``(count, 3, 3)``, each taking a particle onto the structure."""
    residuals: NDArray[np.float64]
    """How far each pair ended up from agreeing [rad], modulo symmetry."""
    kept: NDArray[np.bool_]
    """Which pairs survived the rejection, ``(pairs,)``."""
    iterations: int
    """Steps the last synchronization took."""


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


def synchronize_rotations(
    count: int,
    left: ArrayLike,
    right: ArrayLike,
    measurements: ArrayLike,
    weights: ArrayLike,
    symmetry: str = 'C1',
    rounds: int = 4,
    reject_above: float | None = np.radians(90.0),
    max_iterations: int = 512,
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
        rounds: How many times to snap, synchronize and reject.
        reject_above: Drop pairs whose residual exceeds this [rad]. None
            keeps every pair. The residual is measured modulo the symmetry,
            so the threshold means what it says whatever the group is.
        max_iterations: Steps allowed in each synchronization.

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

        snapped = snap_to_symmetry(
            measurements[kept],
            left[kept],
            right[kept],
            orientations,
            group,
        )
        orientations, iterations = synchronize(
            count,
            left[kept],
            right[kept],
            snapped,
            weights[kept],
            start=orientations,
            max_iterations=max_iterations,
        )
        # The synchronization moved the global frame, so the group has
        # to be found again before the residuals are judged against it. A
        # stale group reads honest pairs as outliers.
        group = _frame_group(left, right, measurements, orientations, order)
        residuals = residual_angles(
            measurements, left, right, orientations, group
        )
        if reject_above is None:
            continue

        surviving = kept & (residuals <= reject_above)
        if not surviving.any():
            _LOGGER.warning(
                'Every pair would be rejected at %.0f degrees; keeping them '
                'all instead.',
                np.degrees(reject_above),
            )
            break
        dropped = int(kept.sum() - surviving.sum())
        kept = surviving
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

    residuals = residual_angles(measurements, left, right, orientations, group)
    return Synchronized(
        orientations=orientations,
        residuals=residuals,
        kept=kept,
        iterations=iterations,
    )


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
