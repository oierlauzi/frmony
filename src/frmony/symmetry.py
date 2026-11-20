"""Point groups, and what they do to a set of orientations.

A particle with point group ``G`` has an orientation that is only defined up
to ``G``: ``Q`` and ``Q @ g`` describe the same density for every ``g`` in
``G``. So a pairwise alignment does not measure ``Q_i @ Q_j.T``. It measures
some member of the coset

    {Q_i @ g @ Q_j.T : g in G},

whichever peak of the correlation happened to win. Handed to the
synchronization unchanged, those measurements disagree with one another and
it averages over the orbit instead of settling on it.

What makes this harder than ordinary outlier rejection is that the ambiguity
is not a fixed group acting on the measurements. Around a triangle,

    A_ij @ A_jk @ A_ki = Q_i @ (g1 @ g2 @ g3) @ Q_i.T,

so the residual is a group element conjugated into a frame nobody knows yet.
The ambiguity at an edge depends on the very orientations being solved for.

Two things follow, and this module provides both. `snap_to_symmetry` picks,
for each measurement, the member of its coset closest to the orientations
estimated so far, which turns the measurements back into something the
synchronization can use. `detect_symmetry` reads the group off the angles of
the triangle residuals, which works without knowing any orientation at all
because the angle of a rotation does not change under conjugation.
"""

from __future__ import annotations

import math
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import project_to_rotation, rotation_angle

__all__ = [
    'detect_symmetry',
    'estimate_group',
    'point_group',
    'snap_to_symmetry',
    'symmetry_order',
]

# Two orientations closer than this are the same group element. The groups
# are exact and their elements are far apart, so the value only has to sit
# between rounding error and the smallest angle any group turns by, which is
# 2 pi / 60 for the icosahedral group.
_SAME = 1e-6

_GOLDEN = 0.5 * (1.0 + math.sqrt(5.0))

_ORDERS = {'T': 12, 'O': 24, 'I': 60}


def _about(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """A rotation of ``angle`` [rad] about ``axis``, by Rodrigues' formula."""
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    cross = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    turn: NDArray[np.float64] = (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )
    return turn


def _close(generators: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Every product of the generators: the group they generate.

    Breadth first from the identity, multiplying by each generator and
    keeping what has not been seen. It terminates because a finite subgroup
    of SO(3) is finite, and generating this way rather than writing the
    elements out means the result is a group by construction.
    """
    identity = np.eye(3)
    elements = [identity]
    frontier = [identity]
    while frontier:
        found = []
        for element in frontier:
            for generator in generators:
                candidate = generator @ element
                if not any(
                    np.abs(candidate - seen).max() < _SAME for seen in elements
                ):
                    elements.append(candidate)
                    found.append(candidate)
        frontier = found
    return np.stack(elements)


def point_group(name: str) -> NDArray[np.float64]:
    """The rotations of a point group, ``(order, 3, 3)``.

    The identity comes first; the rest are in no particular order.

    Args:
        name: ``'C1'``, ``'C<n>'``, ``'D<n>'``, ``'T'``, ``'O'`` or ``'I'``,
            in the axis convention of RELION. A cyclic group turns about z,
            a dihedral group adds a two-fold about x, and the icosahedral
            group has two-fold axes along x, y and z.

    Raises:
        ValueError: If there is no such point group, or if the elements
            found do not form a group of the expected order, which would
            mean a convention in this module had drifted.
    """
    text = name.strip().upper()
    match = re.fullmatch(r'([CD])(\d+)', text)
    if match:
        kind, order = match.group(1), int(match.group(2))
        if order < 1:
            raise ValueError(f'{name!r} is not a point group.')
        generators = [_about((0.0, 0.0, 1.0), 2.0 * math.pi / order)]
        if kind == 'D':
            generators.append(_about((1.0, 0.0, 0.0), math.pi))
        expected = order * (2 if kind == 'D' else 1)
    elif text == 'T':
        # A three-fold about a body diagonal of the cube and a two-fold
        # about z: the rotations of the tetrahedron.
        generators = [
            _about((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0),
            _about((0.0, 0.0, 1.0), math.pi),
        ]
        expected = _ORDERS['T']
    elif text == 'O':
        # A four-fold about z and a three-fold about a body diagonal: the
        # rotations of the cube.
        generators = [
            _about((0.0, 0.0, 1.0), 0.5 * math.pi),
            _about((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0),
        ]
        expected = _ORDERS['O']
    elif text == 'I':
        # Two-folds along the axes, with a five-fold about (0, 1, phi),
        # which is the convention RELION calls I1.
        generators = [
            _about((0.0, 0.0, 1.0), math.pi),
            _about((0.0, 1.0, _GOLDEN), 0.4 * math.pi),
        ]
        expected = _ORDERS['I']
    else:
        raise ValueError(
            f'{name!r} is not a point group. Use C<n>, D<n>, T, O or I.'
        )

    elements = _close(generators)
    if len(elements) != expected:
        raise ValueError(
            f'The generators of {name!r} closed into {len(elements)} '
            f'elements rather than {expected}.'
        )
    return elements


def symmetry_order(name: str) -> int:
    """How many rotations a point group has, without building them."""
    return len(point_group(name))


def snap_to_symmetry(
    measurements: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    orientations: ArrayLike,
    group: ArrayLike,
) -> NDArray[np.float64]:
    """Move each measurement onto the symmetry copy that fits best.

    For a pair ``(i, j)`` whose measurement is ``A``, the copies are
    ``A @ Q_j @ g @ Q_j.T`` over the group, and the one kept maximizes
    ``trace(Q_i.T @ A @ Q_j @ g)``. At the solution this is exact rather
    than approximate: with ``A = Q_i @ g0 @ Q_j.T`` the objective reduces to
    ``trace(g0 @ g)``, which is largest at ``g = g0.T``, and the copy kept
    is ``Q_i @ Q_j.T``.

    A group of one leaves the measurements alone, so there is no separate
    path for particles without symmetry.

    Args:
        measurements: Relative orientations of the pairs, ``(pairs, 3, 3)``.
        left: Index of the first particle of each pair, ``(pairs,)``.
        right: Index of the second, ``(pairs,)``.
        orientations: Orientations estimated so far, ``(particles, 3, 3)``.
        group: Rotations of the point group, ``(order, 3, 3)``.

    Returns:
        Measurements of shape ``(pairs, 3, 3)`` that a synchronization can
        take at face value.
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)
    group = np.asarray(group, dtype=np.float64)
    left = np.asarray(left)
    right = np.asarray(right)
    if len(group) == 1:
        return measurements

    first = orientations[left]
    second = orientations[right]
    # trace(Q_i.T @ A @ Q_j @ g) for every pair and every element at once.
    common = np.swapaxes(first, -2, -1) @ measurements @ second
    scores = np.einsum('pij,gji->pg', common, group)
    best = group[np.argmax(scores, axis=1)]
    snapped: NDArray[np.float64] = (
        measurements @ second @ best @ np.swapaxes(second, -2, -1)
    )
    return snapped


def residual_angles(
    measurements: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    orientations: ArrayLike,
    group: ArrayLike,
) -> NDArray[np.float64]:
    """How far each measurement is from the orientations, modulo symmetry.

    The angle [rad] between the measurement and the closest symmetry copy of
    what the orientations predict. Measuring modulo the group matters: with
    C4 symmetry a residual of 90 degrees is a perfect measurement, and a
    rule that discarded it would throw away the good pairs.
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)
    group = np.asarray(group, dtype=np.float64)
    first = orientations[np.asarray(left)]
    second = orientations[np.asarray(right)]
    common = np.swapaxes(first, -2, -1) @ measurements @ second
    traces = np.einsum('pij,gji->pg', common, group)
    best = np.max(traces, axis=1)
    angles: NDArray[np.float64] = np.arccos(
        np.clip(0.5 * (best - 1.0), -1.0, 1.0)
    )
    return angles


def triangle_residuals(
    measurements: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    rng: np.random.Generator,
    samples: int = 4096,
) -> NDArray[np.float64]:
    """Angles [rad] of the residuals around closed triangles of the graph.

    Around a triangle the measured orientations compose to
    ``Q_i @ (g1 @ g2 @ g3) @ Q_i.T``, a group element conjugated into a
    frame that is not known. Its *angle* survives that conjugation, so these
    angles say what the symmetry is before a single orientation has been
    estimated.

    Args:
        measurements: Relative orientations of the pairs, ``(pairs, 3, 3)``.
        left: Index of the first particle of each pair, ``(pairs,)``.
        right: Index of the second, ``(pairs,)``.
        rng: Where the sampled triangles come from.
        samples: How many triangles to look for.

    Returns:
        One angle for each triangle found, possibly none.
    """
    measurements = np.asarray(measurements, dtype=np.float64)
    left = np.asarray(left)
    right = np.asarray(right)

    # An edge is usable in either direction, so index both.
    forward = {
        (int(a), int(b)): i
        for i, (a, b) in enumerate(zip(left, right, strict=True))
    }
    edges = dict(forward)
    edges.update({(b, a): ~i for (a, b), i in forward.items()})

    def relative(a: int, b: int) -> NDArray[np.float64] | None:
        index = edges.get((a, b))
        if index is None:
            return None
        if index < 0:
            flipped: NDArray[np.float64] = np.swapaxes(
                measurements[~index], -2, -1
            )
            return flipped
        forwards: NDArray[np.float64] = measurements[index]
        return forwards

    neighbours: dict[int, list[int]] = {}
    for a, b in edges:
        neighbours.setdefault(a, []).append(b)

    found = []
    keys = list(forward)
    if not keys:
        return np.empty(0)
    for index in rng.integers(0, len(keys), size=samples):
        a, b = keys[int(index)]
        shared = neighbours.get(b, ())
        if not shared:
            continue
        c = int(shared[int(rng.integers(0, len(shared)))])
        if c in (a, b):
            continue
        parts = (relative(a, b), relative(b, c), relative(c, a))
        if any(part is None for part in parts):
            continue
        loop = parts[0] @ parts[1] @ parts[2]  # type: ignore[operator]
        found.append(loop)
    if not found:
        return np.empty(0)
    return rotation_angle(np.stack(found))


def detect_symmetry(
    measurements: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    rng: np.random.Generator,
    samples: int = 4096,
    tolerance: float = math.radians(15.0),
) -> tuple[str, dict[str, float]]:
    """Guess the point group from the triangle residuals alone.

    Each candidate group is scored by the share of triangle residuals whose
    angle falls within ``tolerance`` of one of the angles that group turns
    by. The smallest group that explains almost as much as the best one
    wins, so that noise alone does not argue for more symmetry than there is.

    What this can and cannot do is worth being plain about. It sees only
    the angles the group turns by, never the axes, so it separates the
    cyclic groups cleanly and confuses the rest with them: ``D2`` turns by
    nothing and by half a turn, which is exactly ``C2``'s set of angles,
    and ``T`` shares its angles with ``C6``. So the name that comes back
    is the smallest group whose angles fit, and a larger group with the
    same angles is not ruled out. It is a warning that the symmetry asked
    for does not match the data, not a decision about what to use.

    The scores say more than the name does, and they are corrected for
    what each group would explain by chance.

    Args:
        measurements: Relative orientations of the pairs, ``(pairs, 3, 3)``.
        left: Index of the first particle of each pair.
        right: Index of the second.
        rng: Where the sampled triangles come from.
        samples: How many triangles to look at.
        tolerance: How far an angle may be from a symmetry angle [rad].

    Returns:
        The name of the group, and the share in ``[0, 1]`` that each
        candidate explains. An empty report means no triangle was found,
        and the name is then ``'C1'``.
    """
    angles = triangle_residuals(measurements, left, right, rng, samples)
    if len(angles) == 0:
        return 'C1', {}

    candidates = [f'C{n}' for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)]
    candidates += ['T', 'O', 'I']
    scores: dict[str, float] = {}
    for name in candidates:
        expected = rotation_angle(point_group(name))
        distance = np.abs(angles[:, None] - expected[None, :]).min(axis=1)
        observed = float(np.mean(distance < tolerance))

        # Corrected for what the group would explain by chance. A large
        # group has many angles, and a band around each of them covers so
        # much of the range that noise alone looks like symmetry: C12 has
        # a turn every thirty degrees, so with a fifteen degree tolerance
        # it explains everything, symmetric or not. The correction asks
        # how much better the group does than it would on rotations drawn
        # uniformly, whose angle has density (1 - cos t) / pi.
        grid = np.linspace(0.0, np.pi, 2048)
        density = (1.0 - np.cos(grid)) / np.pi
        covered = (
            np.abs(grid[:, None] - expected[None, :]).min(axis=1) < tolerance
        )
        chance = float(np.trapezoid(density * covered, grid))
        scores[name] = (observed - chance) / max(1.0 - chance, 1e-9)

    best = max(scores.values())
    # Among the groups that do about as well, the smallest wins: a few
    # stray triangles should not argue for more symmetry than there is.
    for name in candidates:
        if scores[name] >= best - 0.05:
            return name, scores
    return 'C1', scores


def estimate_group(
    residuals: ArrayLike, order: int, tolerance: float = math.radians(25.0)
) -> NDArray[np.float64]:
    """The group the residuals of a synchronization actually fall on.

    A synchronization fixes the orientations only up to one global
    rotation ``U``, and a point group acts in the frame of the particle.
    So the group present in the estimated frame is ``U.T @ G @ U``, which
    is conjugate to the named group but not equal to it: its axes point
    wherever ``U`` left them. Snapping against the named group instead
    takes a perfect solution apart, which is a confusing way to fail,
    because the orientations were right before the step meant to improve
    them.

    Nothing here needs to know ``U``. Once the orientations are roughly
    right the residuals ``Q_i.T @ A_ij @ Q_j`` are the elements of that
    conjugated group, so it can simply be read off them.

    Greedy clustering: take the densest residual not yet accounted for,
    average what sits near it, and repeat until the group is full or
    nothing is left.

    Args:
        residuals: Rotations ``(pairs, 3, 3)``, the measurements seen
            through the current orientations.
        order: How many elements the group has. One returns the identity
            alone, which is the answer for a particle with no symmetry.
        tolerance: How close two residuals must be to be the same element
            [rad].

    Returns:
        Rotations ``(found, 3, 3)`` with the identity first, and at most
        ``order`` of them. Fewer means the residuals did not support more,
        which is itself worth knowing.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if order < 1:
        raise ValueError(f'A group has at least one element, not {order}.')
    identity = np.eye(3)
    if order == 1 or len(residuals) == 0:
        return identity[None]

    found = [identity]
    unexplained = residuals
    while len(found) < order and len(unexplained) > 0:
        # Drop what the elements found so far already explain.
        known = np.stack(found)
        distance = np.min(
            [
                rotation_angle(np.swapaxes(unexplained, -2, -1) @ element)
                for element in known
            ],
            axis=0,
        )
        unexplained = unexplained[distance > tolerance]
        if len(unexplained) == 0:
            break

        # The densest of what is left, by how many sit near it.
        sample = unexplained
        if len(sample) > 512:
            sample = sample[:: max(len(sample) // 512, 1)]
        spread = np.stack(
            [
                rotation_angle(np.swapaxes(unexplained, -2, -1) @ candidate)
                for candidate in sample
            ]
        )
        near = spread <= tolerance
        densest = int(np.argmax(near.sum(axis=1)))
        members = unexplained[near[densest]]
        if len(members) == 0:
            break
        found.append(project_to_rotation(members.mean(axis=0)))

    return np.stack(found)
