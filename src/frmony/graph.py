"""Which pairs of subtomograms to align.

Aligning every pair costs time that grows with the square of the number of
particles, and buys little: synchronization needs the graph of pairs to be
connected and well mixed, not complete. A few partners each is enough, as
long as the graph has no bottleneck that a global orientation could drift
across.

Two generators are combined. A ring of log-spaced jumps gives a connected
backbone with short paths, in the manner of a small-world graph, so no part
of the graph is far from any other. Uniform random pairs break up whatever
regularity the ring has left. Their union is what gets aligned.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'bridges',
    'build_pairs',
    'is_connected',
    'pieces',
    'spanning_tree',
]


def _ring_pairs(count: int, jumps: int) -> set[tuple[int, int]]:
    """Pairs that link each particle to others at log-spaced distances.

    Each particle reaches ``jumps`` others ahead of it and is reached by
    as many from behind, so this gives it about twice ``jumps`` partners.
    """
    if count < 2:
        return set()
    half = max(count // 2, 1)
    exponents = np.linspace(0.0, 1.0, max(jumps, 1))
    distances = np.unique(np.round(half**exponents).astype(np.int64))
    distances = distances[distances > 0]
    pairs = set()
    for index in range(count):
        for distance in distances:
            other = (index + int(distance)) % count
            if other != index:
                pairs.add((min(index, other), max(index, other)))
    return pairs


def _random_pairs(
    count: int, draws: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    """Pairs that link each particle to ``draws`` others at random.

    Each particle is also drawn by about as many others, so this gives it
    about twice ``draws`` partners.
    """
    if count < 2 or draws < 1:
        return set()
    first = np.repeat(np.arange(count), draws)
    second = (first + rng.integers(1, count, len(first))) % count
    stacked = np.sort(np.stack((first, second), axis=1), axis=1)
    return {(int(a), int(b)) for a, b in stacked}


def build_pairs(
    count: int, partners: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Choose the pairs of particles to align.

    Args:
        count: How many particles there are.
        partners: Roughly how many others each particle should be aligned
            to. Clamped to ``count - 1``, at which point every pair is
            chosen and the graph is complete.
        rng: Where the random pairs come from.

    Returns:
        The index of the first and of the second particle of each pair,
        sorted, with ``first < second`` and no pair repeated.

    Raises:
        ValueError: If there are fewer than two particles, or if
            ``partners`` is not positive.
    """
    if count < 2:
        raise ValueError(
            f'Aligning pairs needs at least two particles, not {count}.'
        )
    if partners < 1:
        raise ValueError(
            f'Each particle needs at least one partner, not {partners}.'
        )

    partners = min(partners, count - 1)
    if partners >= count - 1:
        first, second = np.triu_indices(count, 1)
        return first.astype(np.int64), second.astype(np.int64)

    # A pair is a partner to both of its ends, so a particle that reaches
    # out to some number of others ends up with twice that many. Handing
    # the whole of ``partners`` to the ring and half of it again to the
    # random pairs, which is what this did, gave each particle between
    # two and a half and three times what was asked for: 48 for 16, and
    # 380 for 150 of ten thousand. Half the partners come from each
    # generator, and each generator reaches out to half of its share.
    jumps = max(partners // 4, 1)
    draws = max((partners - 2 * jumps) // 2, 0)
    pairs = _ring_pairs(count, jumps) | _random_pairs(count, draws, rng)
    ordered = np.array(sorted(pairs), dtype=np.int64)
    return ordered[:, 0], ordered[:, 1]


def _labels(
    count: int, left: NDArray[np.int64], right: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Which connected component each particle belongs to.

    A union-find over the edges, with path halving. Nothing here is a
    bottleneck, so the simplest correct thing will do.
    """
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for a, b in zip(left, right, strict=True):
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a
    return np.array([find(index) for index in range(count)], dtype=np.int64)


def pieces(
    count: int, left: NDArray[np.int64], right: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Which piece of the graph each particle is in, ``(count,)``.

    The pieces are numbered from the largest, which is piece nought; a
    particle no pair touches is a piece of its own.
    """
    labels = _labels(count, np.asarray(left), np.asarray(right))
    _, inverse, sizes = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    rank = np.empty_like(sizes)
    rank[np.argsort(-sizes, kind='stable')] = np.arange(len(sizes))
    return rank[inverse].astype(np.int64)


def is_connected(
    count: int, left: NDArray[np.int64], right: NDArray[np.int64]
) -> bool:
    """Whether every particle is reachable from every other.

    A graph in pieces has a separate, arbitrary global orientation for each
    piece, and nothing in the measurements relates them, so the result would
    be meaningless across the pieces.
    """
    if count < 1:
        return False
    return bool(len(np.unique(_labels(count, left, right))) == 1)


def spanning_tree(
    count: int,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.int64]:
    """A maximum-weight spanning forest, as indices into the pairs.

    Kruskal's algorithm: take the edges in order of decreasing weight and
    keep the ones that join two pieces.

    The tree is what the synchronization starts from. A tree has no cycles,
    so every one of its edges can be satisfied exactly and at once, by
    walking out from a root. That matters most with symmetry, where a cycle
    is where the ambiguity of each measurement first shows up: along a tree
    there is no ambiguity to resolve, only a choice of representative.

    Returns:
        Indices of the pairs in the tree, at most ``count - 1`` of them and
        fewer if the graph is in pieces.
    """
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    chosen = []
    for edge in np.argsort(-np.asarray(weights), kind='stable'):
        root_a, root_b = find(int(left[edge])), find(int(right[edge]))
        if root_a != root_b:
            parent[root_b] = root_a
            chosen.append(int(edge))
            if len(chosen) == count - 1:
                break
    return np.array(chosen, dtype=np.int64)


def bridges(
    count: int,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    smallest: int = 1,
) -> NDArray[np.int64]:
    """Pairs whose removal would break the graph into pieces.

    A connected graph is not enough. If two groups of particles hang
    together by a single pair, then everything one group says about its
    orientation relative to the other rests on that one measurement, and
    nothing else in the data can contradict it. The synchronization will
    satisfy it exactly, because it can, and the two groups come out in
    frames with no real relation -- while `is_connected` says the graph is
    fine and every residual looks small.

    Measured on a deliberately bottlenecked graph with one bad bridge, the
    orientations inside each group were exact and those across the bridge
    were wrong by a hundred and twelve degrees, with nothing to show for
    it.

    Found by depth-first search: an edge is a bridge when the subtree below
    it can reach nothing above it except through the edge itself.

    Args:
        count: How many particles there are.
        left: First particle of each pair.
        right: Second particle of each pair.
        smallest: Only the bridges that would cut off at least this many
            particles. A bridge to one particle that nothing else holds
            puts nothing in doubt but that particle, and on real data
            there are many: a pick of something else has one pair that
            looks right by chance, and that pair is a bridge.

    Returns:
        Indices of the pairs that are bridges, possibly none.
    """
    found, cut_off = _bridges(count, left, right)
    return found[cut_off >= smallest]


def _bridges(
    count: int, left: NDArray[np.int64], right: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Every bridge, and how many particles it would cut off: the
    smaller of the two sides."""
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for edge, (first, second) in enumerate(zip(left, right, strict=True)):
        neighbours.setdefault(int(first), []).append((int(second), edge))
        neighbours.setdefault(int(second), []).append((int(first), edge))

    seen = np.zeros(count, dtype=bool)
    entered = np.zeros(count, dtype=np.int64)
    lowest = np.zeros(count, dtype=np.int64)
    below = np.ones(count, dtype=np.int64)
    found: list[int] = []
    sides: list[int] = []
    clock = 0

    for root in range(count):
        if seen[root]:
            continue
        first_found = len(found)
        # Explicit stack: a recursive search would overflow on the long
        # paths a ring of ten thousand particles makes.
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]
        while stack:
            node, came_by, visited = stack.pop()
            if visited == 0:
                seen[node] = True
                entered[node] = lowest[node] = clock
                clock += 1
            others = neighbours.get(node, ())
            if visited < len(others):
                stack.append((node, came_by, visited + 1))
                other, edge = others[visited]
                if edge == came_by:
                    continue
                if seen[other]:
                    lowest[node] = min(lowest[node], entered[other])
                else:
                    stack.append((other, edge, 0))
            elif came_by >= 0:
                parent = (
                    int(left[came_by])
                    if int(right[came_by]) == node
                    else int(right[came_by])
                )
                lowest[parent] = min(lowest[parent], lowest[node])
                below[parent] += below[node]
                if lowest[node] > entered[parent]:
                    found.append(came_by)
                    sides.append(int(below[node]))
        # The particles of this piece are what its root has below it.
        whole = int(below[root])
        for index in range(first_found, len(found)):
            sides[index] = min(sides[index], whole - sides[index])
    order = np.argsort(np.asarray(found, dtype=np.int64), kind='stable')
    return (
        np.asarray(found, dtype=np.int64)[order],
        np.asarray(sides, dtype=np.int64)[order],
    )
