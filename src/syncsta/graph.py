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

__all__ = ['build_pairs', 'is_connected', 'spanning_tree']


def _ring_pairs(count: int, partners: int) -> set[tuple[int, int]]:
    """Pairs that link each particle to others at log-spaced distances."""
    if count < 2:
        return set()
    half = max(count // 2, 1)
    exponents = np.linspace(0.0, 1.0, max(partners, 1))
    jumps = np.unique(np.round(half**exponents).astype(np.int64))
    jumps = jumps[jumps > 0]
    pairs = set()
    for index in range(count):
        for jump in jumps:
            other = (index + int(jump)) % count
            if other != index:
                pairs.add((min(index, other), max(index, other)))
    return pairs


def _random_pairs(
    count: int, partners: int, rng: np.random.Generator
) -> set[tuple[int, int]]:
    """Pairs that link each particle to about ``partners`` others at random."""
    if count < 2 or partners < 1:
        return set()
    first = np.repeat(np.arange(count), max(partners // 2, 1))
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

    pairs = _ring_pairs(count, partners) | _random_pairs(count, partners, rng)
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
