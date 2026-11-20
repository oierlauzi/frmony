"""Choosing which pairs to align, and the shape of the graph they make."""

from __future__ import annotations

import numpy as np
import pytest

from frmony.graph import build_pairs, is_connected, spanning_tree


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(4242)


@pytest.mark.parametrize('count', [2, 3, 10, 64, 517])
@pytest.mark.parametrize('partners', [1, 2, 8, 32])
def test_pairs_are_well_formed(
    rng: np.random.Generator, count: int, partners: int
) -> None:
    left, right = build_pairs(count, partners, rng)

    assert len(left) == len(right)
    assert np.all(left < right), 'pairs are ordered'
    assert np.all(right < count)
    assert np.all(left >= 0)
    stacked = np.stack((left, right), axis=1)
    assert len(np.unique(stacked, axis=0)) == len(stacked), 'no repeats'


@pytest.mark.parametrize('count', [2, 5, 37, 200])
@pytest.mark.parametrize('partners', [1, 3, 16])
def test_the_graph_is_connected(
    rng: np.random.Generator, count: int, partners: int
) -> None:
    """Pieces would each have their own global orientation, unrelated to
    the others, which makes the whole result meaningless."""
    left, right = build_pairs(count, partners, rng)
    assert is_connected(count, left, right)


@pytest.mark.parametrize('count', [2, 6, 40])
def test_partners_are_clamped_to_a_complete_graph(
    rng: np.random.Generator, count: int
) -> None:
    left, _ = build_pairs(count, 10 * count, rng)
    assert len(left) == count * (count - 1) // 2

    same, _ = build_pairs(count, count - 1, rng)
    assert len(same) == len(left)


def test_every_particle_gets_a_partner(rng: np.random.Generator) -> None:
    count = 128
    left, right = build_pairs(count, 6, rng)
    degree = np.bincount(np.concatenate((left, right)), minlength=count)

    assert degree.min() >= 1
    assert degree.mean() > 3.0


def test_more_partners_means_more_pairs(rng: np.random.Generator) -> None:
    counts = [
        len(build_pairs(200, partners, rng)[0]) for partners in (2, 4, 8, 16)
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


@pytest.mark.parametrize('count', [0, 1, -3])
def test_too_few_particles_is_refused(
    rng: np.random.Generator, count: int
) -> None:
    with pytest.raises(ValueError, match='two particles'):
        build_pairs(count, 4, rng)


@pytest.mark.parametrize('partners', [0, -1])
def test_too_few_partners_is_refused(
    rng: np.random.Generator, partners: int
) -> None:
    with pytest.raises(ValueError, match='partner'):
        build_pairs(8, partners, rng)


def test_a_broken_graph_is_not_connected() -> None:
    left = np.array([0, 1, 3])
    right = np.array([1, 2, 4])
    assert not is_connected(5, left, right)
    assert is_connected(3, left[:2], right[:2])


def test_spanning_tree_spans(rng: np.random.Generator) -> None:
    count = 64
    left, right = build_pairs(count, 5, rng)
    weights = rng.uniform(0.0, 1.0, len(left))

    tree = spanning_tree(count, left, right, weights)
    assert len(tree) == count - 1
    assert is_connected(count, left[tree], right[tree])


def test_spanning_tree_takes_the_heaviest_edges(
    rng: np.random.Generator,
) -> None:
    """The tree seeds the orientations, so it should lean on the pairs that
    were aligned most confidently."""
    count = 50
    left, right = build_pairs(count, 8, rng)
    weights = rng.uniform(0.0, 1.0, len(left))

    tree = spanning_tree(count, left, right, weights)
    assert weights[tree].mean() > weights.mean()


def test_spanning_tree_of_a_broken_graph_is_a_forest() -> None:
    left = np.array([0, 2])
    right = np.array([1, 3])
    weights = np.array([1.0, 1.0])

    tree = spanning_tree(4, left, right, weights)
    assert len(tree) == 2  # two pieces, so one edge short of a tree
