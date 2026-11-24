"""Particles aligned to a reference, one at a time.

Every particle is aligned to the reference as `frmony.pipeline` aligns
one particle to another: the same expansions, the same search over
rotations and shifts at once, the same particles handed to the device a
block at a time. Here each pair is the reference and a particle, and
there is no graph and nothing to synchronize: the reference's frame is
the one the particles are put in.

A pair measures ``V_ref(x) = V_j(A.T @ (x - a))``. The reference is at
the origin and unturned, so in the terms `frmony.geometry` uses,

    Q_j = A.T        and        t_j = -A.T @ a.

The reference is taken to be clean: measured everywhere, with no wedge,
no transfer and no noise. Particles may be anything the loader reads.

What it cannot be is a check on itself. Particles aligned to a reference
come to look like it whether it is right or not, so an average of them
says little about the reference, and the agreement of its halves is
biased towards it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .blocks import align_against, block_size
from .engine import Engine, Search
from .particles import Subtomograms, recentred_on_host
from .pipeline import memory_in_bytes

__all__ = ['Aligned', 'align_to_reference']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Aligned:
    """Where each particle sits relative to the reference."""

    orientations: NDArray[np.float64]
    """``(particles, 3, 3)``, in the convention of `frmony.geometry`."""
    shifts: NDArray[np.float64]
    """``(particles, 3)`` in voxels of the original box, ``(x, y, z)``."""
    scores: NDArray[np.float64]
    """How well each matched the reference, a constrained correlation."""


def align_to_reference(
    subtomograms: Subtomograms,
    reference: Subtomograms,
    search: Search | None = None,
) -> Aligned:
    """Find each particle's orientation and centre against a reference.

    Args:
        subtomograms: What `frmony.particles.load_subtomograms` built.
        reference: What `frmony.particles.load_reference` built from a
            clean volume, for the same particles.
        search: How to search. The defaults if not given.

    Returns:
        An `Aligned`.
    """
    search = search or Search()
    if len(reference) != 1:
        raise ValueError('A reference is one volume.')
    if reference.band != subtomograms.band:
        raise ValueError('The reference was not cropped like the particles.')
    count = len(subtomograms)
    engine = Engine.build(subtomograms.band, search)

    data, weight = recentred_on_host(
        subtomograms, None, search.batch, noise=search.noise_spectrum
    )
    clean, everywhere = recentred_on_host(reference, None, noise=False)
    fixed = engine.particles(clean, everywhere, 1)(0, 1)
    size = block_size(count, engine.per_particle(), memory_in_bytes(search))

    started = time.perf_counter()
    aligned = align_against(
        fixed,
        count,
        size,
        engine.particles(data, weight, size),
        engine.aligner(engine.joint(search.max_shift)),
        engine.batch,
    )
    _LOGGER.info(
        'Aligned %d particles to the reference in %.1f s.',
        count,
        time.perf_counter() - started,
    )

    turns = np.swapaxes(aligned.rotation.astype(np.float64), -2, -1)
    centres = -np.einsum('nij,nj->ni', turns, aligned.shift.astype(np.float64))
    return Aligned(
        orientations=turns,
        shifts=centres * subtomograms.scale,
        scores=aligned.score.astype(np.float64),
    )
