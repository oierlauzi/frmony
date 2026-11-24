"""How pairs of particles are searched, set up once for a run.

Both ways this package aligns particles, to one another and to a
reference, search a pair the same way: spherical expansions on the same
spheres, the same grid of rotations, the same lattice of trial shifts and
the same band for the shifts. `Search` is what can be said about that,
and `Engine` is it built for particles of a given band.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ._transfers import to_device
from .blocks import pad_to, per_particle
from .harmonics.shells import band_radii
from .harmonics.transform import Basis, make_basis
from .matching.correlation import Expansion, expand_particle
from .matching.pairwise import (
    Alignment,
    Joint,
    Particles,
    Settings,
    align_chunk,
    lattice_spacing,
    shift_lattice,
)

__all__ = ['Engine', 'Search']


@dataclass(frozen=True)
class Search:
    """How a pair of particles is searched."""

    bandwidth: int = 32
    """Degrees kept in the spherical expansion. The grid of rotations has a
    step of ``180 / bandwidth`` degrees, which interpolation improves on."""
    candidates: int = 10
    """Rotations carried from the first stage of a pairwise alignment into
    the second."""
    max_shift: float = 6.0
    """How far apart the centres of two particles may be, in voxels of
    the cropped band. With the phases kept, rotations and shifts are
    searched at once out to this distance, and the further it reaches
    the more a pair has to stand out, so it should be what the picking
    warrants and no more: on centred particles a reach of nothing
    aligned 40 pairs in a hundred and a reach of five voxels 22."""
    lowest: float = 0.125
    """Innermost sphere of the expansion, as a fraction of the box. The
    frequencies below it carry a particle's mass rather than its
    orientation; see `frmony.harmonics.band_radii`."""
    phases: bool = True
    """Keep the phases in the first, rotational stage. They find rotations
    that the amplitudes are too noisy to: on EMPIAR-10304 the amplitudes
    aligned three pairs in a thousand, which is chance. The price is that
    a phased score only sees a pair at the shift it is at, which is why
    rotations and shifts are then searched together; see `max_shift`.
    Turn it off only for clean volumes, where amplitudes are enough and
    need no shift searched."""
    noise_spectrum: bool = False
    """Divide each particle by the noise spectrum of its acquisition; see
    `frmony.particles.noise_spectra`.

    Off. Measured on pairs put on their true centres it helped, and in a
    whole run, where the shifts are searched, it did harm on both
    datasets it was tried on: on the first picking of EMPIAR-10304,
    displaced by thirty angstroms, 84 particles in a hundred within
    fifteen degrees after one pass without it and 54 with it, and on
    the second picking at bin 4 a set that aligned without it did not
    align at all with it. The spectrum it divides by is the total power,
    signal included, so it takes most off the strong low frequencies,
    which are the ones a shift that is not yet known spoils least."""
    batch: int = 8
    """Pairs aligned at once, and particles expanded at once.

    Measured on a 32 GB device, with particles in a band of 38 and the
    rotations and shifts searched together: 1.6 milliseconds a pair at
    two, 1.4 at four and at eight, 2.4 at sixteen, 3.7 at thirty-two and
    4.4 at a hundred and twenty-eight, compilation included. More at
    once is slower, not faster: each pair in flight holds its own grids
    of every trial shift, and past eight they no longer fit where the
    device works fastest. It was thirty-two, reasoned from a device
    waiting on dispatch and not measured.

    Turn it down if a device runs out of memory."""
    memory: float | None = None
    """Device memory to plan for [GB]. None takes what the device says it
    has. Particles that do not fit are handed to it a block at a time;
    see `frmony.blocks`."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.lowest < 0.5:
            raise ValueError('The innermost sphere is inside the box.')


@dataclass(frozen=True)
class Engine:
    """`Search`, built for particles cropped to a band of ``extent``."""

    search: Search
    """What it was built from."""
    basis: Basis
    """The expansions the rotations are searched on."""
    coarse: Basis
    """At half the bandwidth, for the lattice of trial shifts."""
    settings: Settings
    """What a pair is searched for."""
    shells: int
    """Spheres the expansions are made on."""
    spacing: float
    """Of the lattice of trial shifts, in voxels of the band."""
    band: jax.Array
    """One over the frequencies the shifts are searched on, the shape of
    a spectrum. The spheres the rotations are searched on: the lowest
    frequencies carry a particle's mass and not its orientation, and lead
    a search over shifts astray exactly as they would one over rotations.
    Built on the device, since a copy of that shape is how a particle
    crossing on its own would look."""

    @classmethod
    def build(cls, extent: int, search: Search) -> Engine:
        """Set up the search for particles cropped to ``extent``."""
        radii = band_radii(extent, lowest=search.lowest)
        half = max(search.bandwidth // 2, 2)
        offsets = jnp.arange(extent) - extent // 2
        distance = jnp.sqrt(
            offsets[:, None, None] ** 2
            + offsets[None, :, None] ** 2
            + offsets[None, None, :] ** 2
        )
        return cls(
            search=search,
            basis=make_basis(
                radii,
                search.bandwidth,
                extent,
                sampling_bandwidth=2 * search.bandwidth,
            ),
            coarse=make_basis(radii, half, extent, sampling_bandwidth=2 * half),
            settings=Settings(
                bandwidth=search.bandwidth,
                candidates=search.candidates,
                max_shift=search.max_shift,
                phases=search.phases,
            ),
            shells=len(radii),
            spacing=lattice_spacing(extent, float(radii[-1])),
            band=(
                (distance >= float(radii[0]))
                & (distance <= float(radii[-1]) + 0.5)
            ).astype(jnp.float32),
        )

    @property
    def batch(self) -> int:
        """Pairs aligned at once."""
        return max(self.search.batch, 1)

    def per_particle(self) -> int:
        """Bytes one particle takes on the device while it is aligned."""
        return per_particle(
            int(self.band.shape[-1]), self.shells, self.search.bandwidth
        )

    def joint(self, reach: float) -> Joint | None:
        """What searching rotations and shifts at once out to ``reach``
        needs, or nothing if it is not needed.

        Rotations and shifts are searched at once when the phases are
        kept and the particles may be further off than the search over
        shifts reaches by itself. The lattice is searched at half the
        bandwidth.
        """
        if not (self.search.phases and reach >= self.spacing):
            return None
        return Joint(
            coarse=self.coarse,
            fine=self.basis,
            lattice=to_device(shift_lattice(reach, self.spacing), jnp.float32),
            spacing=self.spacing,
        )

    def aligner(
        self, joint: Joint | None
    ) -> Callable[[Particles, Particles, jax.Array], Alignment]:
        """`frmony.matching.pairwise.align_chunk`, with everything but the
        two sets of particles and the pairs given."""
        return functools.partial(
            align_chunk,
            settings=self.settings,
            band=self.band,
            joint=joint,
            batch=self.batch,
        )

    def particles(
        self,
        data: NDArray[np.complex64],
        weight: NDArray[np.float32],
        size: int,
    ) -> Callable[[int, int], Particles]:
        """What hands a block of particles to the device, ready to align.

        Each call dispatches and returns at once: the block travels and
        is expanded while the device finishes what it was given before.
        """

        def make(start: int, stop: int) -> Particles:
            spectra = to_device(pad_to(data[start:stop], size))
            # Two particles are compared over the frequencies both
            # measured, and not in proportion to how well. A weight that
            # follows the transfer is largest where every tilt overlaps,
            # on the tilt axis and at the lowest frequencies, which is
            # also where what surrounds the particle is strongest, and
            # that is not noise the weight knows about. Counting those
            # frequencies by their weight once more took the pairs
            # aligned correctly on EMPIAR-10304 from eighteen in a
            # hundred to five. The average does use the weight, as it
            # must to undo the transfer.
            measured = (
                to_device(pad_to(weight[start:stop], size)) > 0.0
            ).astype(jnp.float32)
            return Particles(
                data=spectra,
                weight=measured,
                expansions=_expand(
                    spectra,
                    measured,
                    self.basis,
                    self.search.phases,
                    self.batch,
                ),
            )

        return make


@functools.partial(jax.jit, static_argnames=('phases', 'batch'))
def _expand(
    data: jax.Array,
    measured: jax.Array,
    basis: Basis,
    phases: bool,
    batch: int,
) -> Expansion:
    """The expansions of a block of particles.

    The basis is an argument and not something closed over. Closed over,
    its tables are folded into the compiled program and uploaded again
    each time one is built, by the runtime and under no name.
    """
    expanded: Expansion = jax.lax.map(
        lambda particle: expand_particle(
            particle[0], particle[1], basis, phases
        ),
        (data, measured),
        batch_size=batch,
    )
    return expanded
