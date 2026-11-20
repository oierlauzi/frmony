"""The score two subtomograms are compared by.

A subtomogram is not a measurement of a particle. It is a measurement of
some of a particle: a tomogram is built from tilts over a limited range, so
a wedge of Fourier space is never measured at all, and what is measured is
multiplied by the contrast transfer function, which crosses zero at
frequencies of its own. Comparing two of them without accounting for that
compares the shapes of their missing regions as much as the particles.

So each particle is carried as a pair ``(D, W)``: its Fourier transform,
already multiplied by whatever reached it, and a non-negative weight saying
how much of each frequency did. Then with ``Omega(k) = W_i(k) W_j(Q^-1 k)``,
the frequencies both of them measured,

    num(Q)  = sum_k Omega D_i conj(Lambda_Q D_j)
    den1(Q) = sum_k Omega |D_i|^2
    den2(Q) = sum_k Omega |Lambda_Q D_j|^2
    c(Q)    = num / sqrt(den1 * den2).

Each of those three is a correlation over the rotation group of one
function belonging to each particle, so each is one fast rotational
matching, and the score is the three combined sample by sample.

The weight only multiplies. Dividing by it -- deconvolving, to recover an
estimate of the particle itself -- would amplify exactly the frequencies
that were measured worst, and at a zero of the transfer function it is not
a question of amplification but of dividing by nothing at all.

The rotational stage uses ``|D|`` in place of ``D``, which costs the phases
but gains everything: an amplitude does not move when the particle does, so
the rotation can be found before the shift is known. The phases come back
in `frmony.matching.translation`, which scores the candidates in full.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .._transfers import to_device
from ..harmonics.transform import Basis, expand
from .rotation import correlation_grid, radial_overlap

__all__ = ['Expansion', 'expand_particle', 'rotation_score']


@dataclass(frozen=True)
class Expansion:
    """The three spherical expansions a particle is compared through.

    Expanded once per particle and then correlated against any number of
    others, which is what makes a graph of pairs affordable: the expansion
    is the expensive part and it does not depend on the partner.
    """

    amplitude: jax.Array
    """Of ``|D| * W``, the numerator's share. Of ``D * W`` itself when the
    expansion keeps the phases."""
    energy: jax.Array
    """Of ``|D|**2 * W``, a denominator's share."""
    weight: jax.Array
    """Of ``W`` alone, the other denominator's share."""


jax.tree_util.register_dataclass(
    Expansion,
    data_fields=['amplitude', 'energy', 'weight'],
    meta_fields=[],
)


def expand_particle(
    data: ArrayLike, weight: ArrayLike, basis: Basis, phases: bool = False
) -> Expansion:
    """Expand what the rotational stage of a comparison needs.

    With ``phases`` the numerator is expanded from ``D`` and not from
    ``|D|``, so the first stage is the real part of the constrained
    correlation at no shift, at every rotation. That gives up what the
    amplitudes were chosen for: the rotation is then only found if the
    two particles are already near enough to centred that their phases
    agree, about a voxel of the band at its outermost sphere.

    What it buys is the signal. With noise ``n`` on a signal ``s``, a
    product of two spectra has an expectation that goes as ``s**2``, and
    a product of two amplitudes one that goes as ``s**4`` over the noise
    squared, because an amplitude of mostly noise barely moves when the
    signal under it does. Simulations at a signal to noise ratio of one do
    not show the difference. Subtomograms do: on EMPIAR-10304 the
    amplitudes put the right rotation at the sixty-third percentile of
    their grid, where chance is the fiftieth, and no centring or
    standardising of them changed that, while the phases put it at the
    ninety-ninth.

    Args:
        data: A centred Fourier volume, ``(size, size, size)``.
        weight: Its non-negative Fourier weight, the same shape.
        basis: What `frmony.harmonics.make_basis` built.
        phases: Keep the phases of the data in the numerator.
    """
    data = to_device(data)
    weight = jnp.maximum(to_device(weight), 0.0)
    amplitude = jnp.abs(data)
    return Expansion(
        amplitude=expand((data if phases else amplitude) * weight, basis),
        energy=expand(amplitude * amplitude * weight, basis),
        weight=expand(weight, basis),
    )


def rotation_score(
    left: Expansion,
    right: Expansion,
    bandwidth: int,
    epsilon: float = 1e-12,
) -> jax.Array:
    """The constrained correlation of two particles at every rotation.

    Three fast rotational matchings, combined sample by sample. The result
    lies in ``[-1, 1]`` wherever the two particles share any measured
    frequency at all, because it is a normalized correlation in the inner
    product that their common weight defines.

    Args:
        left: The expansion of the first particle.
        right: The expansion of the second.
        bandwidth: The bandwidth both were expanded at.
        epsilon: Guards the division where two particles share no measured
            frequency, which a wide missing wedge and an unlucky rotation
            can arrange.

    Returns:
        A real array ``(2B, 2B, 2B)``, indexed by
        `frmony.matching.rotation.grid_to_euler`.
    """
    numerator = correlation_grid(
        radial_overlap(left.amplitude, right.amplitude), bandwidth
    )
    first = correlation_grid(
        radial_overlap(left.energy, right.weight), bandwidth
    )
    second = correlation_grid(
        radial_overlap(left.weight, right.energy), bandwidth
    )
    # The two denominators are correlations of non-negative functions and so
    # are non-negative themselves, up to the rounding of the transforms.
    denominator = jnp.sqrt(jnp.abs(first) * jnp.abs(second) + epsilon)
    return numerator / denominator
