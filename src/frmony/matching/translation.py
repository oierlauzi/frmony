"""The correlation at every shift, for one rotation.

Once a rotation is fixed, a single transform gives the correlation at every
shift at once, because a shift is a phase in Fourier space. The two
denominators of the constrained score do not depend on the shift at all --
the overlap of the two weights is a function of the rotation alone -- so
they are two sums rather than two more transforms.

Everything here works on the pair ``(D, W)``: the Fourier transform of a
subtomogram, already multiplied by whatever the microscope and the
reconstruction applied to it, and a non-negative weight saying how much of
each frequency actually got measured. The weight only ever multiplies.
Nothing is divided by it, which matters because it is exactly zero wherever
the missing wedge is and wherever the contrast transfer function crosses.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    'frequency_grid',
    'rotate_spectrum',
    'shift_correlation',
    'shift_of_index',
]


@functools.lru_cache(maxsize=8)
def frequency_grid(size: int) -> np.ndarray:
    """Offsets of each voxel from the centre, ``(3, size, size, size)``.

    The leading axis is ordered ``(x, y, z)``, which is what a rotation
    acts on, while the volume itself is indexed ``[z, y, x]``.
    """
    axis = np.arange(size) - size // 2
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing='ij')[::-1], axis=0
    ).astype(np.float64)


def rotate_spectrum(
    volume: ArrayLike, rotation: ArrayLike, grid: ArrayLike
) -> jax.Array:
    """Turn a centred spectrum, by reading it at the turned frequencies.

    ``(Lambda_Q g)(k) = g(Q^-1 k)``, so the value at ``k`` is read at
    ``Q.T k``.

    Args:
        volume: A centred volume ``(size, size, size)``, indexed
            ``[z, y, x]``, real or complex.
        rotation: A rotation matrix ``(3, 3)`` acting on ``(x, y, z)``.
        grid: What `frequency_grid` built.

    Returns:
        The turned volume, of the same shape.
    """
    volume = jnp.asarray(volume)
    rotation = jnp.asarray(rotation)
    grid = jnp.asarray(grid)
    size = volume.shape[-1]

    # Q.T applied to every offset, in full precision: a GPU's default
    # reduced-precision product would misplace voxels by a visible fraction.
    source = jnp.einsum(
        'ji,jzyx->izyx',
        rotation.astype(grid.dtype),
        grid,
        precision=jax.lax.Precision.HIGHEST,
    )
    coordinates = [
        source[2] + size // 2,
        source[1] + size // 2,
        source[0] + size // 2,
    ]

    def read(values: jax.Array) -> jax.Array:
        sampled: jax.Array = jax.scipy.ndimage.map_coordinates(
            values, coordinates, order=1, mode='constant', cval=0.0
        )
        return sampled

    if jnp.iscomplexobj(volume):
        return read(volume.real) + 1j * read(volume.imag)
    return read(volume)


def shift_correlation(
    left_data: ArrayLike,
    left_weight: ArrayLike,
    right_data: ArrayLike,
    right_weight: ArrayLike,
    rotation: ArrayLike,
    grid: ArrayLike,
    epsilon: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    """The constrained correlation at every shift, for one rotation.

    With ``Omega(k) = W_i(k) * W_j(Q^-1 k)``,

        num(a)  = Re sum_k Omega D_i conj(Lambda_Q D_j) exp(2 pi i k a)
        den1    =    sum_k Omega |D_i|^2
        den2    =    sum_k Omega |Lambda_Q D_j|^2
        score(a) = num(a) / sqrt(den1 * den2)

    which is a genuine normalized correlation in the inner product that
    ``Omega`` weights, and so lies in ``[-1, 1]``. The mean subtraction that
    a constrained correlation also calls for is already done: it is the
    zero frequency of the masked spectrum, and ``Omega`` at the zero
    frequency does not depend on the rotation, so clearing that one voxel
    of each particle once, when it is loaded, settles it for every rotation
    at once.

    Args:
        left_data: Centred spectrum of the first particle.
        left_weight: Its non-negative Fourier weight, the same shape.
        right_data: Centred spectrum of the second.
        right_weight: Its weight.
        rotation: The rotation to apply to the second, ``(3, 3)``.
        grid: What `frequency_grid` built.
        epsilon: Guards the division where two particles share no measured
            frequency at all.

    Returns:
        The score at every shift, ``(size, size, size)`` and centred so
        that index ``size // 2`` is no shift, and the common denominator.
    """
    left_data = jnp.asarray(left_data)
    left_weight = jnp.asarray(left_weight)

    turned = rotate_spectrum(right_data, rotation, grid)
    turned_weight = rotate_spectrum(right_weight, rotation, grid)
    # A weight is non-negative by construction; interpolation can undershoot.
    overlap = left_weight * jnp.maximum(turned_weight, 0.0)

    first = jnp.sum(overlap * jnp.abs(left_data) ** 2)
    second = jnp.sum(overlap * jnp.abs(turned) ** 2)
    denominator = jnp.sqrt(jnp.abs(first) * jnp.abs(second) + epsilon)

    spectrum = overlap * left_data * jnp.conjugate(turned)
    # The sum over k of X[k] exp(2 pi i k a), with the zero frequency put
    # where an inverse transform expects it and the shifts centred after.
    values = jnp.fft.ifftn(jnp.fft.ifftshift(spectrum)) * spectrum.size
    return jnp.fft.fftshift(values).real / denominator, denominator


def shift_of_index(index: ArrayLike, size: int) -> jax.Array:
    """The shift that an index into the correlation stands for.

    The correlation is indexed ``[z, y, x]`` like the volume; the shift
    comes back as ``(x, y, z)``, which is what the rest of the package
    composes with rotations.

    The shift is the one to apply to the turned second particle to bring it
    onto the first, so that together with the rotation it satisfies the
    relation `frmony.geometry` is written in:

        V_i(x) = V_j(A.T @ (x - a)).
    """
    centred = jnp.asarray(index) - size // 2
    return centred[..., ::-1]
