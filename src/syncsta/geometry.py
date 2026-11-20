"""Orientations and the rigid motions between them.

An orientation is a 3x3 rotation matrix in the convention of RELION: its
rows are the axes of the image, x and y, and the direction of projection,
expressed in the frame of the particle. A subtomogram holds the density of
the particle as

    V(x) = S(Q.T @ (x - t)),

for the structure ``S``, the orientation ``Q`` and the centre ``t`` of the
particle. Composing two of those gives the relation this package solves for:
a pair aligned as ``V_i(x) = V_j(A.T @ (x - a))`` has

    A = Q_i @ Q_j.T        and        a = t_i - A @ t_j.

Everything here is NumPy in double precision. These are per-particle
quantities, thousands of them rather than millions, and the cost of getting
them wrong is far greater than the cost of computing them exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def euler_to_matrix(
    rot: ArrayLike, tilt: ArrayLike, psi: ArrayLike
) -> NDArray[np.float64]:
    """Rotation matrices of ZYZ Euler angles [rad], as RELION defines them.

    The matrix is ``Rz(psi) @ Ry(tilt) @ Rz(rot)``.

    Args:
        rot: First Euler angle, about z [rad].
        tilt: Second Euler angle, about the new y [rad].
        psi: Third Euler angle, about the new z [rad].

    Returns:
        Matrices of shape ``(..., 3, 3)``, the angles broadcast together.
    """
    rot, tilt, psi = np.broadcast_arrays(
        *(np.asarray(angle, dtype=np.float64) for angle in (rot, tilt, psi))
    )
    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)

    out = np.empty((*rot.shape, 3, 3))
    out[..., 0, 0] = cg * cb * ca - sg * sa
    out[..., 0, 1] = cg * cb * sa + sg * ca
    out[..., 0, 2] = -cg * sb
    out[..., 1, 0] = -sg * cb * ca - cg * sa
    out[..., 1, 1] = -sg * cb * sa + cg * ca
    out[..., 1, 2] = sg * sb
    out[..., 2, 0] = sb * ca
    out[..., 2, 1] = sb * sa
    out[..., 2, 2] = cb
    return out


def _wrap(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Angles [rad] folded into (-pi, pi]."""
    wrapped: NDArray[np.float64] = -np.mod(-angle + np.pi, 2.0 * np.pi) + np.pi
    return wrapped


def matrix_to_euler(
    matrix: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ZYZ Euler angles [rad] of rotation matrices, as RELION defines them.

    The inverse of `euler_to_matrix`, up to the degeneracy at ``tilt`` of 0
    and pi, where only ``psi + rot`` and ``psi - rot`` are determined. There
    the whole turn is given to ``psi``, which is what RELION does.

    Args:
        matrix: Rotation matrices, ``(..., 3, 3)``.

    Returns:
        The angles ``rot``, ``tilt`` and ``psi``, each of shape ``(...)``.
        ``tilt`` lies in ``[0, pi]`` and the others in ``(-pi, pi]``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError('An orientation is a 3x3 matrix.')

    # sin(tilt) >= 0 over [0, pi], so the norm of the third column's first
    # two entries gives it without a sign to resolve.
    sb = np.hypot(matrix[..., 0, 2], matrix[..., 1, 2])
    tilt = np.arctan2(sb, matrix[..., 2, 2])

    degenerate = sb < 1e-8
    # Away from the poles, read rot and psi off the last row and column.
    rot = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 0])
    psi = np.arctan2(matrix[..., 1, 2], -matrix[..., 0, 2])
    # At a pole the two angles merge; give the turn to psi, as RELION does.
    # At tilt 0 the matrix determines psi + rot; at tilt pi it determines
    # psi - rot, and through a reflected angle. Giving the whole turn to psi
    # and leaving rot at zero is what RELION does.
    merged = np.arctan2(matrix[..., 0, 1], matrix[..., 0, 0])
    at_pi = _wrap(np.pi - merged)
    rot = np.where(degenerate, 0.0, rot)
    psi = np.where(
        degenerate, np.where(matrix[..., 2, 2] > 0, merged, at_pi), psi
    )
    return rot, tilt, psi


def rotation_angle(matrix: ArrayLike) -> NDArray[np.float64]:
    """How far rotation matrices turn [rad], in ``[0, pi]``.

    This is the geodesic distance from the identity, and it is invariant
    under conjugation: ``rotation_angle(P @ R @ P.T) == rotation_angle(R)``.
    That invariance is what lets the symmetry of the particles be detected
    without knowing a single orientation.

    Args:
        matrix: Rotation matrices, ``(..., 3, 3)``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    angle: NDArray[np.float64] = np.arccos(
        np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    )
    return angle


def relative_angle(left: ArrayLike, right: ArrayLike) -> NDArray[np.float64]:
    """Angle [rad] of the rotation that takes one orientation to another."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return rotation_angle(np.matmul(np.swapaxes(left, -2, -1), right))


def project_to_rotation(matrix: ArrayLike) -> NDArray[np.float64]:
    """The closest rotation matrices, in the sense of least squares.

    The polar factor of each matrix, with its determinant forced positive so
    that the result is a rotation and not a reflection. Leaving the
    determinant alone would give the closest *orthogonal* matrix, which is a
    different thing and the wrong one for an orientation.

    Args:
        matrix: Matrices of shape ``(..., 3, 3)``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(matrix)
    # Flipping the smallest singular direction is the cheapest way back into
    # SO(3), and the singular values are sorted, so it is the last one.
    sign = np.sign(np.linalg.det(u) * np.linalg.det(vh))
    u = u.copy()
    u[..., :, -1] *= sign[..., None]
    out: NDArray[np.float64] = u @ vh
    return out


def exponential_map(vector: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrices of rotation vectors, by Rodrigues' formula.

    The direction of the vector is the axis and its length the angle [rad].

    Args:
        vector: Rotation vectors, ``(..., 3)``.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[-1] != 3:
        raise ValueError('A rotation vector has three components.')
    angle = np.linalg.norm(vector, axis=-1)[..., None]
    # The series of sin(t)/t and (1-cos(t))/t^2 near zero, to keep the
    # derivative finite where the naive quotients are 0/0.
    small = angle < 1e-6
    sinc = np.where(
        small, 1.0 - angle**2 / 6.0, np.sin(angle) / np.where(small, 1.0, angle)
    )
    versine = np.where(
        small,
        0.5 - angle**2 / 24.0,
        (1.0 - np.cos(angle)) / np.where(small, 1.0, angle**2),
    )

    cross = np.zeros((*vector.shape[:-1], 3, 3))
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    cross[..., 0, 1], cross[..., 0, 2] = -z, y
    cross[..., 1, 0], cross[..., 1, 2] = z, -x
    cross[..., 2, 0], cross[..., 2, 1] = -y, x

    eye = np.eye(3)
    out: NDArray[np.float64] = (
        eye + sinc[..., None] * cross + versine[..., None] * (cross @ cross)
    )
    return out


def logarithmic_map(matrix: ArrayLike) -> NDArray[np.float64]:
    """Rotation vectors of rotation matrices, the inverse of the exponential.

    Args:
        matrix: Rotation matrices, ``(..., 3, 3)``.

    Returns:
        Vectors ``(..., 3)`` whose length is at most pi.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    angle = rotation_angle(matrix)
    skew = 0.5 * (matrix - np.swapaxes(matrix, -2, -1))
    axis = np.stack(
        (skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), axis=-1
    )

    # axis has length sin(angle), so the scale is angle/sin(angle): one near
    # zero, and divergent at pi where the antisymmetric part vanishes.
    sine = np.sin(angle)
    scale = np.where(
        angle < 1e-6,
        1.0 + angle**2 / 6.0,
        angle / np.where(np.abs(sine) < 1e-12, 1.0, sine),
    )
    out = scale[..., None] * axis

    # Near pi the antisymmetric part vanishes, so read the axis off the
    # symmetric part instead: R + R.T = 2 cos(a) I + 2 (1 - cos(a)) n n.T,
    # whose diagonal is n_k squared. Signs come from the row of the largest
    # component, which is the best conditioned one.
    halfturn = angle > np.pi - 1e-4
    if np.any(halfturn):
        symmetric = 0.5 * (matrix + np.swapaxes(matrix, -2, -1))
        cosine = np.cos(angle)[..., None, None]
        outer = (symmetric - cosine * np.eye(3)) / np.maximum(
            1.0 - cosine, 1e-12
        )
        squared = np.clip(np.diagonal(outer, axis1=-2, axis2=-1), 0.0, None)
        leading = np.argmax(squared, axis=-1)
        pivot = np.take_along_axis(
            outer, leading[..., None, None], axis=-2
        ).squeeze(-2)
        scale = np.sqrt(
            np.take_along_axis(squared, leading[..., None], axis=-1)
        )
        axis_near = pivot / np.maximum(scale, 1e-12)
        axis_near /= np.maximum(
            np.linalg.norm(axis_near, axis=-1, keepdims=True), 1e-12
        )
        out = np.where(halfturn[..., None], angle[..., None] * axis_near, out)
    vector_out: NDArray[np.float64] = out
    return vector_out


def random_rotations(
    rng: np.random.Generator, count: int
) -> NDArray[np.float64]:
    """Rotations drawn uniformly over SO(3), ``(count, 3, 3)``.

    By the polar factor of a Gaussian matrix, which is uniform on SO(3)
    because the Gaussian is invariant under rotation on both sides.
    """
    rotations: NDArray[np.float64] = project_to_rotation(
        rng.normal(size=(count, 3, 3))
    )
    return rotations
