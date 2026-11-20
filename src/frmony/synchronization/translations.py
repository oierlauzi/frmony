"""Where each particle sits, once its orientation is known.

A pair that has been aligned says how far apart two particles are as well
as how they are turned relative to one another. With the orientations
settled, what is left is linear. From the relation `frmony.geometry` is
written in,

    V_i(x) = V_j(A_ij.T @ (x - a_ij)),    A_ij = Q_i @ Q_j.T,

substituting ``V_i(x) = S(Q_i.T @ (x - t_i))`` and matching the constant
part gives

    t_i - A_ij @ t_j = a_ij,

three equations for each pair, linear in the centres. Stacking them is an
ordinary least-squares problem, solved here with the synchronized ``A_ij``
rather than the measured one, so that the rotations the synchronization
agreed on are the ones the shifts are fitted against.

The system has a null space, and it is not the obvious one. Moving every
particle by the same amount does *not* leave the equations alone, because
each is subtracted through its own rotation. What does leave them alone is
moving the structure: taking the structure to ``S(y - d)`` sends ``t_i`` to
``t_i - Q_i @ d``, and

    (t_i - Q_i d) - A_ij @ (t_j - Q_j d) = a_ij - Q_i d + Q_i Q_j.T Q_j d

leaves the right-hand side unchanged. So the null space is three
dimensional and spanned by ``t_i = Q_i @ d``, and the condition that
removes it is ``sum_i Q_i.T @ t_i = 0``: the structure sits at the middle
of its own box. Asking instead that the centres average to zero would
appear to work -- the system comes out non-singular -- while quietly
answering a different question.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray

__all__ = ['synchronize_shifts']

_DIMENSION = 3


def synchronize_shifts(
    count: int,
    left: ArrayLike,
    right: ArrayLike,
    shifts: ArrayLike,
    orientations: ArrayLike,
    weights: ArrayLike | None = None,
    tolerance: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Centres that agree best with the measured shifts.

    Args:
        count: How many particles there are.
        left: Index of the first particle of each pair.
        right: Index of the second.
        shifts: The shift measured for each pair, ``(pairs, 3)``, ordered
            ``(x, y, z)``.
        orientations: What the rotations were synchronized to,
            ``(count, 3, 3)``.
        weights: How much each pair is trusted, ``(pairs,)``. Equal if not
            given.
        tolerance: Passed to the least-squares solver.

    Returns:
        The centres ``(count, 3)``, gauged so that ``sum_i Q_i.T @ t_i`` is
        zero, and the residual of each pair ``(pairs,)`` in the same units
        as the shifts.

    Raises:
        ValueError: If the pairs and shifts disagree in number.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)

    pairs = len(left)
    if not (len(right) == len(shifts) == pairs):
        raise ValueError('There must be one shift for each pair.')
    if shifts.shape[1:] != (_DIMENSION,):
        raise ValueError('A shift has three components.')
    if orientations.shape != (count, _DIMENSION, _DIMENSION):
        raise ValueError('The orientations are 3x3, one for each particle.')

    if weights is None:
        trust = np.ones(pairs)
    else:
        trust = np.asarray(weights, dtype=np.float64)
        if len(trust) != pairs:
            raise ValueError('There must be one weight for each pair.')
        if np.any(trust < 0.0):
            raise ValueError('A weight is not negative.')
    root = np.sqrt(trust)

    # A_ij from the synchronized orientations, not from the measurement.
    relative = orientations[left] @ np.swapaxes(orientations[right], -2, -1)

    rows = []
    columns = []
    values = []
    offset = np.arange(_DIMENSION)
    index = np.arange(pairs)

    # The identity on the first particle of each pair.
    rows.append((_DIMENSION * index[:, None] + offset[None, :]).ravel())
    columns.append((_DIMENSION * left[:, None] + offset[None, :]).ravel())
    values.append(np.repeat(root, _DIMENSION))

    # Minus A_ij on the second.
    block_rows = (
        _DIMENSION * index[:, None, None]
        + offset[None, :, None]
        + np.zeros((1, 1, _DIMENSION), dtype=np.int64)
    )
    block_columns = (
        _DIMENSION * right[:, None, None]
        + offset[None, None, :]
        + np.zeros((1, _DIMENSION, 1), dtype=np.int64)
    )
    rows.append(block_rows.ravel())
    columns.append(block_columns.ravel())
    values.append((-relative * root[:, None, None]).ravel())

    # Three more rows fixing the gauge: sum_i Q_i.T @ t_i = 0, which puts
    # the structure at the middle of its own box. See the module docstring
    # for why it is this and not the centres averaging to zero.
    particle = np.arange(count)
    gauge_rows = (
        _DIMENSION * pairs
        + offset[None, :, None]
        + np.zeros((count, 1, _DIMENSION), dtype=np.int64)
    )
    gauge_columns = (
        _DIMENSION * particle[:, None, None]
        + offset[None, None, :]
        + np.zeros((1, _DIMENSION, 1), dtype=np.int64)
    )
    rows.append(gauge_rows.ravel())
    columns.append(gauge_columns.ravel())
    values.append(np.swapaxes(orientations, -2, -1).ravel())

    design = scipy.sparse.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(_DIMENSION * (pairs + 1), _DIMENSION * count),
    ).tocsr()

    target = np.zeros(_DIMENSION * (pairs + 1))
    target[: _DIMENSION * pairs] = (shifts * root[:, None]).ravel()

    solution = scipy.sparse.linalg.lsqr(
        design, target, atol=tolerance, btol=tolerance
    )[0]
    centres = solution.reshape(count, _DIMENSION)

    predicted = centres[left] - np.einsum(
        'pij,pj->pi', relative, centres[right]
    )
    residuals = np.linalg.norm(predicted - shifts, axis=1)
    return centres, residuals
