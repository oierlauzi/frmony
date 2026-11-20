"""Finding the orientations that agree best with every measurement at once.

Each aligned pair says what the relative orientation of two particles is,
noisily and sometimes wrongly. No set of orientations satisfies all of them,
so the one to want is the set that satisfies them best:

    maximize  sum over pairs of  w_ij * trace(Q_i.T @ A_ij @ Q_j)

over orientations. Written with the measurements gathered into one symmetric
block matrix ``S``, that is ``maximize trace(X.T @ S @ X)``, which is a
synchronization over the orthogonal group.

It is solved by the generalized power method: multiply by ``S``, project
each block back onto the group, repeat. The objective rises every step, so
the iteration says when to stop by itself.

Two details matter. The blocks are lifted to more columns than rows, the
relaxation of Burer and Monteiro, which gives the iteration room to leave
the local maxima that square blocks get stuck in; the answer collapses back
to three columns at the end, and a test checks that it does. And the
projection forces the determinant positive. The closest *orthogonal* matrix
to a block is not the closest *rotation*: taking it would let half the
particles come out mirrored, which no amount of further iteration would
notice.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from numpy.typing import ArrayLike, NDArray

from ..geometry import project_to_rotation

__all__ = ['collapse', 'polar_rows', 'synchronize']

_DIMENSION = 3


def polar_rows(blocks: ArrayLike) -> NDArray[np.float64]:
    """The closest matrices with orthonormal rows, ``(n, rows, columns)``.

    No determinant is forced here: with more columns than rows there is no
    determinant to force. `collapse` deals with it once the lift is undone.
    """
    stacked = np.asarray(blocks, dtype=np.float64)
    u, _, vh = np.linalg.svd(stacked, full_matrices=False)
    projected: NDArray[np.float64] = u @ vh
    return projected


def _block_matrix(
    count: int,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    measurements: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> scipy.sparse.csr_matrix:
    """The symmetric block matrix the measurements make.

    Block ``(i, j)`` holds ``w_ij * A_ij`` and block ``(j, i)`` its
    transpose, which is what makes the matrix symmetric and the objective
    real.
    """
    scaled = measurements * weights[:, None, None]
    rows = []
    columns = []
    values = []
    for first, second, block in (
        (left, right, scaled),
        (right, left, np.swapaxes(scaled, -2, -1)),
    ):
        offset = np.arange(_DIMENSION)
        row = (
            _DIMENSION * first[:, None, None] + offset[None, :, None]
        ) * np.ones((1, 1, _DIMENSION), dtype=np.int64)
        column = (
            _DIMENSION * second[:, None, None] + offset[None, None, :]
        ) * np.ones((1, _DIMENSION, 1), dtype=np.int64)
        rows.append(row.ravel())
        columns.append(column.ravel())
        values.append(block.ravel())

    size = _DIMENSION * count
    matrix = scipy.sparse.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(size, size),
    )
    return matrix.tocsr()


def synchronize(
    count: int,
    left: ArrayLike,
    right: ArrayLike,
    measurements: ArrayLike,
    weights: ArrayLike,
    start: ArrayLike | None = None,
    rank: int = 2 * _DIMENSION + 1,
    max_iterations: int = 512,
    tolerance: float = 1e-9,
) -> tuple[NDArray[np.float64], int]:
    """Orientations that agree best with the measurements.

    Args:
        count: How many particles there are.
        left: Index of the first particle of each pair.
        right: Index of the second.
        measurements: Relative orientations ``(pairs, 3, 3)``, each about
            ``Q_i @ Q_j.T``.
        weights: How much each measurement is trusted, ``(pairs,)``,
            non-negative. They need no particular scale.
        start: Orientations to begin from, ``(count, 3, 3)``. A spanning
            tree gives a far better start than chance; see
            `frmony.synchronization.rotations`.
        rank: Columns to lift the blocks to. The default, one more than
            twice the dimension, is the relaxation of Burer and Monteiro.
        max_iterations: How many steps at most.
        tolerance: Stop when the objective gains less than this, relative.

    Returns:
        The orientations ``(count, 3, 3)`` and how many steps were taken.
        A particle no pair mentions keeps the orientation it started with.

    Raises:
        ValueError: If the pairs and measurements disagree in number, if a
            weight is negative, or if the rank is below the dimension.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    measurements = np.asarray(measurements, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if not (len(left) == len(right) == len(measurements) == len(weights)):
        raise ValueError(
            'There must be one measurement and one weight for each pair.'
        )
    if measurements.shape[1:] != (_DIMENSION, _DIMENSION):
        raise ValueError('A relative orientation is a 3x3 matrix.')
    if np.any(weights < 0.0):
        raise ValueError('A weight is not negative.')
    if rank < _DIMENSION:
        raise ValueError(
            f'A rank of {rank} has no room for {_DIMENSION} dimensions.'
        )

    if start is None:
        generator = np.random.default_rng(0)
        current = polar_rows(generator.normal(size=(count, _DIMENSION, rank)))
    else:
        begun = np.asarray(start, dtype=np.float64)
        if begun.shape != (count, _DIMENSION, _DIMENSION):
            raise ValueError('The orientations to start from are 3x3.')
        current = np.zeros((count, _DIMENSION, rank))
        current[:, :, :_DIMENSION] = begun

    if len(left) == 0:
        return collapse(current), 0

    matrix = _block_matrix(count, left, right, measurements, weights)
    flat = current.reshape(_DIMENSION * count, rank)

    previous = -np.inf
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        gathered = matrix @ flat
        # The objective, which rises every step and so says when to stop.
        objective = float(np.sum(flat * gathered))
        if objective - previous < tolerance * max(abs(objective), 1.0):
            break
        previous = objective
        flat = polar_rows(gathered.reshape(count, _DIMENSION, rank)).reshape(
            _DIMENSION * count, rank
        )

    return collapse(flat.reshape(count, _DIMENSION, rank)), iterations


def collapse(lifted: ArrayLike) -> NDArray[np.float64]:
    """Bring a lifted solution back to rotations.

    The lift leaves the answer spread over more columns than it needs, and
    once the iteration has converged every block is the same rotation times
    one common matrix: ``X_i = Q_i @ P``. Recovering ``Q_i`` means undoing
    ``P``, and how that is done matters more than it looks.

    Taking the three strongest left singular directions of the stacked
    blocks is the obvious way and it is a trap. All three singular values
    are equal, so the basis that comes back is arbitrary, and an arbitrary
    basis leaves ``X_i`` as ``Q_i`` times a matrix that is often nearly
    singular -- a determinant of two thousandths, in the case that found
    this. Projecting that onto a rotation is then ill-conditioned, and the
    orientations come out scrambled while the objective sits at its
    maximum, which is a confusing way to be wrong.

    So the Gram matrix is used instead. It is small, its leading
    eigenvectors are whitened by their own eigenvalues, and what multiplies
    ``Q_i`` afterwards is orthogonal by construction rather than by luck.

    Args:
        lifted: ``(count, 3, rank)``.

    Returns:
        Rotations ``(count, 3, 3)``.
    """
    stacked = np.asarray(lifted, dtype=np.float64)
    count = len(stacked)
    gram = np.einsum('nir,nis->rs', stacked, stacked) / max(count, 1)
    values, vectors = np.linalg.eigh(gram)

    leading = vectors[:, -_DIMENSION:]
    scale = 1.0 / np.sqrt(np.maximum(values[-_DIMENSION:], 1e-30))
    whitened = stacked @ (leading * scale)

    # Whitening is defined up to a reflection, and a reflected answer is a
    # mirror image of the structure. The measurements are rotations, so the
    # blocks all share one determinant; if it came out negative, turn one
    # direction around.
    if np.median(np.linalg.det(whitened)) < 0.0:
        whitened[:, :, -1] *= -1.0
    return project_to_rotation(whitened)
