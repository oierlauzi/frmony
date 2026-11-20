"""From pairwise measurements to one consistent set of orientations.

Each aligned pair says how two particles are turned relative to one
another, noisily and sometimes wrongly. No set of orientations satisfies
every measurement, so what is wanted is the set that satisfies them best,
which is a synchronization over the orthogonal group. The shifts follow
afterwards from a linear least-squares problem.

Symmetry is handled here too, because it is a property of the measurements
rather than of any one of them: a particle with a point group has an
orientation defined only up to that group, so each alignment lands on an
arbitrary member of a coset. `frmony.symmetry` explains what that does and
why it cannot be undone one pair at a time.
"""

from .power_method import collapse, polar_rows, synchronize
from .rotations import (
    Synchronized,
    propagate_along_tree,
    synchronize_rotations,
)
from .translations import synchronize_shifts

__all__ = [
    'Synchronized',
    'collapse',
    'polar_rows',
    'propagate_along_tree',
    'synchronize',
    'synchronize_rotations',
    'synchronize_shifts',
]
