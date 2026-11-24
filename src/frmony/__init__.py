"""Ab-initio subtomogram averaging for cryo-electron tomography.

Subtomograms are aligned to each other in pairs, each pair giving a noisy
relative rotation and shift. Orthogonal group synchronization then finds the
one set of orientations that agrees best with all of those measurements at
once, and the shifts follow from a linear least-squares problem. No initial
reference is needed, so the result does not carry a reference's bias. When
there is one, `align_to_reference` aligns each particle to it instead.

The missing wedge, and the contrast transfer function of a pseudo-subtomogram,
are carried as a non-negative weight in Fourier space. That weight only ever
multiplies, so nothing is divided by a value that the microscope left at zero.
"""

from __future__ import annotations

from .averaging import average, fourier_shell_correlation, resolution_at
from .engine import Search
from .geometry import euler_to_matrix, matrix_to_euler
from .graph import build_pairs
from .particles import Subtomograms, load_reference, load_subtomograms
from .pipeline import Design, Result, run
from .reference import align_to_reference
from .symmetry import detect_symmetry, point_group

__all__ = [
    'Design',
    'Result',
    'Search',
    'Subtomograms',
    '__version__',
    'align_to_reference',
    'average',
    'build_pairs',
    'detect_symmetry',
    'euler_to_matrix',
    'fourier_shell_correlation',
    'load_reference',
    'load_subtomograms',
    'matrix_to_euler',
    'point_group',
    'resolution_at',
    'run',
]

__version__ = '0.1.0'
