"""Aligning one subtomogram to another.

The search is in two stages. A rotation is found first, on the amplitudes
of the Fourier transforms, which do not move when the particle does, so the
rotation can be found before the shift is known. The best few rotations are
kept rather than the best one, because amplitudes discard the phases and so
discriminate less than the full score does. Then each candidate gets a
single transform, which gives the correlation at every shift at once and
scores the candidate in full.

Throughout, a particle is the pair ``(D, W)``: its Fourier transform, and a
non-negative weight saying how much of each frequency was ever measured.
The weight only multiplies, so nothing is divided by a value the missing
wedge or a zero of the contrast transfer function left at nothing.
"""

from .correlation import Expansion, expand_particle, rotation_score
from .pairwise import (
    Alignment,
    Settings,
    align_pair,
    align_pairs,
    candidate_rotations,
)
from .peaks import refine_peak, top_peaks
from .refine import refine_alignment, score_at
from .rotation import (
    correlation_grid,
    grid_angles,
    grid_to_euler,
    radial_overlap,
    rotational_correlation,
)
from .translation import (
    frequency_grid,
    rotate_spectrum,
    shift_correlation,
    shift_of_index,
)

__all__ = [
    'Alignment',
    'Expansion',
    'Settings',
    'align_pair',
    'align_pairs',
    'candidate_rotations',
    'correlation_grid',
    'expand_particle',
    'frequency_grid',
    'grid_angles',
    'grid_to_euler',
    'radial_overlap',
    'refine_alignment',
    'refine_peak',
    'rotate_spectrum',
    'rotation_score',
    'rotational_correlation',
    'score_at',
    'shift_correlation',
    'shift_of_index',
    'top_peaks',
]
