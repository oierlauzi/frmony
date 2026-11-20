"""Spherical harmonics, and the expansion of a volume into them.

A rotation moves each sphere around the centre of a volume onto itself, so
the volume's restriction to those spheres is where a rotational search
belongs. This subpackage reads a volume off concentric spheres and expands
each one into spherical harmonics, exactly, with a Gauss-Legendre quadrature
in the colatitude and an FFT around the azimuth.

It also builds the quarter-turn Wigner matrices, which are what let the
second Euler angle become an exponential like the other two, and so the
rotational correlation become one inverse FFT.
"""

from .legendre import gauss_legendre, legendre_table, orders
from .shells import sample_shells, shell_directions, shell_grid
from .transform import Basis, expand, expand_shells, make_basis
from .wigner import half_turn_table, wigner_d

__all__ = [
    'Basis',
    'expand',
    'expand_shells',
    'gauss_legendre',
    'half_turn_table',
    'legendre_table',
    'make_basis',
    'orders',
    'sample_shells',
    'shell_directions',
    'shell_grid',
    'wigner_d',
]
