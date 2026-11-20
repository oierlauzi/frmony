"""Reading subtomograms and their metadata, and writing the answer back.

vitrio reads the pixels; it parses no metadata, so the tables are handled
here with ``starfile``. Which files hold what is a matter of convention,
and conventions are registered rather than hard-wired.
"""

from .maps import write_map, write_spectrum
from .registry import Source, available, read_source, register
from .star import Metadata, read_star, write_star
from .volumes import crop_spectrum, prepare, soft_ball
from .wedge import cone_weight, half_to_centred, relion_weight, wedge_weight

__all__ = [
    'Metadata',
    'Source',
    'available',
    'cone_weight',
    'crop_spectrum',
    'half_to_centred',
    'prepare',
    'read_source',
    'read_star',
    'register',
    'relion_weight',
    'soft_ball',
    'wedge_weight',
    'write_map',
    'write_spectrum',
    'write_star',
]
