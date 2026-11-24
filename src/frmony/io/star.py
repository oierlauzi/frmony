"""RELION's metadata, read and written.

vitrio reads the pixels and parses no metadata, so the tables are handled
here with ``starfile``. Two layouts are supported and they differ less than
their version numbers suggest.

RELION 4.0 and 5.0 write *identical* three-dimensional pseudo-subtomograms:
the same extraction code, the same pair of files, and the same two columns
naming them. What changed in 5.0 is where a particle is said to be --
``rlnCenteredCoordinate*Angst``, measured in angstroms from the middle of
the tomogram, in place of ``rlnCoordinate*`` in voxels -- together with a
good deal of tomogram metadata this package does not need. So the version
is detected from the columns rather than asked for, and only the
coordinates branch on it. They are carried through for provenance; the
alignment happens in the particle's own box and never consults them.

What is not supported is the two-dimensional particle stack, which is
RELION 5's *default*. There the contrast transfer function lives in the
images themselves and no weight volume is written at all, so a reader that
simply looked for the column would fail with a missing key. It says so
instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ['MIRRORED', 'Metadata', 'read_star', 'write_star']

MIRRORED = 'frmonyMirrored'
"""The column that marks a particle found to be a mirror image: one where
it is, written beside the angles of the rotation it is minus."""

_LOGGER = logging.getLogger(__name__)

_PARTICLES = 'particles'
_OPTICS = 'optics'

_IMAGE = 'rlnImageName'
_WEIGHT = 'rlnCtfImage'
_STACKED = 'rlnTomoSubTomosAre2DStacks'
_ANGLES = ('rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi')
_ORIGINS = ('rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst')
_CENTRED = (
    'rlnCenteredCoordinateXAngst',
    'rlnCenteredCoordinateYAngst',
    'rlnCenteredCoordinateZAngst',
)
_VOXELS = ('rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ')
_PIXEL_SIZE = (
    'rlnImagePixelSize',
    'rlnTomoTiltSeriesPixelSize',
    'rlnPixelSize',
)


@dataclass(frozen=True)
class Metadata:
    """What a particle table says, in the form this package uses."""

    particles: pd.DataFrame
    """Every row of the table, carried through so it can be written back."""
    optics: pd.DataFrame | None
    """The optics table, if the file had one."""
    data: list[str]
    """Where each particle's volume is, one per row."""
    weights: list[str] | None
    """Where each particle's Fourier weight is, if the file names them."""
    pixel_size: float
    """Of the particle box [A]."""
    version: str
    """``'relion5'`` or ``'relion4'``, decided by the columns present."""

    def __len__(self) -> int:
        return len(self.particles)


def _find_pixel_size(
    particles: pd.DataFrame, optics: pd.DataFrame | None
) -> float:
    """The pixel size of the particle box [A], from wherever it is given."""
    for table in (particles, optics):
        if table is None:
            continue
        for column in _PIXEL_SIZE:
            if column in table.columns:
                values = np.asarray(table[column], dtype=np.float64)
                if len(values) and np.all(np.isfinite(values)):
                    if len(np.unique(np.round(values, 6))) > 1:
                        _LOGGER.warning(
                            'The particles have more than one pixel size; '
                            'using the first, %.4f A.',
                            float(values[0]),
                        )
                    return float(values[0])
    raise ValueError(
        'The table gives no pixel size. Expected one of '
        + ', '.join(_PIXEL_SIZE)
        + ' in the particle or optics table.'
    )


def read_star(path: str | Path, root: str | Path | None = None) -> Metadata:
    """Read a RELION particle table.

    Args:
        path: The STAR file.
        root: Directory that relative paths in it are relative to. The
            file's own directory by default, which is where RELION writes
            them relative to the project.

    Returns:
        A `Metadata`.

    Raises:
        ValueError: If the file holds no particles, if it names no
            volumes, or if it describes two-dimensional particle stacks,
            which this package cannot use.
    """
    import starfile

    path = Path(path)
    blocks = starfile.read(path, always_dict=True)
    if _PARTICLES in blocks:
        particles = blocks[_PARTICLES]
    else:
        tables = [
            value
            for key, value in blocks.items()
            if isinstance(value, pd.DataFrame) and key != _OPTICS
        ]
        if not tables:
            raise ValueError(f'{path} holds no table of particles.')
        particles = max(tables, key=len)
    optics = blocks.get(_OPTICS)

    if _STACKED in particles.columns and np.any(
        np.asarray(particles[_STACKED], dtype=np.int64) != 0
    ):
        raise ValueError(
            f'{path} describes particles extracted as stacks of tilt '
            'images. This package aligns three-dimensional subtomograms, '
            'and a stack carries its contrast transfer function in the '
            'images rather than in a weight volume. Re-extract with 3D '
            'pseudo-subtomograms.'
        )
    if _IMAGE not in particles.columns:
        raise ValueError(f'{path} has no {_IMAGE} column naming the volumes.')

    base = Path(root) if root is not None else path.parent
    data = [_resolve(name, base) for name in particles[_IMAGE]]

    weights: list[str] | None = None
    if _WEIGHT in particles.columns:
        weights = [_resolve(name, base) for name in particles[_WEIGHT]]
    else:
        _LOGGER.info(
            '%s names no %s column, so the missing wedge will have to be '
            'given on the command line.',
            path,
            _WEIGHT,
        )

    version = 'relion5' if _CENTRED[0] in particles.columns else 'relion4'
    return Metadata(
        particles=particles,
        optics=optics,
        data=data,
        weights=weights,
        pixel_size=_find_pixel_size(particles, optics),
        version=version,
    )


def _resolve(name: Any, root: Path) -> str:
    """A path from the table, made absolute."""
    text = str(name).strip()
    # RELION writes a stack member as "index@file"; a three-dimensional
    # pseudo-subtomogram is a file of its own and has no index.
    prefix, separator, rest = text.partition('@')
    body = rest if separator else text
    resolved = Path(body)
    if not resolved.is_absolute():
        resolved = root / resolved
    return f'{prefix}@{resolved}' if separator else str(resolved)


def write_star(
    path: str | Path,
    metadata: Metadata,
    rotations: NDArray[np.float64],
    shifts: NDArray[np.float64],
    scores: NDArray[np.float64] | None = None,
) -> None:
    """Write the orientations and centres back out, beside what came in.

    Every incoming column is kept, so nothing the alignment did not touch
    is lost, and reading the result back in here round-trips.

    What has *not* been checked is the half of the convention that lives
    in RELION. The angles are this package's orientations written as ZYZ
    Euler angles and the origins are its centres in angstroms, which is
    the natural reading of those columns; whether RELION takes an
    orientation to mean this one or its inverse, and whether it adds or
    subtracts the origin, has not been tested against RELION itself,
    because that needs RELION and a real dataset. Both would be silent if
    wrong: the map would come out wrong rather than the program failing.
    Check it against a refinement of the same particles before trusting
    the output downstream.

    Args:
        path: Where to write.
        metadata: What `read_star` returned.
        rotations: Orientations ``(particles, 3, 3)``, mirrors among them
            if a search allowed them; see `MIRRORED`.
        shifts: Centres ``(particles, 3)`` in voxels, ordered
            ``(x, y, z)``.
        scores: A figure of merit for each particle, written as
            ``rlnMaxValueProbDistribution`` if given.
    """
    import starfile

    from ..geometry import matrix_to_euler

    if len(rotations) != len(metadata):
        raise ValueError('There must be one orientation for each particle.')
    if len(shifts) != len(metadata):
        raise ValueError('There must be one centre for each particle.')

    particles = metadata.particles.copy()
    # Euler angles describe rotations only. A particle found to be a
    # mirror image has an orientation of determinant minus one, minus a
    # rotation; that rotation is written, and the mirror said apart, in a
    # column of this package's own, which RELION does not know.
    mirror = np.linalg.det(rotations) < 0.0
    rot, tilt, psi = matrix_to_euler(
        np.where(mirror[:, None, None], -rotations, rotations)
    )
    for column, values in zip(_ANGLES, (rot, tilt, psi), strict=True):
        particles[column] = np.degrees(values)
    for column, axis in zip(_ORIGINS, range(3), strict=True):
        particles[column] = shifts[:, axis] * metadata.pixel_size
    if scores is not None:
        particles['rlnMaxValueProbDistribution'] = np.asarray(scores)
    if np.any(mirror):
        particles[MIRRORED] = mirror.astype(np.int64)

    blocks: dict[str, pd.DataFrame] = {}
    if metadata.optics is not None:
        blocks[_OPTICS] = metadata.optics
    blocks[_PARTICLES] = particles
    starfile.write(blocks, Path(path), overwrite=True)
