"""How a set of subtomograms is named on disk.

Which files hold the particles, and which hold their Fourier weights, is a
matter of convention, and the conventions change faster than the
mathematics does. A reader is a function that turns a path into the lists
of files plus the pixel size, and new ones are registered rather than added
to a chain of if-statements.

Two come built in. ``relion`` reads a RELION particle table, four or five,
telling them apart by their columns rather than by being told. ``mrc``
takes a plain list of volumes, one path a line, with the missing region
named separately.

Both take a root, the directory that the paths inside are relative to.
RELION writes them relative to the project, which is not always where the
table itself ended up, so a table that has been moved or copied needs to
be told where its images stayed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ['Source', 'available', 'read_source', 'register']

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """Where a set of subtomograms lives."""

    data: list[str]
    """Where each particle's volume is."""
    weights: list[str] | None
    """Where each particle's Fourier weight is, if the format has them."""
    pixel_size: float | None
    """Of the box [A], if the format says."""
    metadata: object = None
    """Whatever the reader wants to carry through to the writer."""
    format: str = ''
    """Which reader produced this."""
    groups: list[str] | None = None
    """Which acquisition each particle came from, if the format says: a
    tomogram, for RELION. Particles of one acquisition share a noise
    spectrum, which is estimated from them together."""


Reader = Callable[[Path, Path | None], Source]

_READERS: dict[str, Reader] = {}


def register(name: str, reader: Reader) -> None:
    """Add a format, or replace one.

    A reader takes the path of the table and the root that the paths
    inside it are relative to, which is None when the caller did not say.
    """
    _READERS[name] = reader


def available() -> tuple[str, ...]:
    """The formats that can be read."""
    return tuple(sorted(_READERS))


def _read_relion(path: Path, root: Path | None) -> Source:
    from .star import read_star

    metadata = read_star(path, root)
    _LOGGER.info(
        'Read %d particles from a %s table.', len(metadata), metadata.version
    )
    return Source(
        data=metadata.data,
        weights=metadata.weights,
        pixel_size=metadata.pixel_size,
        metadata=metadata,
        format=metadata.version,
        groups=(
            [str(name) for name in metadata.particles['rlnTomoName']]
            if 'rlnTomoName' in metadata.particles.columns
            else None
        ),
    )


def _read_mrc(path: Path, root: Path | None) -> Source:
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]
    if not lines:
        raise ValueError(f'{path} names no volumes.')
    base = Path(root) if root is not None else Path(path).parent
    resolved = [
        str(base / line) if not Path(line).is_absolute() else line
        for line in lines
    ]
    return Source(data=resolved, weights=None, pixel_size=None, format='mrc')


register('relion', _read_relion)
register('mrc', _read_mrc)


def read_source(
    path: str | Path,
    format: str | None = None,
    root: str | Path | None = None,
) -> Source:
    """Find the subtomograms a path names.

    Args:
        path: A RELION particle table, or a list of volumes.
        format: Which reader to use. Chosen from the suffix if not given:
            ``.star`` is RELION and anything else is a list.
        root: Directory that the paths inside are relative to. The table's
            own directory if not given, which is right when the table is
            still where it was written and wrong as soon as it is copied
            somewhere else.

    Raises:
        ValueError: If the format is not one that is registered, or if the
            root is not a directory.
    """
    path = Path(path)
    if root is not None and not Path(root).is_dir():
        raise ValueError(f'{root} is not a directory to find images under.')
    if format is None:
        format = 'relion' if path.suffix.lower() == '.star' else 'mrc'
    if format not in _READERS:
        raise ValueError(
            f'{format!r} is not a format this knows. There is '
            + ', '.join(available())
            + '.'
        )
    return _READERS[format](path, Path(root) if root is not None else None)
