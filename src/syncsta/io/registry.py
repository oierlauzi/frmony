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


_READERS: dict[str, Callable[[Path], Source]] = {}


def register(name: str, reader: Callable[[Path], Source]) -> None:
    """Add a format, or replace one."""
    _READERS[name] = reader


def available() -> tuple[str, ...]:
    """The formats that can be read."""
    return tuple(sorted(_READERS))


def _read_relion(path: Path) -> Source:
    from .star import read_star

    metadata = read_star(path)
    _LOGGER.info(
        'Read %d particles from a %s table.', len(metadata), metadata.version
    )
    return Source(
        data=metadata.data,
        weights=metadata.weights,
        pixel_size=metadata.pixel_size,
        metadata=metadata,
        format=metadata.version,
    )


def _read_mrc(path: Path) -> Source:
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]
    if not lines:
        raise ValueError(f'{path} names no volumes.')
    root = Path(path).parent
    resolved = [
        str(root / line) if not Path(line).is_absolute() else line
        for line in lines
    ]
    return Source(data=resolved, weights=None, pixel_size=None, format='mrc')


register('relion', _read_relion)
register('mrc', _read_mrc)


def read_source(path: str | Path, format: str | None = None) -> Source:
    """Find the subtomograms a path names.

    Args:
        path: A RELION particle table, or a list of volumes.
        format: Which reader to use. Chosen from the suffix if not given:
            ``.star`` is RELION and anything else is a list.

    Raises:
        ValueError: If the format is not one that is registered.
    """
    path = Path(path)
    if format is None:
        format = 'relion' if path.suffix.lower() == '.star' else 'mrc'
    if format not in _READERS:
        raise ValueError(
            f'{format!r} is not a format this knows. There is '
            + ', '.join(available())
            + '.'
        )
    return _READERS[format](path)
