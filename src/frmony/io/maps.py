"""Writing a structure out as a map.

vitrio 0.2.2 binds no header metadata -- the C++ library carries a pixel
size through its write path, but nothing reaches Python -- so a map written
through it would claim a pixel size of nothing. The map is written with
``mrcfile`` instead, which does carry one, and the pixel size also goes
into the STAR file beside it so that it survives either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ['read_pixel_size', 'write_map', 'write_spectrum']


def write_map(
    path: str | Path, volume: NDArray[np.floating], pixel_size: float
) -> None:
    """Write a real-space volume, with its pixel size in the header."""
    import mrcfile

    data = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(data)
        handle.voxel_size = float(pixel_size)


def write_spectrum(
    path: str | Path,
    spectrum: NDArray[np.complexfloating],
    pixel_size: float,
) -> None:
    """Write a centred spectrum as the real volume it stands for."""
    volume = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(np.asarray(spectrum)))
    ).real
    write_map(path, volume, pixel_size)


def read_pixel_size(path: str | Path) -> float | None:
    """The pixel size a map's header gives [A], if it gives one."""
    import mrcfile

    with mrcfile.open(str(path), permissive=True, header_only=True) as handle:
        size = float(handle.voxel_size.x)
    return size if size > 0.0 else None
