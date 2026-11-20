"""Progress bars, shown when the log is.

The long steps of a run are silent otherwise: reading ten thousand
particles takes minutes, and a pass of the pairs tens of them, and a run
that prints nothing in that time looks stuck. A bar goes to standard error
when the package logs at the level of information, which is what
``--verbose`` sets, and stays out of the way otherwise, in tests and in
scripts that did not ask for it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from tqdm.auto import tqdm

__all__ = ['progress']

_LOGGER = logging.getLogger('frmony')


def progress[T](
    items: Iterable[T],
    description: str,
    total: int | None = None,
    unit: str = 'it',
    scale: int = 1,
) -> tqdm[T]:
    """Wrap an iterable in a bar, shown only if frmony logs information.

    Args:
        items: What to go through.
        description: What the bar says it is doing.
        total: How many there are, if ``items`` cannot say.
        unit: What one of them is called.
        scale: How many units each item stands for, so that a bar over
            chunks of pairs can count pairs.
    """
    bar: tqdm[T] = tqdm(
        items,
        desc=description,
        total=total,
        unit=unit,
        unit_scale=scale if scale != 1 else False,
        disable=not _LOGGER.isEnabledFor(logging.INFO),
        dynamic_ncols=True,
        leave=True,
    )
    return bar
