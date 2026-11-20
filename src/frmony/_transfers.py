"""Moving data between the host and the device, on purpose.

JAX will copy an array across on its own whenever one is needed on the
other side. That is convenient and it hides the two things worth watching
in a run of any size: how much is crossing, and how often.

Both of the quiet ways are easy to write by accident. ``jnp.asarray`` on a
NumPy array copies it to the device. ``np.asarray`` on a device array
copies it back. Neither looks like a transfer, and a loop that does one per
particle will sit idle behind the copies while a device waits.

So every crossing in this package goes through `to_device` or `to_host`,
which say what they do, and `no_implicit_transfers` turns anything else
into an error.

One thing to know before reading the counts. A constant that a traced
function closes over, such as the quarter-turn table or the grid of
frequencies, is folded into the compiled program and uploaded by the
runtime rather than by a ``device_put``, so it is reported as an implicit
copy however it was written. That happens once per compiled function and
not once per pair, so the number to watch is whether it grows with the
work, not whether it is zero. It will not be zero.

The guard distinguishes the two kinds: an explicit ``device_put`` or
``device_get`` is allowed, and only the quiet kind raises.
`tests/test_transfers.py` counts both over a run and over twice the
particles, so a copy that is made once per particle fails a test rather
than quietly slowing a run.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike, NDArray

__all__ = [
    'Transfer',
    'log_implicit_transfers',
    'no_implicit_transfers',
    'on_device',
    'record_transfers',
    'to_device',
    'to_host',
]


def on_device(value: Any) -> bool:
    """Whether a value already lives on a device."""
    return isinstance(value, jax.Array) and not isinstance(value, np.ndarray)


def to_device(
    value: Any,
    dtype: DTypeLike | None = None,
    device: jax.Device | None = None,
) -> jax.Array:
    """Put host data on a device, saying so.

    A value that is already there is left where it is, and only cast if a
    dtype asks for it. Casting on the host first would copy it back and
    forth for nothing.

    Args:
        value: Anything array-like, on the host or already on a device.
        dtype: What to store it as. Its own type if not given.
        device: Where to put it. The default device if not given.

    Returns:
        The value, on the device.
    """
    if on_device(value):
        array: jax.Array = value
        if dtype is not None:
            wanted = jnp.dtype(dtype)
            if array.dtype != wanted:
                array = array.astype(wanted)
        return array if device is None else jax.device_put(array, device)
    # np.asarray keeps this on the host, so the one copy is the device_put.
    host = np.asarray(value, dtype=dtype)
    moved: jax.Array = jax.device_put(host, device)
    return moved


def to_host(value: Any, dtype: DTypeLike | None = None) -> NDArray[Any]:
    """Bring device data back to the host, saying so.

    Args:
        value: A device array, or anything already on the host.
        dtype: What to return it as. Its own type if not given.

    Returns:
        A NumPy array.
    """
    fetched = jax.device_get(value) if on_device(value) else value
    return np.asarray(fetched, dtype=dtype)


@contextlib.contextmanager
def no_implicit_transfers() -> Generator[None]:
    """Make a quiet copy between host and device an error.

    Explicit ones, through `to_device` and `to_host`, still work. Only the
    copies nobody asked for raise.

    Wrap a run in this when a device looks busier than the work should
    make it. The traceback points at the line that copies.
    """
    with jax.transfer_guard('disallow'):
        yield


@contextlib.contextmanager
def log_implicit_transfers() -> Generator[None]:
    """Report quiet copies instead of refusing them.

    For finding out how many there are before deciding what to do about
    them.
    """
    with jax.transfer_guard('log'):
        yield


@dataclass(frozen=True)
class Transfer:
    """One copy between the host and a device."""

    direction: str
    """``'host-to-device'`` or ``'device-to-host'``."""
    dtype: str
    """What was copied."""
    shape: tuple[int, ...]
    """How much of it. Empty for a scalar."""

    @property
    def count(self) -> int:
        """How many values crossed."""
        total = 1
        for extent in self.shape:
            total *= extent
        return total

    @property
    def nbytes(self) -> int:
        """Roughly how many bytes crossed."""
        return self.count * int(np.dtype(self.dtype).itemsize)


_LOGGED = re.compile(
    r'(host-to-device|device-to-host) transfer: '
    r'aval=ShapedArray\(([a-z0-9_]+)\[([^\]]*)\]'
)


@contextlib.contextmanager
def record_transfers(
    kind: str = 'implicit',
    smallest: int = 64,
) -> Generator[list[Transfer]]:
    """Collect the copies of one kind that happen inside the block.

    JAX separates the two kinds and will report one or the other, never
    both at once, so this does too. They answer different questions.
    ``'implicit'`` says whether anything is crossing by accident, and the
    answer for this package should be nothing of any size. ``'explicit'``
    says how much deliberate traffic there is, and the answer should not
    grow with the number of particles.

    A strict guard is too blunt to assert either, because every Python
    number used in an expression counts as an implicit copy to the device:
    ``x * 2`` trips it. Counting, with the scalars filtered out, is what a
    test can actually say something about.

    JAX reports copies from C++, on the standard error stream rather than
    through logging, so this redirects that stream for the length of the
    block. Anything else written there is swallowed, and it does not nest.
    It is a diagnostic, not something to leave switched on.

    Args:
        kind: ``'implicit'`` for the copies nobody asked for, or
            ``'explicit'`` for the ones through `to_device` and `to_host`.
        smallest: Ignore copies of fewer values than this, which is what
            keeps the scalars out.

    Yields:
        A list, filled in once the block ends.

    Raises:
        ValueError: If the kind is neither of the two.
    """
    modes = {'implicit': 'log', 'explicit': 'log_explicit'}
    if kind not in modes:
        raise ValueError(
            f"A kind of copy is 'implicit' or 'explicit', not {kind!r}."
        )
    found: list[Transfer] = []
    with tempfile.TemporaryFile(mode='w+') as sink:
        saved = os.dup(2)
        os.dup2(sink.fileno(), 2)
        try:
            with jax.transfer_guard(modes[kind]):
                yield found
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            sink.seek(0)
            for line in sink:
                found_line = _LOGGED.search(line)
                if found_line is None:
                    continue
                shape = tuple(
                    int(extent)
                    for extent in found_line.group(3).split(',')
                    if extent.strip()
                )
                crossing = Transfer(
                    direction=found_line.group(1),
                    dtype=found_line.group(2),
                    shape=shape,
                )
                if crossing.count >= smallest:
                    found.append(crossing)
