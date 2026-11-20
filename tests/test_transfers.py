"""Copies between the host and the device, which should be few and named."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import simulate
from frmony._transfers import (
    no_implicit_transfers,
    on_device,
    record_transfers,
    to_device,
    to_host,
)
from frmony.particles import Subtomograms
from frmony.pipeline import Design, run


def test_to_device_and_back(rng: np.random.Generator) -> None:
    values = rng.normal(size=(4, 5))
    moved = to_device(values, dtype=np.float32)

    assert on_device(moved)
    assert moved.dtype == jnp.float32
    np.testing.assert_allclose(to_host(moved), values, atol=1e-6)


def test_moving_what_is_already_there_is_free() -> None:
    """Casting on the host first would send it back and forth."""
    already = to_device(np.zeros(4, dtype=np.float32))
    assert to_device(already) is already
    assert to_device(already, dtype=np.float32) is already
    # A real cast still happens, and still on the device.
    recast = to_device(already, dtype=np.complex64)
    assert on_device(recast)
    assert recast.dtype == jnp.complex64


def test_to_host_passes_host_data_through(rng: np.random.Generator) -> None:
    values = rng.normal(size=3)
    np.testing.assert_array_equal(to_host(values), values)


def test_the_named_moves_are_allowed_under_the_guard() -> None:
    """The guard refuses the quiet copies, not the deliberate ones."""
    values = np.arange(8.0, dtype=np.float32)
    with no_implicit_transfers():
        moved = to_device(values)
        back = to_host(moved)
    np.testing.assert_allclose(back, values)


def test_the_guard_refuses_a_quiet_copy() -> None:
    values = np.arange(8.0, dtype=np.float32)
    with (
        pytest.raises(Exception, match=r'[Dd]isallowed'),
        no_implicit_transfers(),
    ):
        jax.block_until_ready(jnp.asarray(values) + 0.0)


def test_recording_ignores_scalars() -> None:
    """Every Python number in an expression counts as a copy, which is
    why a strict guard cannot be used to assert anything about a run."""
    big = np.arange(4096.0, dtype=np.float32)
    with record_transfers('implicit', smallest=512) as log:
        jax.block_until_ready(jnp.asarray(big) * 2)

    assert len(log) == 1
    assert log[0].shape == (4096,)
    assert log[0].nbytes == 4096 * 4
    assert log[0].direction == 'host-to-device'


def test_the_two_kinds_are_told_apart() -> None:
    """JAX reports one kind or the other, never both at once."""
    big = np.arange(4096.0, dtype=np.float32)

    with record_transfers('implicit', smallest=512) as quiet:
        jax.block_until_ready(to_device(big) * 2)
    assert quiet == []

    with record_transfers('explicit', smallest=512) as named:
        jax.block_until_ready(to_device(big) * 2)
    assert len(named) == 1


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match='implicit'), record_transfers('both'):
        pass


def test_recording_leaves_the_error_stream_alone(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """It redirects the stream to read what JAX writes there, and has to
    give it back."""
    import sys

    with record_transfers():
        pass
    print('still here', file=sys.stderr)
    assert 'still here' in capfd.readouterr().err


def _one_run(rng: np.random.Generator, count: int, box: int, kind: str) -> list:
    """Run the pipeline twice and record the copies of the second.

    The first run compiles, and a constant that a traced function closes
    over uploads when it does. That is real, and it happens once, so
    counting it would make the answer depend on what else ran first.
    """
    simulation = simulate(rng, count, size=box, tilt=60.0)
    subtomograms = Subtomograms(
        data=simulation.data,
        weight=simulation.weight,
        pixel_size=4.0,
        box=box,
    )
    design = Design(
        partners=3, bandwidth=8, candidates=4, max_shift=2.0, detect=False
    )
    run(subtomograms, design)
    with record_transfers(kind, smallest=512) as log:
        run(subtomograms, design)
    return log


def test_the_particles_cross_once_as_a_set(
    rng: np.random.Generator,
) -> None:
    """The property that matters, and the one that would regress first.

    Expanding the particles in a Python loop, which is what this used to
    do, sent each one to the device on its own, brought its coefficients
    back, and sent those over again. The copies then grew with the number
    of particles.

    Asserted on the shapes rather than on a count. A copy the shape of one
    particle means the loop is back, whatever the count happens to be, and
    a count is noisy because compiling for a new number of particles
    uploads the constants again.
    """
    count, box = 10, 24
    log = _one_run(rng, count, box, 'explicit')
    shapes = {entry.shape for entry in log}

    assert (count, box, box, box) in shapes, 'the set did not cross whole'
    assert (box, box, box) not in shapes, 'a particle crossed on its own'


def test_nothing_crosses_by_accident(rng: np.random.Generator) -> None:
    """Every copy of any size should be one that was asked for."""
    log = _one_run(rng, 10, 24, 'implicit')
    assert [entry.shape for entry in log] == []
