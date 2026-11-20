# Working on frmony

Read `README.md` first for what the package does. This file is about how to
change it without breaking it.

## Commands

Set up:

```bash
uv venv && uv pip install -e '.[dev]'
```

Run what CI runs, in this order:

```bash
uv run --extra dev ruff check . && uv run --extra dev ruff format --check . && uv run --extra dev mypy && uv run --extra dev pytest
```

Note the `.` on both ruff commands. `ruff format` also formats Python
inside Markdown code blocks, so running it over `src tests` alone will pass
locally and fail in CI. That has happened once already.

The full suite takes about ninety seconds. `-k "not slow"` skips the
end-to-end command line test.

## Conventions that bite

**Array order versus vector order.** Volumes are indexed `[z, y, x]`, as
the files store them. Rotations and shifts are `(x, y, z)`. The two meet in
`harmonics/shells.py` and `matching/translation.py`, and nowhere else.
Reversing in the wrong place gives a result that looks plausible and is
mirrored.

**Euler angles follow RELION.** `geometry.euler_to_matrix(rot, tilt, psi)`
is `Rz(psi) @ Ry(tilt) @ Rz(rot)`. This is not negotiable, because it is
how the output interoperates. `tests/test_geometry.py` checks it against
the composition written out by hand.

**The meaning of a pairwise alignment.** A pair gives `(A, a)` with

```
V_i(x) = V_j(A.T @ (x - a)),   A = Q_i @ Q_j.T,   a = t_i - A @ t_j.
```

Everything downstream assumes this. `geometry.py`'s module docstring is the
reference.

**Expansions carry the radial weight.** `harmonics.expand` bakes its share
of the `r**2 dr` into the coefficients, so the three-dimensional
correlation is a plain sum over shells. Anything that integrates the radial
direction itself has to divide it back out. `matching/frm5d.py` does.

**Precision is single.** JAX defaults to float32 and the package does not
change that. Tolerances around `1e-6` are the floor, not sloppiness. Also,
`arccos` near a zero angle loses half its digits, so an angular error that
is exactly zero still reads as about `1e-6` degrees.

## Traps

Each of these was a real bug here. They are all the kind that look like
something else.

**Do not re-derive the grid conventions.** Which Euler angles a correlation
grid index stands for was found by holding the grid against a direct
quadrature of the integral, not reasoned out. See
`matching.rotation.grid_to_euler`. On a grid whose values were of size
seven, the winning candidate was wrong by 2e-6 and the next best by 2.0, so
there is no room to doubt it, and `tests/test_rotation.py` holds it in
place. The precedent this package grew
out of left four open questions about this exact chain. If you think it is
wrong, write a brute force check, do not argue from the papers.

**Degenerate singular values.** In `synchronization/power_method.py`, every
block at the solution is the same rotation times one common matrix, and all
three singular values are equal. Taking the top three left singular
directions therefore returns an arbitrary basis, which often leaves a
nearly singular matrix in the way. The orientations then come out scrambled
*while the objective sits at its maximum*. `collapse` whitens by the Gram
matrix instead. Do not simplify it back.

**The shift null space is not a translation.** Moving every particle by the
same amount does not leave the equations alone, because each is subtracted
through its own rotation. The freedom is `t_i -> t_i - Q_i @ d`, so the
gauge is `sum_i Q_i.T @ t_i = 0`. Asking for the centres to average to zero
gives a non-singular system that answers a different question.

**The symmetry group is not the one the user named.** Synchronization fixes
the orientations only up to a global rotation, and a point group acts in
the frame of the particle. What is present in the estimated frame is
`U.T @ G @ U`. Snapping against `G` takes a perfect solution apart.
`synchronization/rotations.py` reads the group off the residuals, and reads
it again every round, because the global frame moves as the iteration goes
and a stale group starts rejecting the pairs that are right.

**`jnp.linalg.norm` has a NaN gradient at the origin.** The NaN then
spreads through `jnp.where` from the branch that was *not* taken. Every
call to `matching/refine.py` starts at the origin, so this is not an edge
case. Work from the squared norm and substitute a harmless value inside the
branch not taken.

**Which spheres a volume is expanded on matters enormously.** The lowest
frequencies carry a particle's mass and not its orientation. Leaving them
in took a median pairwise error from three degrees to eight and sent four
pairs in ten past thirty. `harmonics.band_radii` drops them, and says so.

**The Nyquist column.** `fftfreq(2B)` lays the orders out as
`0 .. B-1, -B, -(B-1) .. -1`. The column at index `B` is the order `-B`,
which no degree below the bandwidth reaches. Slicing a longer FFT down has
to leave that slot empty, or every negative order shifts by one.

**The two FRM papers conjugate different operands.** The 2002 paper
conjugates the second function, the 2003 paper the first. So
`matching.frm5d.two_centre_integrals` comes out as the *conjugate* of
`matching.rotation.radial_overlap` at zero distance, not equal to it.

## Testing

Tests pin properties, not golden values. A test should say what has to be
true and why, so that someone reading the failure knows whether the code or
the expectation is wrong.

Tolerances are measured, not guessed. Where a number in a test came from a
measurement, the comment says so and gives the number. Where a bound is a
dtype's floor rather than the method's accuracy, the comment says that too.

Prefer a brute force reference that shares no code with what it checks.
`tests/test_rotation.py` integrates the correlation directly over the
sphere. `tests/test_frm5d.py` leans on an exact identity, that at zero
distance the two-centre integrals collapse onto the ordinary radial
overlap.

Simulated subtomograms come from `tests/conftest.py`. Rotations there use
cubic interpolation on purpose. Linear interpolation blurs by an amount
that depends on the angle, which shows up as an error of the matching
rather than of the simulation.

## What is not verified

Do not claim otherwise in documentation or commit messages.

- The STAR output has never been checked against RELION. Whether RELION
  reads an orientation as this one or its inverse, and whether it adds or
  subtracts the origin, is untested. Both would be silent if wrong.
- Accuracy has only been measured on simulated data. It holds to a signal
  to noise ratio of about one and breaks down near a tenth of that. The
  package has not been run on real data.
- `matching/frm5d.py` has the mathematics but not the map from a grid index
  back to a pair of rotations. It is not an engine yet, and `matching` does
  not dispatch to it.
- The loader reads volumes one at a time. It does not overlap reads with
  computation.

## Style

Eighty columns, single quotes, Google style docstrings, `mypy --strict`.
Match the surrounding code.

Prose in docstrings and commit messages explains *why*, and especially what
was tried and did not work. A reader who knows what the code does still
cannot tell which of the obvious simplifications is a trap. That is what
this file and those comments are for.

Short sentences. No em-dashes.
