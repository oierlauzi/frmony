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

**Every copy between host and device is named.** `_transfers.to_device`
and `to_host` are the only two places data crosses. Writing
`jnp.asarray` on a NumPy array, or `np.asarray` on a device array, does it
quietly instead, and `tests/test_transfers.py` fails when something does.
Reach for `record_transfers` when a device looks busier than the work
should make it.

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

**Amplitudes are not enough under noise.** The first stage on `|D|` put
the right rotation at the sixty-third percentile of its grid on real
subtomograms, where chance is the fiftieth. Centring the amplitudes,
standardising them shell by shell and using the power instead all left it
there. The reason is the order of the signal: a product of two noisy
spectra has an expectation that goes as the signal squared, and a product
of their amplitudes one that goes as its fourth power. `Design.phases`
expands `D` itself. Do not expect a better normalization of the amplitudes
to save them.

**One model for every input.** A spectrum comes with a transfer `W`, and
is `W S` plus noise of variance `W`. `particles.whiten` divides it by the
square root, above a floor, which leaves the weight times the structure
plus white noise. A wedge of ones and zeros and a clean volume go through
it unchanged, so there is no branch on where a particle came from. Do not
add one. RELION's weight file is the transfer as it stands, not its square
root: the pseudo-subtomogram is a sum of tilts each multiplied by its own
transfer. Without the whitening the tilt axis, which every tilt contains,
decides every correlation.

**A model of the variance fitted to the data did not work.** Weighting a
pair by the transfer over a variance of `P W**2 + s W`, fitted shell by
shell, aligned four pairs in a hundred where whitening aligned thirty
seven. The fit puts all of the variance in the second term, which says the
plain correlation is already the matched filter, and the plain correlation
aligned eleven. Nobody knows why yet.

**Whose weight goes in the denominator matters.** With the same numerator,
normalizing each particle's energy by its own weight over the region both
measured aligned 37 pairs in a hundred. Letting the partner's weight into
it aligned 29.

**The Nyquist column.** `fftfreq(2B)` lays the orders out as
`0 .. B-1, -B, -(B-1) .. -1`. The column at index `B` is the order `-B`,
which no degree below the bandwidth reaches. Slicing a longer FFT down has
to leave that slot empty, or every negative order shifts by one.

**A traced constant reads as an implicit copy.** A table a traced
function closes over, such as the quarter-turn matrices, is folded into
the compiled program and uploaded by the runtime rather than through a
`device_put`. So it is reported as implicit however it was written. That
happens once per compiled function, not once per pair, which is why the
transfer tests assert on shapes rather than on counts: a count depends on
what else compiled first.

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

- The origins of the STAR output have never been checked against RELION.
  The angles agree with a refinement up to one global rotation. Whether
  RELION adds or subtracts the origin is untested, and would be silent if
  wrong.
- Accuracy on real data has been measured on one dataset, EMPIAR-10304,
  with particles that were already centred. How `Design.phases` holds up
  when they are not is unmeasured.
- The half-set correlation is not independent. The halves are aligned
  together.
- The second stage picks among the candidates slightly worse than the
  first stage ranked them, 29 right in a hundred against 37. Why is not
  known.
- `matching/frm5d.py` has the mathematics but not the map from a grid index
  back to a pair of rotations. It is not an engine yet, and `matching` does
  not dispatch to it.
- The loader reads volumes one at a time. It does not overlap reads with
  computation.
- The memory formula in `pipeline.Design` is reasoned, not measured. What
  is measured: 10357 particles in a band of 38 fit on a 32 GB device and
  align at about half a millisecond a pair. Two runs at once do not fit,
  since each takes most of the device for itself.

## Style

Eighty columns, single quotes, Google style docstrings, `mypy --strict`.
Match the surrounding code.

Prose in docstrings and commit messages explains *why*, and especially what
was tried and did not work. A reader who knows what the code does still
cannot tell which of the obvious simplifications is a trap. That is what
this file and those comments are for.

Short sentences. No em-dashes.
