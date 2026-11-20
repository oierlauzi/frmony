# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first run on real data, EMPIAR-10304.

### Fixed

- `graph.build_pairs` gave each particle between two and a half and three
  times the partners asked for.
- The resolution of every real run came out infinite, because the empty
  innermost shell of the correlation was reported as zero.
- RELION pseudo-subtomograms were read as if their noise were white. It
  has the variance of the weight file. Every input is now read as one
  model, a spectrum with a transfer `W` and noise of variance `W`, and
  `particles.whiten` divides it by the square root above a floor. A wedge
  and a clean volume come through unchanged. A RELION weight is taken as
  it stands by default, where it used to be its square root.
- Two particles are compared over the frequencies both measured, and no
  longer in proportion to their weights.
- The time a pair took was logged in whole milliseconds, so as zero.

- The rotational search stopped at a third of the sampling rate. It
  now runs to the last sphere that fits in the band.
- The search over shifts ran over every frequency, the lowest among
  them. It runs over the band of the rotations.
- The series of `matching.frm5d` was wrong, and is derived again from
  the three-dimensional one and held against a direct sum.

### Changed

- With phases, the second stage finds the shift of the first stage's
  best rotation and no longer chooses among candidates.
- `Design.batch` defaults to 8, which was measured, where 32 was
  reasoned and is two and a half times slower.
- A mask is carried with the particles and applied at the centre each is
  believed to have. `load_subtomograms` no longer applies it.
- `--max-shift` is how far apart two particles' centres may be, and with
  phases it is how far rotations and shifts are searched together.

### Added

- `Design.phases` and `--phases`, on by default, which keep the phases in
  the rotational search. Real subtomograms needed it. `--no-phases` goes
  back to amplitudes.
- Rotations and shifts searched at once, over a lattice of trial shifts,
  for particles that are off centre: `matching.Joint`.
- `Design.passes` and `--passes`, which bring the particles back to the
  centres found and align again.
- `matching.frm5d.grid_to_rotations`, which makes the five-dimensional
  method usable as a yardstick.

## [0.1.0] - 2026-11-20

First release.

### Changed

- A format reader registered with `io.register` now takes the root as a
  second argument.
- `Design.batch` defaults to 32 rather than 4, which was chosen for a
  laptop and leaves a GPU waiting on dispatch.

### Added

- The skeleton of the package: packaging, linting, type checking, tests and
  continuous integration.
- `geometry`, with the orientation convention of RELION, the maps between
  Euler angles, matrices and rotation vectors, and the projection onto the
  closest rotation.
- `symmetry`, with the point groups and with what a particle's symmetry does
  to a set of pairwise measurements: snapping them back onto one coset, and
  detecting the group from the angles of the triangle residuals alone.
- `graph`, which chooses the pairs to align and finds a spanning tree of them.
- `harmonics`, which reads a volume off concentric spheres and expands each
  one into spherical harmonics. The colatitude is integrated by a
  Gauss-Legendre quadrature, which is exact for a band-limited function,
  rather than by a uniform grid with a `sin(theta)` weight, which is not
  exact at any number of nodes. It also builds the quarter-turn Wigner
  matrices, as the exponential of the generator of rotations about y.
- `matching.rotation`, the fast rotational matching itself: one inverse FFT
  gives the correlation of two expansions at every rotation of a grid. Which
  Euler angles a grid index stands for was found by holding the grid against
  a quadrature of the integral over the sphere, not reasoned out on paper,
  and a test keeps it honest.
- The rest of `matching`: the constrained score that two subtomograms are
  compared by, the translational search, peak interpolation, and the driver
  that aligns a pair in two stages. The missing wedge and the contrast
  transfer function are carried as one non-negative Fourier weight, which
  only ever multiplies.
- `harmonics.band_radii`, which chooses the spheres to expand on. The
  choice matters more than almost anything else: dropping the lowest
  frequencies, which carry a particle's mass but not its orientation, took
  a median pairwise error from eight degrees to three and removed every
  catastrophic failure on the simulation it was measured on.
- `synchronization`, which turns the pairwise measurements into one set of
  orientations by a generalized power method, and then into centres by
  linear least squares. Symmetry is handled by snapping each measurement
  onto the coset member nearest the current estimate, and outliers by their
  residual measured modulo the group.
- `io`, `particles`, `averaging`, `pipeline` and a command line. RELION 4
  and 5 three-dimensional pseudo-subtomograms are read, as are plain
  volumes with a tilt range, through a registry that takes new conventions
  without being edited. The average is a weighted one in Fourier space,
  where the particles supply each other the frequencies their own missing
  wedges never reached.
- `matching.refine`, an optional polish of each pair against the score
  itself, by gradient. Off by default: it raises the score every time, but
  where a wide wedge is missing the peak of the score is not quite at the
  true orientation, so climbing harder can walk away from it.
- `matching.frm5d`, the mathematics of the five-degree-of-freedom method of
  Kovacs et al. (2003): the two-centre integrals, through the cylindrical
  factorization that makes them one batched matrix product, and the
  five-dimensional series over them. The integrals are held to an exact
  identity, that at a distance of zero they collapse onto the ordinary
  radial overlap. Which rotations a grid index stands for is not yet
  settled, so this is not yet an engine `pairwise` can dispatch to; the
  module says so.
- `_transfers`, which gives every copy between host and device a name,
  and `record_transfers` to count them. The particles now cross once as a
  set: expanding them in a Python loop sent each one over, brought its
  coefficients back and sent those over again, so the copies grew with the
  number of particles.
- `--root`, for a STAR file whose images are not beside it. RELION writes
  those paths relative to the project.
- `graph.bridges`, and a warning when the pairs that survive rejection
  include one. A connected graph is not enough: two groups held together
  by a single pair come out in unrelated frames while every residual looks
  small and `is_connected` says the graph is fine.
