# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
