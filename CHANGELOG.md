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
