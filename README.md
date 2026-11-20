<h1 align="center">syncsta</h1>

<p align="center">
Ab-initio subtomogram averaging by rotational matching and orthogonal group
synchronization.
</p>

---

`syncsta` builds a structure from cryo-electron tomograms without a reference
to start from. It aligns subtomograms to each other in **pairs**, which gives a
noisy relative rotation and shift for each pair, and then finds the single set
of orientations that agrees best with all of those measurements at once. The
rotations come from orthogonal group synchronization and the shifts from a
linear least-squares problem.

Because nothing is aligned against a growing average, the result carries no
reference bias.

The missing wedge, and the contrast transfer function of a RELION
pseudo-subtomogram, are both carried as one non-negative weight in Fourier
space. That weight only ever multiplies: nothing is divided by a value the
microscope left at zero.

> `syncsta` is young. Its API may change between minor releases until 1.0.

## Status

Under construction. See `CHANGELOG.md`.

## License

GPL-3.0-or-later. See `LICENSE`.
