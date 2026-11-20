<h1 align="center">frmony</h1>

<p align="center">
Ab-initio subtomogram averaging, by fast rotational matching and orthogonal
group synchronization.
</p>

---

`frmony` reconstructs an average volume from cryo-electron subtomograms
with no reference to start from.

The usual method refines each subtomogram against a growing average. That
needs a starting reference, and it tends to find whatever it was given.
`frmony` works the other way round. It aligns subtomograms to each other in
pairs. Each pair gives one noisy relative orientation and shift. It then
looks for the single set of orientations that fits all of those
measurements at once. Nothing is ever compared against a reference, so no
reference can bias the result.

> `frmony` is young. Its API may change between minor releases until 1.0.

## Installing

```bash
pip install frmony
```

With a GPU:

```bash
pip install 'frmony[cuda12]'
```

## Using it

RELION three-dimensional pseudo-subtomograms, version 4 or 5. The two write
the same files, and the version is detected rather than asked for.

```bash
frmony run Extract/job006/particles.star run1/ --partners 150 \
    --phases --mask-radius 16 --max-shift 4.5 --passes 3 --root . --verbose
```

`--phases` and `--mask-radius` are what real subtomograms have needed. The
first keeps the phases in the rotational search. The second is the radius
of a soft mask in voxels of the box, a little wider than the particle.

`--max-shift` is how far apart the centres of two particles may be, in
voxels of the analysis band. Rotations and shifts are searched together
out to that distance. Make it what the picking warrants and no more: the
further it reaches, the fewer pairs it aligns. For particles extracted at
refined centres, 1 is right. `--passes` brings the particles back to the
centres found and aligns again, which particles that are off centre need.

`--root` is the RELION project, which is what the paths in the table are
relative to.

Plain volumes, with the missing wedge given on the command line:

```bash
frmony run subtomograms.txt run1/ --tilt 60 --pixel-size 4.2
```

Four files are written. `average.mrc` is the average. `support.mrc` says
how much weight stands behind each frequency of it. `resolution.txt` holds
the half-set shell correlation. `synchronized.star` holds the orientations
and centres, beside every column that came in.

The Python API exposes each stage on its own:

```python
import frmony

subtomograms = frmony.load_subtomograms(
    paths, weight_paths=weights, pixel_size=4.2
)
result = frmony.run(subtomograms, frmony.Design(partners=16, symmetry='C1'))

print(result.resolution, result.orientations.shape)
```

## How it works

### Pairwise alignment

A pair has six unknowns, a rotation and a shift, and they are searched
together.

Each subtomogram is expanded in spherical harmonics on spheres of its
Fourier transform. One inverse FFT then gives the correlation of two of
them at every rotation of a grid at once. This is Kovacs and Wriggers
(2002). With `--phases` the correlation is the full one, phases included,
which is what real subtomograms need. It is then only right at the shift
the pair is actually at.

So the rotational search is run at a lattice of trial shifts. A shift of
a pair is a shift of its first particle, which is a phase on its
spectrum, so each trial costs one more expansion and one more grid. The
lattice is searched at half the bandwidth. The rotation is then found at
the full bandwidth, at the shift the lattice gave, and one more transform
refines the shift.

This searches the same six unknowns as the five degree of freedom method
of Kovacs et al. (2003), which `frmony.matching.frm5d` implements and
which it was measured against. They align the same pairs. This one takes
a thousandth of the time.

Without `--phases` the rotation is searched on amplitudes, which do not
move when the particle does, and the best few are then scored in full.
That is enough for clean volumes and not for subtomograms.

### Synchronization

A generalized power method finds the orientations that fit every pair best.
Least squares then gives the centres.

With `--passes`, the particles are brought back to those centres and
aligned again. A pair of centred particles is easier to align than a
pair that is not, so each pass aligns more.

### Averaging

The average is weighted, and taken in Fourier space. Each particle's
missing wedge turns with the particle. Averaging the volumes would stack
the gaps on top of one another. Weighting each particle by what it actually
measured lets the particles fill in each other's gaps instead. On simulated
data the coverage of Fourier space rises from 0.71 for a single particle to
0.996 for twenty-four.

### Symmetry

A particle with point group `G` has an orientation that is only defined up
to `G`. An alignment therefore measures some member of a coset, and not the
relative orientation itself. Go round a triangle of pairs and the residual
is a group element, but conjugated into a frame that nobody knows yet. That
is what makes this harder than ordinary outlier rejection.

Pass `--symmetry C4`. Before each round, every measurement is moved onto
the coset member closest to the current estimate. The step is exact at the
solution. It does nothing at all when the group is trivial, so there is one
code path and not two.

Residuals are measured modulo the group. This matters more than it sounds.
With C4 symmetry a residual of ninety degrees is a perfect measurement. A
rule that did not know about the symmetry would throw away exactly the
pairs that are right.

`frmony` also reports the symmetry it reads off the angles of the triangle
residuals. That works with no orientation estimate at all, because
conjugation does not change an angle. It sees angles and not axes, so it
tells the cyclic groups apart but reads `D2` as `C2`. Treat it as a warning
that the data disagrees with what you asked for, not as a decision about
what to use.

## Limitations

Worth reading before trusting the output.

**The origins in the STAR output have not been checked against RELION.**
The angles have: on EMPIAR-10304 they agree with those of a RELION
refinement up to one rotation of the whole set, and not with their
inverses. Whether RELION adds or subtracts the origin has not been tested.
It would be silent if wrong. The particles of that test were centred
already, so their origins were too small to tell.

**On simulated data.** Take a sixty degree tilt range and no noise. The
median error of a pair is then about five degrees, and synchronization
brings it down to three. With noise as large as the signal, the
synchronized median is five degrees. At twice the signal it is fourteen. At
three times it breaks down, at forty-three.

**On real data, one dataset, picked twice.** EMPIAR-10304, ribosomes as
RELION 5 pseudo-subtomograms of 48 voxels at 12.6 A, compared with the
orientations of a RELION refinement. All with `--phases
--mask-radius 16` and 150 partners.

The first picking is 10357 particles extracted at refined centres. With
`--max-shift 1`, the orientations came out with a median error of 5.2
degrees, 90 in a hundred within fifteen, in five minutes on one GPU.

The second is 12546 particles from another picking, about 24 A off centre,
of which 7671 are within 60 A of a refined particle and can be judged.
With `--max-shift 4.5 --passes 3`, those had a median error of 13 degrees,
57 in a hundred within fifteen and 81 within thirty, in 33 minutes. Their
centres came back to 5 A in the plane and 13 A along the beam. The
quarter picked furthest off, beyond 20 A in the plane, did worst, at 38
in a hundred.

That is one dataset, of a large particle, at one binning.

**Without `--phases` it does not work on that data.** The rotational
search on amplitudes aligned three pairs in a thousand, which is what
chance gives. Amplitudes do know something once each voxel is
standardized over the particles, which this package does not yet do, but
under half of what the phases know, and that is not enough to pick a
rotation out of a hundred thousand.

**A single pair is weak, and weaker when its shift is unknown.** On
centred particles 40 pairs in a hundred come out within fifteen degrees.
Letting the shift vary by five voxels halves that, whatever does the
searching: the true rotation is already at its best shift and every wrong
one gets to choose its luckiest. Synchronization is what makes up for it.

**The resolution is not a gold standard.** The two halves are aligned
together, and they share whatever the tomograms have in common. On the
data above, particles given no orientation at all still correlated to
36 A.

**Volumes are read one at a time.** The loader does not yet overlap reads
with computation.

## What is here

| Module | |
|---|---|
| `frmony.harmonics` | Spherical expansion of a volume, on a Gauss-Legendre quadrature |
| `frmony.matching` | The pairwise engine: rotation, translation, the constrained score |
| `frmony.synchronization` | Orientations by generalized power method, centres by least squares |
| `frmony.symmetry` | Point groups, and what they do to a set of measurements |
| `frmony.io` | RELION tables and volumes, through `starfile` and `vitrio` |

Two more engines ship, both off by default.

`matching.refine` polishes each pair against the score itself, by gradient.
It raises the score every time. Where a wide wedge is missing, though, the
peak of the score sits a little away from the true orientation. Climbing
harder can then walk away from the right answer.

`matching.frm5d` is the five degree of freedom method of Kovacs et al.
(2003), held against a direct sum by its tests. It is a yardstick and
not an engine: it searches what the pairwise engine searches, at 1.4
seconds a pair.

What it needs before it says anything about real subtomograms is in its
module docstring.

## References

- Kovacs & Wriggers (2002), *Fast rotational matching*, Acta Cryst. D**58**, 1282.
- Kovacs, Chacón, Cong, Metwally & Wriggers (2003), *Fast rotational matching
  of rigid bodies by fast Fourier transform acceleration of five degrees of
  freedom*, Acta Cryst. D**59**, 1371.
- Förster, Pruggnaller, Seybert & Frangakis (2008), constrained correlation
  with a missing wedge.
- Xu, Beck & Alber (2012), *High-throughput subtomogram alignment and
  classification by Fourier space constrained fast volumetric matching*,
  J. Struct. Biol. **178**, 150.

## Development

```bash
uv venv && uv pip install -e '.[dev]'
```

```bash
uv run --extra dev pytest && uv run --extra dev ruff check . && uv run --extra dev mypy
```

## License

GPL-3.0-or-later. See `LICENSE`.
