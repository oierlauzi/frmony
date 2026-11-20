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
frmony run Extract/job006/particles.star run1/ --partners 16 --verbose
```

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

This happens in two stages.

The first stage finds the rotation. It works on the amplitudes of the two
Fourier transforms, which do not move when the particle does, so the
rotation can be found before the shift is known. One inverse FFT gives the
correlation at every rotation of a grid at once. This is Kovacs and
Wriggers (2002).

That stage keeps the best few rotations rather than the best one.
Amplitudes throw away the phases, so they discriminate less than the full
score does. The right rotation is reliably among the best few even when it
is not the best one.

The second stage takes each of those candidates and runs one more
transform. That gives the correlation at every shift at once, with the
phases back in.

### Synchronization

A generalized power method finds the orientations that fit every pair best.
Least squares then gives the centres.

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

**The STAR output has not been checked against RELION.** `frmony` writes
its orientations as ZYZ Euler angles and its centres in angstroms. That is
the natural reading of those columns. Whether RELION takes an orientation
to mean that one or its inverse, and whether it adds or subtracts the
origin, has not been tested against RELION itself. Either would be silent
if wrong, and would give a wrong map rather than an error. Check the output
against a refinement of the same particles before relying on it.

**Accuracy has only been measured on simulated data.** Take a sixty degree
tilt range and no noise. The median error of a pair is then about five
degrees, and synchronization brings it down to three. With noise as large
as the signal, the synchronized median is five degrees. At twice the signal
it is fourteen. At three times it breaks down, at forty-three. Real
subtomograms are often noisier than that. `frmony` has not been run on real
data at all.

**Volumes are read one at a time.** The loader does not yet overlap reads
with computation.

**A connected graph of pairs is not enough.** Two groups of particles may
hang together by a single pair. Everything they say about each other then
rests on that one measurement. Nothing else in the data can contradict it.
`frmony` warns when the surviving pairs include such a bridge. The graphs
it builds by default have none. This is about what heavy outlier rejection
can leave behind.

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

`matching.frm5d` holds the mathematics of the five degree of freedom method
of Kovacs et al. (2003). The two-centre integrals and the five-dimensional
series are built and tested against an exact identity. What is missing is
the map from a grid index back to a pair of rotations, so this is not yet
an engine that `matching` will dispatch to.

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
