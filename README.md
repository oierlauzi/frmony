<h1 align="center">frmony</h1>

<p align="center">
Ab-initio subtomogram averaging, by fast rotational matching and orthogonal
group synchronization.
</p>

---

`frmony` builds a structure from cryo-electron tomograms with no reference
to start from.

The usual route — refine against a growing average — needs something to
begin with, and tends to find whatever it was given. `frmony` takes the
other one. It aligns subtomograms to each other in **pairs**, which gives a
noisy relative orientation and shift for each pair, and then finds the
single set of orientations that agrees best with all of those measurements
at once. The rotations come from orthogonal group synchronization and the
centres from linear least squares. Nothing is ever compared against a
reference, so nothing can be biased by one.

The missing wedge and the contrast transfer function are the same kind of
fact about the data — some frequencies were measured well, some badly, some
not at all — and both are carried as one non-negative weight in Fourier
space. **That weight only ever multiplies.** Nothing is divided by it, which
matters because it is exactly zero across the wedge and at every zero of the
transfer function.

> `frmony` is young. Its API may change between minor releases until 1.0.

## Installing

```bash
pip install frmony
```

For a GPU, which is where this belongs:

```bash
pip install 'frmony[cuda12]'
```

## Using it

RELION three-dimensional pseudo-subtomograms, four or five — the format is
identical and the version is detected, not asked for:

```bash
frmony run Extract/job006/particles.star run1/ --partners 16 --verbose
```

Plain volumes, with the missing wedge named:

```bash
frmony run subtomograms.txt run1/ --tilt 60 --pixel-size 4.2
```

Out comes `average.mrc`, a `support.mrc` saying which frequencies the answer
actually rests on, a `resolution.txt` holding the half-set shell
correlation, and a `synchronized.star` carrying the orientations and centres
beside every column that came in.

From Python, where each stage stands on its own:

```python
import frmony

subtomograms = frmony.load_subtomograms(paths, weight_paths=weights,
                                        pixel_size=4.2)
result = frmony.run(subtomograms, frmony.Design(partners=16, symmetry='C1'))

print(result.resolution, result.orientations.shape)
```

## How it works

**Pairwise alignment** is in two stages. The rotation is found first, on the
*amplitudes* of the two Fourier transforms, which do not move when the
particle does — so the rotation can be found before the shift is known. That
is one inverse FFT over all of SO(3) at once, which is Kovacs and Wriggers
(2002). The best few rotations are kept rather than the best one, because
amplitudes throw away the phases and so discriminate less; the right one is
reliably among the best few even when it is not the best. Each candidate
then gets a single transform giving the correlation at every shift at once,
scored in full.

**Synchronization** maximizes the agreement with every pair at once by a
generalized power method, then solves the centres by least squares.

**Averaging** happens in Fourier space, weighted. Each particle's wedge
turns with the particle, so a mean of the volumes would add the gaps
together; weighting by what each one actually measured lets them supply each
other instead. On simulated data the coverage of Fourier space goes from
0.71 for a single particle to 0.996 for twenty-four.

### Symmetry

A particle with point group `G` has an orientation defined only up to `G`,
so an alignment measures an arbitrary member of a coset rather than the
relative orientation itself. Around a triangle the residual is a group
element *conjugated into a frame nobody knows yet*, which is what makes this
harder than ordinary outlier rejection.

Pass `--symmetry C4` and each measurement is snapped onto the coset member
nearest the current estimate before each round, which is exact at the
solution and does nothing at all when the group is trivial. Residuals are
measured modulo the group, because with C4 a residual of ninety degrees is a
*perfect* measurement and a rule that did not know would discard exactly the
pairs that are right.

The symmetry is also detected, always, from the angles of the triangle
residuals — which works with no orientation in hand, because an angle does
not change under conjugation. It sees angles and not axes, so it separates
the cyclic groups and confuses `D2` with `C2`; it is a warning that what you
asked for does not match the data, not a decision about what to use.

## What is here

| Module | |
|---|---|
| `frmony.harmonics` | Spherical expansion of a volume, exactly, on a Gauss-Legendre quadrature |
| `frmony.matching` | The pairwise engine: rotation, translation, the constrained score |
| `frmony.synchronization` | Orientations by generalized power method, centres by least squares |
| `frmony.symmetry` | Point groups, and what they do to a set of measurements |
| `frmony.io` | RELION tables and volumes, through `starfile` and `vitrio` |

Two further engines ship and are off by default. `matching.refine` polishes
each pair against the score itself by gradient; it raises the score every
time, but where a wide wedge is missing the peak of the score is not quite
at the true orientation, so climbing harder can walk away from it.
`matching.frm5d` holds the mathematics of the five-degree-of-freedom method
of Kovacs et al. (2003) — the two-centre integrals and the five-dimensional
series, held to an exact identity — but not yet the map from a grid index
back to a pair of rotations, so it is not yet an engine to dispatch to.

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
