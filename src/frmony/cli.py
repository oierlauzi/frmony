"""Command line interface.

It is made of commands, ``frmony COMMAND ...``. A command is a function
that adds its parser to those of the others, with the function that runs it
as the ``handle`` default, and is listed in `build_parser`. A handler is
given the parsed arguments whole.

Options that more than one command takes are defined once, in the parent
parsers below, so that they read and behave the same everywhere.

``ab-initio`` aligns particles to one another and needs no reference.
``align`` aligns them to a reference. ``run`` is what ``ab-initio`` was
called before there were two, and still works, with a warning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .io.registry import Source
    from .particles import Subtomograms
    from .reconstruction import Reconstruction

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'

_LOGGER = logging.getLogger(__name__)


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f'{text} is not positive.')
    return value


def _count(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'{text} is not a count.')
    return value


# ------------------------------------------------------------ parent parsers


def _shared_parser() -> argparse.ArgumentParser:
    """Options that every command takes."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--device',
        help="Where JAX computes: 'cpu', 'gpu' or, for a given one, 'gpu:1'.",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Report what each step is doing.',
    )
    return parser


def _particles_parser() -> argparse.ArgumentParser:
    """The particles, and how to read them."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        'input',
        help='A RELION particle table, or a file listing volumes one a line.',
    )
    parser.add_argument(
        '--format',
        help='How the input is laid out. Guessed from its suffix if not given.',
    )
    parser.add_argument(
        '--root',
        help='Directory the image paths inside the input are relative to. '
        "The input's own directory if not given. RELION writes those paths "
        'relative to the project, so a table copied away from its project '
        'needs this to find the images.',
    )
    parser.add_argument(
        '--band',
        type=_positive,
        default=0.4,
        help='Highest frequency kept, as a fraction of the sampling rate '
        '(default: %(default)s).',
    )
    parser.add_argument(
        '--tilt',
        type=float,
        help='Half the tilt range [deg], to build a missing wedge from '
        'when the input names no weight volumes.',
    )
    parser.add_argument(
        '--cone',
        type=float,
        help='Half aperture [deg], for a missing cone instead of a wedge.',
    )
    parser.add_argument(
        '--weight-transform',
        choices=('identity', 'sqrt', 'none'),
        default='identity',
        help='How to read a RELION weight volume as the transfer of the '
        "data (default: %(default)s). 'identity' takes it as it stands, "
        "which is what RELION accumulates. 'none' keeps only where it is "
        'not zero.',
    )
    parser.add_argument(
        '--mask-radius',
        type=_positive,
        help='Radius of a soft real-space mask [voxels]. The volumes are '
        'left alone if not given.',
    )
    parser.add_argument(
        '--pixel-size',
        type=_positive,
        help='Of the box [A], when the input does not say.',
    )
    return parser


def _search_parser() -> argparse.ArgumentParser:
    """How a pair of particles is searched; see `frmony.engine.Search`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--bandwidth',
        type=_count,
        default=32,
        help='Degrees kept in the spherical expansion (default: '
        '%(default)s). The grid of rotations steps by 180 over this, in '
        'degrees.',
    )
    parser.add_argument(
        '--candidates',
        type=_count,
        default=10,
        help='Rotations carried into the translational search (default: '
        '%(default)s).',
    )
    parser.add_argument(
        '--max-shift',
        type=float,
        default=6.0,
        help='How far apart the centres of two particles may be, in '
        'voxels of the analysis band (default: %(default)s). With '
        '--phases, rotations and shifts are searched together out to '
        'this distance. Make it what the picking warrants and no more: '
        'the further the search reaches, the fewer pairs it aligns.',
    )
    parser.add_argument(
        '--phases',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep the phases in the rotational search (default: on). '
        'Real subtomograms need it: on amplitudes alone they align by '
        'chance. --no-phases searches rotations on amplitudes, which '
        'is enough for clean volumes.',
    )
    parser.add_argument(
        '--batch',
        type=_count,
        default=8,
        help='Pairs aligned at once, and particles expanded at once '
        '(default: %(default)s). Turn it down if the device runs out of '
        'memory. Turning it up has been measured to be slower.',
    )
    parser.add_argument(
        '--mirrors',
        action='store_true',
        help='Also consider mirror images: a particle may come out as the '
        'mirror image of the others, or of the reference. Its orientation '
        'is then written as the rotation it is minus, with frmonyMirrored '
        'set in the STAR file. Off unless asked for: a chiral particle has '
        'no mirror image in the data. Needs --phases, which tell a particle '
        'from its mirror; amplitudes cannot.',
    )
    parser.add_argument(
        '--memory',
        type=_positive,
        help='Device memory to plan for [GB]. What the device says it has '
        'if not given. Particles that do not fit are handed to it a block '
        'at a time, so a small device takes longer but not less.',
    )
    return parser


def _search_options(arguments: argparse.Namespace) -> dict[str, Any]:
    """What `_search_parser` read, as `frmony.engine.Search` takes it."""
    return {
        'bandwidth': arguments.bandwidth,
        'candidates': arguments.candidates,
        'max_shift': arguments.max_shift,
        'phases': arguments.phases,
        'batch': arguments.batch,
        'memory': arguments.memory,
        'mirrors': arguments.mirrors,
    }


def _writing_parser() -> argparse.ArgumentParser:
    """What is written besides the orientations."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--average',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Average the aligned particles at the size they were '
        'extracted at, with its support and the correlation of its halves '
        '(default: on). --no-average writes the alignment alone and does '
        'not read the particles a second time.',
    )
    return parser


# ------------------------------------------------------- what commands share


@dataclass(frozen=True)
class _Loaded:
    source: Source
    subtomograms: Subtomograms


def _load(arguments: argparse.Namespace) -> _Loaded:
    """Read the particles `_particles_parser` names."""
    from .io.registry import read_source
    from .particles import load_subtomograms

    source = read_source(arguments.input, arguments.format, arguments.root)
    size = arguments.pixel_size or source.pixel_size
    if size is None:
        raise ValueError(
            'The input does not give a pixel size; pass --pixel-size.'
        )
    if (
        source.weights is None
        and arguments.tilt is None
        and arguments.cone is None
    ):
        raise ValueError(
            'The input names no weight volumes, so the missing region '
            'has to be given: pass --tilt or --cone.'
        )
    subtomograms = load_subtomograms(
        source.data,
        weight_paths=_weight_paths(arguments, source),
        pixel_size=size,
        band=arguments.band,
        tilt=arguments.tilt,
        cone=arguments.cone,
        weight_transform=arguments.weight_transform,
        mask_radius=arguments.mask_radius,
        groups=source.groups,
    )
    return _Loaded(source=source, subtomograms=subtomograms)


def _weight_paths(
    arguments: argparse.Namespace, source: Source
) -> list[str] | None:
    """The weight files, unless a wedge or a cone is to be built instead."""
    if arguments.tilt or arguments.cone:
        return None
    return source.weights


def _write_orientations(
    directory: Path,
    name: str,
    source: Source,
    orientations: np.ndarray,
    shifts: np.ndarray,
    confidence: np.ndarray | None,
) -> None:
    """The orientations and centres: a STAR file beside what came in, or a
    table for an input that was a list of volumes."""
    from .io.star import Metadata, write_star

    if isinstance(source.metadata, Metadata):
        write_star(
            directory / f'{name}.star',
            source.metadata,
            orientations,
            shifts,
            confidence,
        )
        return
    np.savetxt(
        directory / 'orientations.txt',
        np.concatenate(
            (orientations.reshape(len(orientations), 9), shifts), axis=1
        ),
        header='nine matrix entries then the centre in voxels (x y z)',
    )


def _write_average(
    directory: Path,
    arguments: argparse.Namespace,
    loaded: _Loaded,
    orientations: np.ndarray,
    shifts: np.ndarray,
    aligned_up_to: float,
) -> Reconstruction:
    """The average at the size the particles were extracted at, its
    support, and the correlation of its halves."""
    from .io.maps import write_map, write_spectrum
    from .reconstruction import reconstruct

    full = reconstruct(
        loaded.source.data,
        orientations,
        shifts,
        loaded.subtomograms.pixel_size,
        weight_paths=_weight_paths(arguments, loaded.source),
        weight_transform=arguments.weight_transform,
        tilt=arguments.tilt,
        cone=arguments.cone,
    )
    write_spectrum(directory / 'average.mrc', full.structure, full.pixel_size)
    write_map(
        directory / 'support.mrc',
        np.fft.fftshift(full.support),
        full.pixel_size,
    )
    np.savetxt(
        directory / 'resolution.txt',
        np.stack((full.frequency, full.correlation), axis=1),
        header=f'{_resolution_words(full)}; aligned up to '
        f'{aligned_up_to:.2f} A\nfrequency fsc',
    )
    return full


def _resolution_words(full: Reconstruction) -> str:
    if np.isfinite(full.resolution):
        return f'resolution {full.resolution:.2f} A'
    return f'the halves agree to Nyquist, {2 * full.pixel_size:.2f} A'


def _print_mirrors(orientations: np.ndarray) -> None:
    mirrors = int(np.sum(np.linalg.det(orientations) < 0.0))
    if mirrors:
        print(
            f'{mirrors} of them are mirror images, marked as frmonyMirrored '
            'in the STAR file.'
        )


def _on_device(arguments: argparse.Namespace) -> Any:
    import jax

    from ._device import resolve_device

    return jax.default_device(resolve_device(arguments.device))


# ----------------------------------------------------------------- ab-initio


def _add_ab_initio(
    commands: Any, parents: list[argparse.ArgumentParser]
) -> None:
    parser = commands.add_parser(
        'ab-initio',
        aliases=['run'],
        parents=parents,
        help='Align subtomograms to one another, and average them.',
        description=(
            'Align subtomograms to each other in pairs, find the '
            'orientations that agree best with every pair at once, and '
            'average what comes out. No reference is needed and none is '
            'used, so the result carries no reference bias.'
        ),
    )
    parser.add_argument('output', help='Directory to write the results into.')
    parser.add_argument(
        '--partners',
        type=_count,
        default=16,
        help='Roughly how many others each particle is aligned to '
        '(default: %(default)s). Clamped to one fewer than the number of '
        'particles.',
    )
    parser.add_argument(
        '--symmetry',
        default='C1',
        help='Point group of the particle: C1, C<n>, D<n>, T, O or I '
        '(default: %(default)s).',
    )
    parser.add_argument(
        '--rounds',
        type=_count,
        default=8,
        help='Times to snap, synchronize and weigh the pairs (default: '
        '%(default)s).',
    )
    parser.add_argument(
        '--reject-above',
        type=float,
        default=0.0,
        help='Also drop pairs whose residual exceeds this [deg], measured '
        'modulo the symmetry. Zero, the default, keeps them all: each '
        'pair is weighed by how likely its residual says it is to be '
        'right, which needs no threshold.',
    )
    parser.add_argument(
        '--seed', type=int, default=0, help='Which pairs get chosen.'
    )
    parser.add_argument(
        '--passes',
        type=_count,
        default=1,
        help='Times to align, synchronize and bring the particles back '
        'to the centres found (default: %(default)s). Particles that are '
        'off centre need more than one.',
    )
    parser.add_argument(
        '--refine',
        action='store_true',
        help='Polish each pair against the score itself after the grid '
        'search. This reliably raises the score, but where a wide wedge is '
        'missing the peak of the score is not quite at the true '
        'orientation, so it can move the answer away from it. Off unless '
        'asked for.',
    )
    parser.set_defaults(handle=_ab_initio)


def _ab_initio(arguments: argparse.Namespace) -> None:
    """Align the particles to one another and write what was found."""
    from .pipeline import Design, run

    if arguments.command == 'run':
        _LOGGER.warning("'frmony run' is now 'frmony ab-initio'.")
    # Built first, so that what it refuses is refused before the particles
    # are read.
    design = Design(
        partners=arguments.partners,
        symmetry=arguments.symmetry,
        rounds=arguments.rounds,
        reject_above=(
            arguments.reject_above if arguments.reject_above > 0.0 else None
        ),
        seed=arguments.seed,
        refine=arguments.refine,
        passes=arguments.passes,
        # The average over the band would go unused: the one written
        # is made from the particles as they were extracted.
        average=False,
        **_search_options(arguments),
    )
    with _on_device(arguments):
        loaded = _load(arguments)
        result = run(loaded.subtomograms, design)

    directory = Path(arguments.output)
    directory.mkdir(parents=True, exist_ok=True)
    # The orientations first, so that they are kept whatever happens to
    # the average.
    _write_orientations(
        directory,
        'synchronized',
        loaded.source,
        result.orientations,
        result.shifts,
        result.confidence,
    )
    full = None
    if arguments.average:
        with _on_device(arguments):
            full = _write_average(
                directory,
                arguments,
                loaded,
                result.orientations,
                result.shifts,
                result.band_limit,
            )
    _LOGGER.info('Wrote the results to %s.', directory)

    print(
        f'{len(loaded.subtomograms)} particles, {int(result.kept.sum())} of '
        f'{len(result.left)} pairs kept.'
    )
    if result.confidence is not None:
        held = int(np.sum(result.confidence > 0.0))
        print(
            f'{held} of them held together by the pairs that look right; '
            "the share of each particle's pairs that do is written as "
            'rlnMaxValueProbDistribution, nought for the rest.'
        )
    if result.symmetry_doubted:
        print(
            f'symmetry asked for {arguments.symmetry}, but the pairs look '
            f'more like {result.detected}'
        )
    _print_mirrors(result.orientations)
    if full is not None:
        print(
            f'{_resolution_words(full)}, at {full.pixel_size:.2f} A a '
            f'pixel; aligned up to {result.band_limit:.1f} A, and the '
            'halves were aligned together, so the curve is fair only beyond '
            'that'
        )


# --------------------------------------------------------------------- align


def _add_align(commands: Any, parents: list[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser(
        'align',
        parents=parents,
        help='Align subtomograms to a reference, and average them.',
        description=(
            'Align each subtomogram to a clean reference volume, one at a '
            'time, the way ab-initio aligns two of them to each other, and '
            'average them in its frame. The particles may be anything the '
            'input can hold: clean or noisy volumes, with or without a '
            'missing wedge, or RELION pseudo-subtomograms. The reference '
            'is taken to be clean, measured everywhere, and in the same box '
            'as the particles.'
        ),
    )
    parser.add_argument(
        'reference', help='The reference: one volume, in a file of its own.'
    )
    parser.add_argument('output', help='Directory to write the results into.')
    parser.set_defaults(handle=_align)


def _align(arguments: argparse.Namespace) -> None:
    """Align the particles to the reference and write what was found."""
    from .engine import Search
    from .particles import load_reference
    from .reference import align_to_reference

    # Built first, so that what it refuses is refused before the particles
    # are read.
    search = Search(**_search_options(arguments))
    with _on_device(arguments):
        loaded = _load(arguments)
        reference = load_reference(arguments.reference, loaded.subtomograms)
        found = align_to_reference(loaded.subtomograms, reference, search)

    directory = Path(arguments.output)
    directory.mkdir(parents=True, exist_ok=True)
    _write_orientations(
        directory,
        'aligned',
        loaded.source,
        found.orientations,
        found.shifts,
        found.scores,
    )
    subtomograms = loaded.subtomograms
    full = None
    if arguments.average:
        aligned_up_to = 2.0 * subtomograms.pixel_size * subtomograms.scale
        with _on_device(arguments):
            full = _write_average(
                directory,
                arguments,
                loaded,
                found.orientations,
                found.shifts,
                aligned_up_to,
            )
    _LOGGER.info('Wrote the results to %s.', directory)

    print(
        f'{len(subtomograms)} particles aligned to the reference; their '
        'scores are written as rlnMaxValueProbDistribution.'
    )
    _print_mirrors(found.orientations)
    if full is not None:
        print(
            f'{_resolution_words(full)}, at {full.pixel_size:.2f} A a '
            'pixel. Aligned to a reference, the particles come to look like '
            'it, so the agreement of the halves is biased towards it.'
        )


# ------------------------------------------------------------------- formats


def _add_formats(commands: Any, parents: list[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser(
        'formats',
        parents=parents,
        help='List the input layouts that can be read.',
    )
    parser.set_defaults(handle=_list_formats)


def _list_formats(arguments: argparse.Namespace) -> None:
    del arguments
    from .io.registry import available

    for name in available():
        print(name)


# -------------------------------------------------------------------- wiring

_Adder = Callable[[Any, list[argparse.ArgumentParser]], None]


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command."""
    parser = argparse.ArgumentParser(
        prog='frmony',
        description=(
            'Subtomogram averaging by rotational matching: ab-initio, by '
            'orthogonal group synchronization, or to a reference.'
        ),
    )
    parser.add_argument(
        '--version',
        action='version',
        version=metadata.version('frmony'),
    )
    commands = parser.add_subparsers(dest='command', required=True)
    shared = _shared_parser()
    aligning = [
        shared,
        _particles_parser(),
        _search_parser(),
        _writing_parser(),
    ]
    adders: tuple[tuple[_Adder, list[argparse.ArgumentParser]], ...] = (
        (_add_ab_initio, aligning),
        (_add_align, aligning),
        (_add_formats, [shared]),
    )
    for add, parents in adders:
        add(commands, parents)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command.

    Args:
        argv: The arguments, without the name of the program. Those of the
            process when none are given.

    Returns:
        The exit status of the process.
    """
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format=_LOG_FORMAT,
    )
    try:
        arguments.handle(arguments)
    except (OSError, ValueError) as error:
        print(f'frmony: {error}', file=sys.stderr)
        return 1
    return 0
