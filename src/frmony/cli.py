"""Command line interface.

It is made of commands, ``frmony COMMAND ...``. A command is a function
that adds its parser to those of the others, with the function that runs it
as the ``run`` default, and is listed in `build_parser`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


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


def _add_run(commands: Any, shared: argparse.ArgumentParser) -> None:
    parser = commands.add_parser(
        'run',
        parents=[shared],
        help='Align, synchronize and average a set of subtomograms.',
        description=(
            'Align subtomograms to each other in pairs, find the '
            'orientations that agree best with every pair at once, and '
            'average what comes out. No reference is needed and none is '
            'used, so the result carries no reference bias.'
        ),
    )
    parser.add_argument(
        'input',
        help='A RELION particle table, or a file listing volumes one a line.',
    )
    parser.add_argument('output', help='Directory to write the results into.')
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
        '--partners',
        type=_count,
        default=16,
        help='Roughly how many others each particle is aligned to '
        '(default: %(default)s). Clamped to one fewer than the number of '
        'particles.',
    )
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
        '--band',
        type=_positive,
        default=0.4,
        help='Highest frequency kept, as a fraction of the sampling rate '
        '(default: %(default)s).',
    )
    parser.add_argument(
        '--symmetry',
        default='C1',
        help='Point group of the particle: C1, C<n>, D<n>, T, O or I '
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
        '--batch',
        type=_count,
        default=8,
        help='Pairs aligned at once, and particles expanded at once '
        '(default: %(default)s). Turn it down if the device runs out of '
        'memory. Turning it up has been measured to be slower.',
    )
    parser.add_argument(
        '--memory',
        type=_positive,
        help='Device memory to plan for [GB]. What the device says it has '
        'if not given. Particles that do not fit are handed to it a block '
        'at a time, so a small device takes longer but not less.',
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
        '--phases',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep the phases in the rotational search (default: on). '
        'Real subtomograms need it: on amplitudes alone they align by '
        'chance. --no-phases searches rotations on amplitudes, which '
        'is enough for clean volumes.',
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
    parser.set_defaults(run=_run_pipeline)


def _run_pipeline(
    input: str,
    output: str,
    format: str | None,
    root: str | None,
    partners: int,
    bandwidth: int,
    candidates: int,
    max_shift: float,
    band: float,
    symmetry: str,
    tilt: float | None,
    cone: float | None,
    weight_transform: str,
    mask_radius: float | None,
    pixel_size: float | None,
    rounds: int,
    reject_above: float,
    batch: int,
    memory: float | None,
    seed: int,
    refine: bool,
    passes: int,
    phases: bool,
    device: str | None,
) -> None:
    """Run the whole pipeline and write what it found."""
    import jax

    from ._device import resolve_device
    from .io.maps import write_map, write_spectrum
    from .io.registry import read_source
    from .io.star import write_star
    from .particles import load_subtomograms
    from .pipeline import Design, run

    logger = logging.getLogger(__name__)
    with jax.default_device(resolve_device(device)):
        source = read_source(input, format, root)
        size = pixel_size or source.pixel_size
        if size is None:
            raise ValueError(
                'The input does not give a pixel size; pass --pixel-size.'
            )
        if source.weights is None and tilt is None and cone is None:
            raise ValueError(
                'The input names no weight volumes, so the missing region '
                'has to be given: pass --tilt or --cone.'
            )

        subtomograms = load_subtomograms(
            source.data,
            weight_paths=None if (tilt or cone) else source.weights,
            pixel_size=size,
            band=band,
            tilt=tilt,
            cone=cone,
            weight_transform=weight_transform,
            mask_radius=mask_radius,
            groups=source.groups,
        )
        design = Design(
            partners=partners,
            bandwidth=bandwidth,
            candidates=candidates,
            max_shift=max_shift,
            symmetry=symmetry,
            rounds=rounds,
            reject_above=reject_above if reject_above > 0.0 else None,
            batch=batch,
            memory=memory,
            seed=seed,
            refine=refine,
            passes=passes,
            phases=phases,
        )
        result = run(subtomograms, design)

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)

    # The orientations first, so that they are kept whatever happens to
    # the average.
    from .io.star import Metadata

    if isinstance(source.metadata, Metadata):
        write_star(
            directory / 'synchronized.star',
            source.metadata,
            result.orientations,
            result.shifts,
            result.confidence,
        )
    else:
        np.savetxt(
            directory / 'orientations.txt',
            np.concatenate(
                (
                    result.orientations.reshape(len(result.orientations), 9),
                    result.shifts,
                ),
                axis=1,
            ),
            header='nine matrix entries then the centre in voxels (x y z)',
        )

    # The average from the particles as they were extracted, not from the
    # band the alignment worked on.
    from .reconstruction import reconstruct

    with jax.default_device(resolve_device(device)):
        full = reconstruct(
            source.data,
            result.orientations,
            result.shifts,
            size,
            weight_paths=None if (tilt or cone) else source.weights,
            weight_transform=weight_transform,
            tilt=tilt,
            cone=cone,
        )
    write_spectrum(directory / 'average.mrc', full.structure, full.pixel_size)
    write_map(
        directory / 'support.mrc',
        np.fft.fftshift(full.support),
        full.pixel_size,
    )
    if np.isfinite(full.resolution):
        verdict = f'resolution {full.resolution:.2f} A'
    else:
        verdict = f'the halves agree to Nyquist, {2 * full.pixel_size:.2f} A'
    edge = f'{result.band_limit:.2f} A'
    np.savetxt(
        directory / 'resolution.txt',
        np.stack((full.frequency, full.correlation), axis=1),
        header=f'{verdict}; aligned up to {edge}\nfrequency fsc',
    )

    logger.info('Wrote the results to %s.', directory)
    print(
        f'{len(subtomograms)} particles, {int(result.kept.sum())} of '
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
            f'symmetry asked for {symmetry}, but the pairs look more like '
            f'{result.detected}'
        )
    if np.isfinite(full.resolution):
        print(
            f'resolution {full.resolution:.1f} A, at {full.pixel_size:.2f} A '
            f'a pixel; aligned up to {result.band_limit:.1f} A, and the halves '
            'were aligned together, so the curve is fair only beyond that'
        )
    else:
        print(
            f'the halves agree to Nyquist, {2 * full.pixel_size:.1f} A; '
            f'aligned up to {result.band_limit:.1f} A'
        )


def _add_formats(commands: Any, shared: argparse.ArgumentParser) -> None:
    parser = commands.add_parser(
        'formats',
        parents=[shared],
        help='List the input layouts that can be read.',
    )
    parser.set_defaults(run=_list_formats)


def _list_formats(device: str | None) -> None:
    del device
    from .io.registry import available

    for name in available():
        print(name)


_COMMANDS: tuple[Callable[[Any, argparse.ArgumentParser], None], ...] = (
    _add_run,
    _add_formats,
)


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command."""
    parser = argparse.ArgumentParser(
        prog='frmony',
        description=(
            'Ab-initio subtomogram averaging by rotational matching and '
            'orthogonal group synchronization.'
        ),
    )
    parser.add_argument(
        '--version',
        action='version',
        version=metadata.version('frmony'),
    )
    commands = parser.add_subparsers(dest='command', required=True)
    shared = _shared_parser()
    for add in _COMMANDS:
        add(commands, shared)
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
    run = arguments.run
    del arguments.run, arguments.command, arguments.verbose
    try:
        run(**vars(arguments))
    except (OSError, ValueError) as error:
        print(f'frmony: {error}', file=sys.stderr)
        return 1
    return 0
