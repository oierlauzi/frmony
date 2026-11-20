"""Command line interface.

It is made of commands, ``syncsta COMMAND ...``. A command is a function
that adds its parser to those of the others, with the function that runs it
as the ``run`` default, and is listed in `build_parser`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from typing import Any

_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


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


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command."""
    parser = argparse.ArgumentParser(
        prog='syncsta',
        description=(
            'Ab-initio subtomogram averaging by rotational matching and '
            'orthogonal group synchronization.'
        ),
    )
    parser.add_argument(
        '--version',
        action='version',
        version=metadata.version('syncsta'),
    )
    commands = parser.add_subparsers(dest='command', required=True)
    shared = _shared_parser()
    for add in _COMMANDS:
        add(commands, shared)
    return parser


_COMMANDS: tuple[Callable[[Any, argparse.ArgumentParser], None], ...] = ()


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
        print(f'syncsta: {error}', file=sys.stderr)
        return 1
    return 0
