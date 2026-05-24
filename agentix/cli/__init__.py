"""`agentix` command-line interface.

The core CLI intentionally stays narrow: `agentix build` packages a
project into a bundle artifact. Other workflows should expose their own
`console_scripts` entry instead of expanding the central CLI.

Argument parsing is delegated to click — each subcommand is a
`click.Command` registered on the `agentix` group. Click owns `--help`,
error formatting, and option validation; the CLI modules stay free of
routing boilerplate.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import click

from agentix.cli.build import build as _build

_HELP_OPTIONS = {"help_option_names": ["-h", "--help"]}


@click.group(name="agentix", help="Agentix developer CLI.", context_settings=_HELP_OPTIONS)
def cli() -> None:
    """Agentix developer CLI."""


cli.add_command(_build)


def main(argv: Sequence[str] | None = None) -> int:
    """`agentix` console-script entry point — returns the exit code.

    `standalone_mode=False` keeps click from calling `sys.exit` itself,
    so this function composes cleanly from tests and other in-process
    callers. Two failure paths reach here:

      * Click's own usage errors (`UsageError`, including missing
        arguments and `BadParameter`) — we render them to stderr and
        raise `SystemExit(2)`.
      * Command bodies that signal failure with `raise SystemExit(...)`
        — they propagate through untouched.

    Both cases land at Python's default `SystemExit` handling in the
    console-script wrapper, which yields the right process exit code.
    """
    try:
        cli.main(args=argv, prog_name="agentix", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
