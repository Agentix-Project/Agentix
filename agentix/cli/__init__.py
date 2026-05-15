"""`agentix` command-line interface.

A small argparse-based dispatcher for the developer-facing tools:

    agentix build  primitives/bash               # build a closure image
    agentix check  primitives/                   # stub ↔ impl signature drift

Subcommands live in sibling modules so they can also be invoked
directly (`python -m agentix.cli.build …`). The CLI is intentionally
thin — most logic lives in those subcommand modules.

`tools/gen_manifest.py` is **not** exposed as a subcommand because it
runs inside the closure's nix build environment, which doesn't have
the `agentix` package available. Keep it standalone.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix",
        description="Agentix developer CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Lazy import — keeps `agentix --help` fast and avoids pulling docker /
    # nix / pydantic schema generation into every CLI invocation.
    build_p = subparsers.add_parser("build", help="build a closure image")
    build_p.set_defaults(_run=lambda args, rest: _run_build(rest))

    check_p = subparsers.add_parser("check", help="stub ↔ impl signature drift")
    check_p.set_defaults(_run=lambda args, rest: _run_check(rest))

    # `parse_known_args` lets each subcommand define its own flags without
    # forcing every option to be declared up front in the root parser.
    args, rest = parser.parse_known_args(argv)
    return args._run(args, rest)


def _run_build(rest: list[str]) -> int:
    from agentix.cli.build import main as build_main
    return build_main(rest)


def _run_check(rest: list[str]) -> int:
    from agentix.cli.check import main as check_main
    return check_main(rest)


if __name__ == "__main__":
    sys.exit(main())
