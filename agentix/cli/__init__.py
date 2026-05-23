"""`agentix` command-line interface.

The core CLI intentionally stays narrow: `agentix build` packages a
project into a bundle artifact. Other workflows should expose their own
`console_scripts` entry instead of expanding the central CLI.

The CLI deliberately doesn't use argparse subparsers — argparse
intercepts `--help` greedily at the root level, so `agentix build --help`
would never reach `build`'s parser. Manual routing keeps each
subcommand's `--help` intact.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable, Sequence

# Built-in subcommands. Each value names the submodule under
# `agentix.cli` whose `main(argv)` handles the verb.
_COMMANDS: tuple[tuple[str, str], ...] = (("build", "agentix.cli.build"),)


def _first_doc_line(obj: object) -> str:
    """First non-empty line of an object's docstring, or empty."""
    doc = inspect.getdoc(obj) or ""
    return next((line.strip() for line in doc.splitlines() if line.strip()), "")


def _load(module_name: str) -> Callable[[list[str]], int]:
    return importlib.import_module(module_name).main  # type: ignore[no-any-return]


def _describe(module_name: str) -> str:
    """Description for help output. main() docstring → module docstring → empty."""
    mod = importlib.import_module(module_name)
    desc = _first_doc_line(getattr(mod, "main", None))
    return desc or _first_doc_line(mod)


def _print_root_help() -> None:
    print("usage: agentix <command> [args...]\n")
    print("Agentix developer CLI\n")
    print("commands:")
    width = max(len(name) for name, _ in _COMMANDS) + 2
    for name, mod in _COMMANDS:
        print(f"  {name.ljust(width)}{_describe(mod)}")
    print("\nRun `agentix <command> --help` for command-specific options.")


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help()
        return 0
    cmd, *rest = argv
    for name, mod in _COMMANDS:
        if name == cmd:
            return _load(mod)(rest)
    print(f"unknown command: {cmd!r}\n", file=sys.stderr)
    _print_root_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
