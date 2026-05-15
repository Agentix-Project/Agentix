"""`agentix` command-line interface — dynamically discovered subcommands.

Subcommands are entry-point plugins under the `agentix.cli` group. Each
entry is a `module:main` callable that takes the post-subcommand argv
and returns an exit code. The framework's builtins (`build`, `install`,
`deploy`, `check`, `plugins`) ship via the framework's own pyproject;
downstream packages add their own with one line:

```toml
[project.entry-points."agentix.cli"]
my-cmd = "my_pkg:main"
```

A `pip install` makes `agentix my-cmd` work with no framework changes.

The dispatcher deliberately doesn't use argparse subparsers — argparse
intercepts `--help` greedily at the root level, so `agentix build --help`
would never reach `build`'s parser. Manual dispatch keeps each
subcommand's `--help` intact.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import sys
from collections.abc import Callable, Sequence

_CLI_GROUP = "agentix.cli"


def _iter_entry_points():
    eps = importlib.metadata.entry_points()
    return (
        list(eps.select(group=_CLI_GROUP))
        if hasattr(eps, "select")
        else list(eps.get(_CLI_GROUP, []))  # type: ignore[attr-defined]
    )


def _first_doc_line(obj: object) -> str:
    """First non-empty line of an object's docstring, or empty."""
    doc = inspect.getdoc(obj) or ""
    return next((line.strip() for line in doc.splitlines() if line.strip()), "")


def _discover_commands() -> dict[str, tuple[str, Callable[[list[str]], int]]]:
    """Walk the `agentix.cli` group; return `{name: (description, main)}`.

    Descriptions fall back through: the `main()`'s own docstring →
    its module's docstring → empty. Import errors are logged but don't
    crash the CLI — the broken subcommand is just absent. Use
    `agentix plugins --verbose` to see why.
    """
    out: dict[str, tuple[str, Callable[[list[str]], int]]] = {}
    for ep in _iter_entry_points():
        try:
            main = ep.load()
        except Exception as exc:
            print(
                f"warning: `agentix {ep.name}` failed to load: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        desc = _first_doc_line(main)
        if not desc:
            import importlib
            mod_name = getattr(main, "__module__", None)
            if mod_name:
                try:
                    mod = importlib.import_module(mod_name)
                    desc = _first_doc_line(mod)
                except Exception:
                    pass
        out[ep.name] = (desc, main)
    return out


def _print_root_help(commands: dict[str, tuple[str, Callable[[list[str]], int]]]) -> None:
    print("usage: agentix <command> [args...]\n")
    print("Agentix developer CLI\n")
    if not commands:
        print("(no `agentix.cli` entry points installed)")
        return
    print("commands:")
    width = max(len(c) for c in commands) + 2
    for cmd, (desc, _main) in sorted(commands.items()):
        print(f"  {cmd.ljust(width)}{desc}")
    print("\nRun `agentix <command> --help` for command-specific options.")


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    commands = _discover_commands()
    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help(commands)
        return 0
    cmd, *rest = argv
    entry = commands.get(cmd)
    if entry is None:
        print(f"unknown command: {cmd!r}\n", file=sys.stderr)
        _print_root_help(commands)
        return 2
    _desc, main_fn = entry
    return main_fn(rest)


if __name__ == "__main__":
    sys.exit(main())
