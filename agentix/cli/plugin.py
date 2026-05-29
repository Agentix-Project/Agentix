"""`agentix plugin` — inspect installed Agentix plugins.

Read-only introspection. Today it surfaces the deployment-backend
registry (the `agentix.provider` entry-point group): which backends a
`pip install` made available, and which failed to load — so a broken
plugin is visible here instead of silently missing until you select it.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import click

from agentix.provider.base import providers

_HELP_OPTIONS = {"help_option_names": ["-h", "--help"]}


@click.group(name="plugin", help="Inspect installed Agentix plugins.", context_settings=_HELP_OPTIONS)
def plugin() -> None:
    """Inspect installed Agentix plugins."""


@plugin.command(name="list", short_help="List installed deployment backends and their load status.")
def list_() -> int:
    """List deployment backends discovered via the `agentix.provider` group.

    Each backend prints `ok` with its distribution, or `ERROR` with the
    exception that broke its entry-point load — so a misbuilt plugin shows
    up here rather than only when you try to use it.
    """
    registry = providers()
    loaded = registry.all()
    errors = registry.errors()
    sources = registry.sources()

    names = sorted(set(loaded) | set(errors))
    if not names:
        print("no deployment backends installed (e.g. `pip install agentix-deployment-docker`)")
        return 0

    for name in names:
        if name in loaded:
            source = sources.get(name)
            label = f"  ({source.label()})" if source else ""
            print(f"{name:<14} ok{label}")
        else:
            exc = errors[name]
            print(f"{name:<14} ERROR  {type(exc).__name__}: {exc}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        plugin.main(args=argv, prog_name="agentix plugin", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    return 0


__all__ = ["main", "plugin"]
