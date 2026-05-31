"""`agentix plugin` — inspect installed Agentix plugins.

Read-only introspection over both plugin axes:

  * **providers** — host-side sandbox backends discovered via the
    `agentix.provider` entry-point group (e.g. `agentix-provider-docker`).
  * **nix closures** — sandbox-side system-dep closures discovered via
    the `agentix.nix` entry-point group (e.g. `agentix-runtime-basic`
    ships `bash` + `files`).

A broken plugin in either axis surfaces here with the loader exception,
so misbuilt installs are visible without having to actually use them.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from collections.abc import Sequence

import click

from agentix.provider.base import providers

_HELP_OPTIONS = {"help_option_names": ["-h", "--help"]}

# Entry-point group sandbox-side nix-closure plugins register under.
# Mirrors `agentix.cli.build.closures.NIX_ENTRY_POINT_GROUP`; kept as a
# module-level constant so tests can stub the group lookup without
# importing the build step.
NIX_GROUP = "agentix.nix"

_PROVIDER_HEADER = "providers (host-side sandbox backends):"
_NIX_HEADER = "nix closures (sandbox-side system deps):"


@click.group(name="plugin", help="Inspect installed Agentix plugins.", context_settings=_HELP_OPTIONS)
def plugin() -> None:
    """Inspect installed Agentix plugins."""


@plugin.command(name="list", short_help="List installed Agentix plugins and their load status.")
def list_() -> int:
    """List provider backends + nix closures discovered in the current env."""
    registry = providers()
    loaded = registry.all()
    errors = registry.errors()
    sources = registry.sources()
    provider_names = sorted(set(loaded) | set(errors))

    nix_eps = sorted(md.entry_points(group=NIX_GROUP), key=lambda ep: ep.name)

    if not provider_names and not nix_eps:
        print("no Agentix plugins installed")
        print()
        print(f"{_PROVIDER_HEADER} pip install agentix-provider-docker (or -daytona, -e2b, -apptainer)")
        print(f"{_NIX_HEADER} pip install agentix-runtime-basic (or another agentix.nix plugin)")
        return 0

    print(_PROVIDER_HEADER)
    if provider_names:
        for name in provider_names:
            if name in loaded:
                source = sources.get(name)
                label = f"  ({source.label()})" if source else ""
                print(f"  {name:<14} ok{label}")
            else:
                exc = errors[name]
                print(f"  {name:<14} ERROR  {type(exc).__name__}: {exc}")
    else:
        print("  none installed")

    print()
    print(_NIX_HEADER)
    if nix_eps:
        for ep in nix_eps:
            dist = getattr(ep, "dist", None)
            dist_label = ""
            if dist is not None:
                dist_name = getattr(dist, "name", None)
                dist_version = getattr(dist, "version", None)
                if dist_name:
                    dist_label = f"  ({dist_name}@{dist_version or '?'})"
            print(f"  {ep.name:<14} {ep.value}{dist_label}")
    else:
        print("  none installed")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        plugin.main(args=argv, prog_name="agentix plugin", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    return 0


__all__ = ["NIX_GROUP", "main", "md", "plugin"]
