"""`agentix deploy` — discover plugin-defined deploy subcommands.

The core CLI deliberately owns *no* backend-specific knowledge. Every
provider plugin registers its own `click.Command` via the
`agentix.deploy.commands` entry-point group; this module walks that
group at import time and assembles them into the `agentix deploy`
click `Group`. `uv add agentix-provider-X` is therefore sufficient
for `agentix deploy X --help` to start working — no core edit
required.

The one built-in subcommand is `agentix deploy list`, which surfaces
the discovered backends as structured output (text or JSON) so users
don't have to scroll through `--help` to learn what deploy targets
are installed.

Shared scaffolding for plugin subcommands lives here too:

* `common_options` — Click decorator that adds the `bundle` positional
  plus `--name` / `--platform` / `--format`; plugins layer their own
  engine-specific flags on top.
* `print_deploy_result` — render a `DeployedBundle` to stdout in text
  or JSON (including the shell-comment `hints` block).

Plugin subcommands import those two helpers so every backend's
`--help` and output stay uniform. They're defined above the discovery
machinery so plugin modules loaded via entry points can import them
without hitting a circular-import race.

Discovery failures are tolerated: a single broken plugin (import
error, malformed entry point) is logged and skipped so the rest of
the CLI still works. The user sees the missing subcommand as
"No such command 'X'" from click.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import click

from agentix.provider.base import DeployedBundle, providers

logger = logging.getLogger("agentix.cli.deploy")

F = TypeVar("F", bound=Callable[..., Any])


# ── Public scaffolding for plugin deploy subcommands ─────────────────
#
# Defined ahead of the discovery machinery so plugin modules loaded by
# `_make_deploy_group()` (during the `deploy = _make_deploy_group()`
# binding below) can import these without tripping over a circular
# import — when the docker plugin runs
# `from agentix.cli.deploy import common_options, print_deploy_result`
# those names must already exist on the partially-initialised module.


def common_options(f: F) -> F:
    """Attach the `bundle` argument + `--name` / `--platform` / `--format`
    options every `agentix deploy <backend>` subcommand shares.

    Click decorators apply bottom-up: outermost decorator (here `bundle`)
    becomes the first positional in the rendered help, then `--name`,
    `--platform`, `--format`. Subcommand functions receive everything as
    kwargs (`bundle`, `name`, `platform`, `output_format`), so plugins
    can place their own engine-specific decorators above or below this
    one without worrying about parameter ordering.
    """
    f = click.option(
        "--format",
        "output_format",
        type=click.Choice(["text", "json"]),
        default="text",
        help="Output format: `text` (default, `key -> value` lines) or `json`.",
    )(f)
    f = click.option(
        "--platform",
        default=None,
        metavar="PLATFORM",
        help="Optional bundle runtime platform; defaults to the bundle manifest's value.",
    )(f)
    f = click.option(
        "-n",
        "--name",
        default=None,
        metavar="NAME[:TAG]",
        help="Optional backend bundle label.",
    )(f)
    f = click.argument("bundle", type=click.Path(path_type=Path))(f)
    return f


def print_deploy_result(result: DeployedBundle, *, output_format: str) -> None:
    """Render a `DeployedBundle` to stdout.

    Text form: `bundle -> <ref>`, then `platform -> <plat>` (when set),
    then sorted `metadata` entries, then `hints` rendered shell-comment
    style (`# label\\n<command>`) so the user can copy-paste the whole
    block into a terminal and only the command lines execute. The hints
    section is omitted entirely when the provider supplied none, so the
    output stays tight for the common case.

    JSON form: a single object with `bundle` / `platform` / `metadata` /
    `hints` keys — machine-readable for `agentix deploy ... --format json`
    | `jq -r .bundle` style pipelines.
    """
    if output_format == "json":
        print(
            json.dumps(
                {
                    "bundle": result.bundle,
                    "platform": result.platform,
                    "metadata": result.metadata,
                    "hints": result.hints,
                }
            )
        )
        return

    print(f"bundle -> {result.bundle}")
    if result.platform:
        print(f"platform -> {result.platform}")
    for key, value in sorted(result.metadata.items()):
        print(f"{key} -> {value}")
    if result.hints:
        print()
        for label, command in result.hints.items():
            print(f"# {label}")
            print(command)


# ── Discovery machinery + built-in `list` subcommand ─────────────────

# `list` is owned by core — discovery refuses to overwrite it so a plugin
# can't accidentally (or maliciously) shadow the introspection command.
_RESERVED_SUBCOMMANDS = frozenset({"list"})

_DEPLOY_HELP = """\
Deploy an Agentix bundle tar to a provider backend.

`agentix build` always writes a portable bundle tar (`manifest.json + nix/`).
Each provider plugin contributes its own `agentix deploy <name>` subcommand
that turns that tar into the backend-native reference `SandboxConfig.bundle`
should use:

\b
  - Local backends (`docker`, `podman`) extract the tar into a
    content-addressed host cache; the ref is the cache path.
  - Managed services (when wired: e2b, modal, daytona, fly) upload the
    tar and register it as a template / volume / image; the ref is the
    service-side ID.

\b
Use `agentix deploy <backend> --help` for backend-specific flags. Run
`agentix deploy` with no args to see which backends are installed.

\b
Examples:
    agentix deploy docker dist/hello.bundle.tar
    agentix deploy podman dist/hello.bundle.tar --run-arg --runtime=crun
    agentix deploy docker dist/hello.bundle.tar --format json
"""


def _deploy_command_entry_points() -> list[importlib.metadata.EntryPoint]:
    """Walk the `agentix.deploy.commands` entry-point group."""
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group="agentix.deploy.commands"))
    return list(eps.get("agentix.deploy.commands", []))  # type: ignore[attr-defined]  # pragma: no cover


def _make_deploy_group() -> click.Group:
    """Build the `agentix deploy` group by discovering plugin subcommands.

    Walks the `agentix.deploy.commands` entry-point group via
    `importlib.metadata`. Each entry must resolve to a `click.Command`
    (typically a `@click.command(...)` function); the entry-point name
    becomes the subcommand name. A loader that raises is logged and
    skipped so a single broken plugin can't take the whole CLI down.
    Names in `_RESERVED_SUBCOMMANDS` (currently just `list`) are owned
    by core and refused — a plugin can't shadow `agentix deploy list`.
    """
    group = click.Group(
        name="deploy",
        help=_DEPLOY_HELP,
        short_help="Deploy a bundle tar to a provider backend.",
        context_settings={"help_option_names": ["-h", "--help"]},
        invoke_without_command=False,
    )

    group.add_command(deploy_list_cmd, name="list")

    for ep in _deploy_command_entry_points():
        if ep.name in _RESERVED_SUBCOMMANDS:
            logger.warning(
                "deploy plugin %r tried to register reserved subcommand name; skipping",
                ep.name,
            )
            continue
        try:
            cmd = ep.load()
        except Exception as exc:
            logger.warning("deploy plugin %r failed to load: %s", ep.name, exc)
            continue
        if not isinstance(cmd, click.Command):
            logger.warning(
                "deploy plugin %r resolved to %r (expected click.Command); skipping",
                ep.name,
                type(cmd).__name__,
            )
            continue
        group.add_command(cmd, name=ep.name)

    return group


@click.command(
    "list",
    short_help="List installed deploy backends.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format: `text` (default, one backend per line) or `json` (list of objects).",
)
def deploy_list_cmd(output_format: str) -> None:
    """Show every provider plugin that registered an `agentix deploy <name>`
    subcommand, plus its source dist + short help. Also names providers that
    are installed (in the `agentix.provider` registry) but did *not* contribute
    a deploy CLI command — so users can tell "is e2b just not yet wired for
    deploy?" from "is e2b not installed?".
    """
    entries: list[dict[str, str | None]] = []
    for ep in sorted(_deploy_command_entry_points(), key=lambda e: e.name):
        if ep.name in _RESERVED_SUBCOMMANDS:
            continue
        dist = getattr(ep, "dist", None)
        dist_name = getattr(dist, "name", None) if dist else None
        dist_version = getattr(dist, "version", None) if dist else None
        summary: str | None = None
        try:
            cmd = ep.load()
            if isinstance(cmd, click.Command):
                summary = cmd.short_help or cmd.help or None
        except Exception as exc:
            summary = f"ERROR: {type(exc).__name__}: {exc}"
        entries.append(
            {
                "name": ep.name,
                "source": dist_name,
                "version": dist_version,
                "summary": summary,
            }
        )

    deploy_names = {entry["name"] for entry in entries}
    installed_providers = sorted(set(providers().all()) | set(providers().errors()))
    no_deploy = [name for name in installed_providers if name not in deploy_names]

    if output_format == "json":
        print(json.dumps({"deploy": entries, "providers_without_deploy": no_deploy}))
        return

    if not entries:
        print("no deploy backends installed.")
        print("install one: pip install agentix-provider-docker  (or another agentix.deploy.commands plugin)")
    else:
        name_w = max((len(entry["name"] or "") for entry in entries), default=0)
        source_w = max((len(_format_source(entry)) for entry in entries), default=0)
        for entry in entries:
            name = entry["name"] or ""
            source = _format_source(entry)
            summary = entry["summary"] or ""
            print(f"{name:<{name_w}}  {source:<{source_w}}  {summary}")

    if no_deploy:
        print()
        print(
            f"{len(no_deploy)} provider(s) installed without a deploy subcommand "
            f"(see `agentix plugin list`):"
        )
        print(f"  {', '.join(no_deploy)}")


def _format_source(entry: dict[str, str | None]) -> str:
    source = entry.get("source") or "(local)"
    version = entry.get("version")
    return f"{source}@{version}" if version else source


deploy = _make_deploy_group()


def main(argv: Sequence[str] | None = None) -> int:
    """`agentix deploy` standalone entry point — returns the exit code."""
    try:
        deploy.main(args=argv, prog_name="agentix deploy", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    return 0


__all__ = ["common_options", "deploy", "main", "print_deploy_result"]
