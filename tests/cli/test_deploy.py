"""Tests for the `agentix deploy` discovery shell + shared rendering.

The core CLI now owns *no* per-backend logic — provider plugins each
register their own `click.Command` via the `agentix.deploy.commands`
entry-point group. These tests cover:

  * the discovery mechanism (subcommands appear in `agentix deploy --help`)
  * the shared output rendering (`print_deploy_result` text + JSON,
    including the shell-comment `hints` block)
  * graceful handling of unknown / broken plugin entries

Per-backend deploy CLI behavior lives in each plugin's own tests (e.g.
`plugins/providers/docker/tests/test_deploy_cli.py`).
"""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

import agentix.cli.deploy as deploy_mod
from agentix.cli.deploy import print_deploy_result
from agentix.provider.base import DeployedBundle

# ── discovery: plugin subcommands appear in the deploy group ─────────


def test_deploy_group_includes_installed_provider_subcommands() -> None:
    """The docker plugin (workspace-installed) registers `docker` and
    `podman` deploy subcommands; both must resolve through the group's
    lazy discovery (`list_commands` / `get_command`)."""
    ctx = click.Context(deploy_mod.deploy)
    names = deploy_mod.deploy.list_commands(ctx)
    assert "docker" in names
    assert "podman" in names
    assert isinstance(deploy_mod.deploy.get_command(ctx, "docker"), click.Command)
    assert isinstance(deploy_mod.deploy.get_command(ctx, "podman"), click.Command)


def test_importing_provider_module_has_no_circular_import() -> None:
    """Regression: a deploy-command entry point points back into the
    provider module that imports `agentix.cli.deploy` (for
    `common_options` / `print_deploy_result`). The deploy group must be
    built lazily — eager `ep.load()` at module import re-enters the
    half-initialized provider module and fails with a partial-init
    `AttributeError`, which the registry would swallow as a "failed to
    load" warning. Importing the provider in a fresh interpreter must
    therefore be warning-free.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::Warning",
            "-c",
            "import agentix.provider.docker; "
            "import agentix.cli.deploy as d; "
            "assert d.deploy.get_command(__import__('click').Context(d.deploy), 'docker') is not None",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # The registry logs circular-import failures via `logging`, not
    # `warnings`, so also assert the tell-tale string never appears.
    assert "circular import" not in (proc.stderr + proc.stdout)
    assert "failed to load" not in (proc.stderr + proc.stdout)


def test_deploy_unknown_backend_reports_no_such_command(tmp_path) -> None:
    """Click owns the `no such command` error path now that every
    backend is a click subcommand — the discovery shell adds nothing."""
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    result = CliRunner().invoke(deploy_mod.deploy, ["bogus-backend", str(bundle)])

    assert result.exit_code != 0
    # Click's wording varies slightly between versions but the message
    # always names the bad command. `result.output` includes both
    # streams in newer Click; `result.stderr` only exists when
    # they were captured separately.
    assert "bogus-backend" in result.output


def test_deploy_help_lists_subcommands() -> None:
    """`agentix deploy --help` prints the subcommand list assembled from
    installed plugins (here: docker + podman, from agentix-provider-docker)
    plus the built-in `list` discovery subcommand."""
    result = CliRunner().invoke(deploy_mod.deploy, ["--help"])
    assert result.exit_code == 0, result.output
    assert "docker" in result.output
    assert "podman" in result.output
    assert "list" in result.output


# ── built-in `list` discovery subcommand ─────────────────────────────


def test_deploy_list_text_output_includes_installed_backends() -> None:
    """`agentix deploy list` text output shows one row per installed
    deploy subcommand with `<name>  <source>@<version>  <short_help>`."""
    result = CliRunner().invoke(deploy_mod.deploy, ["list"])
    assert result.exit_code == 0, result.output
    out = result.output

    # docker + podman come from the workspace-installed agentix-provider-docker.
    assert "docker" in out
    assert "podman" in out
    assert "agentix-provider-docker" in out


def test_deploy_list_text_output_separates_providers_without_deploy() -> None:
    """Providers registered in the `agentix.provider` registry but without a
    matching `agentix.deploy.commands` entry are surfaced under a separate
    'without a deploy subcommand' section — distinguishing 'not installed' from
    'installed but not yet wired'."""
    result = CliRunner().invoke(deploy_mod.deploy, ["list"])
    assert result.exit_code == 0, result.output
    out = result.output

    # e2b ships SandboxProvider class but no deploy command — must appear here.
    assert "without a deploy subcommand" in out
    assert "e2b" in out


def test_deploy_list_json_format_emits_structured_payload() -> None:
    result = CliRunner().invoke(deploy_mod.deploy, ["list", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)

    assert "deploy" in data
    assert "providers_without_deploy" in data

    names = {entry["name"] for entry in data["deploy"]}
    assert {"docker", "podman"} <= names
    for entry in data["deploy"]:
        assert set(entry.keys()) >= {"name", "source", "version", "summary"}
        if entry["name"] in {"docker", "podman"}:
            assert entry["source"] == "agentix-provider-docker"
            assert entry["version"]  # non-empty


def test_deploy_list_cannot_be_shadowed_by_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin that registers an `agentix.deploy.commands` entry named
    `list` must NOT replace the core's built-in introspection command."""
    import importlib.metadata as md

    class _FakeEntryPoint:
        name = "list"
        value = "evil_plugin:bad_cmd"
        dist = None

        def load(self):  # pragma: no cover — should never be called
            return click.Command("list")

    # Patch only the deploy-commands selector; everything else stays real.
    original_entry_points = md.entry_points

    def fake_entry_points(*args, **kwargs):
        eps = original_entry_points(*args, **kwargs)
        if hasattr(eps, "select"):
            real_selected = list(eps.select(group="agentix.deploy.commands"))
        else:  # pragma: no cover
            real_selected = list(eps.get("agentix.deploy.commands", []))
        original_select = eps.select if hasattr(eps, "select") else None

        class _Wrapped:
            def select(self_inner, *, group: str):
                if group == "agentix.deploy.commands":
                    return [_FakeEntryPoint(), *real_selected]
                return original_select(group=group) if original_select else []

        return _Wrapped()

    monkeypatch.setattr(deploy_mod.importlib.metadata, "entry_points", fake_entry_points)

    group = deploy_mod._make_deploy_group()
    ctx = click.Context(group)

    # `list` resolves to the built-in command, never the plugin's — the
    # reserved name is owned by core and the plugin entry is skipped (its
    # `load()` is never called). `list` appears exactly once in the group.
    assert group.get_command(ctx, "list") is deploy_mod.deploy_list_cmd
    assert group.list_commands(ctx).count("list") == 1


# ── rendering: print_deploy_result text + JSON ───────────────────────


def _bundle(**overrides) -> DeployedBundle:
    defaults = {
        "bundle": "/tmp/agentix-runtime-pytest/sha256-abc/",
        "platform": "linux/amd64",
        "metadata": {"cache": "/tmp/agentix-runtime-pytest/sha256-abc/", "name": "demo:1.0.0"},
        "hints": {},
    }
    defaults.update(overrides)
    return DeployedBundle(**defaults)


def test_print_deploy_result_text_renders_bundle_platform_and_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_deploy_result(_bundle(), output_format="text")
    out = capsys.readouterr().out
    assert "bundle -> /tmp/agentix-runtime-pytest/sha256-abc/" in out
    assert "platform -> linux/amd64" in out
    assert "cache -> /tmp/agentix-runtime-pytest/sha256-abc/" in out
    assert "name -> demo:1.0.0" in out


def test_print_deploy_result_text_renders_hints_shell_comment_style(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hints render as `# label\\n<command>` so the whole block is
    copy-pasteable into a terminal — only the command lines execute."""
    result = _bundle(
        hints={
            "inspect contents": "ls -la /tmp/.../nix/",
            "remove from cache": "rm -rf /tmp/...",
        },
    )
    print_deploy_result(result, output_format="text")
    out = capsys.readouterr().out
    assert "# inspect contents\nls -la /tmp/.../nix/" in out
    assert "# remove from cache\nrm -rf /tmp/..." in out
    # Provider order is preserved (dict insertion order, Python 3.7+).
    assert out.index("# inspect contents") < out.index("# remove from cache")


def test_print_deploy_result_text_omits_hint_section_when_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Providers that don't surface hints get no trailing blank line or
    `#` block — the output stays tight for the common case."""
    print_deploy_result(_bundle(), output_format="text")
    out = capsys.readouterr().out
    assert "#" not in out
    assert not out.endswith("\n\n")


def test_print_deploy_result_text_omits_platform_when_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_deploy_result(_bundle(platform=None), output_format="text")
    out = capsys.readouterr().out
    assert "platform ->" not in out


def test_print_deploy_result_json_emits_machine_readable_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _bundle(hints={"remove": "rm -rf /foo"})
    print_deploy_result(result, output_format="json")
    data = json.loads(capsys.readouterr().out)
    assert data["bundle"] == "/tmp/agentix-runtime-pytest/sha256-abc/"
    assert data["platform"] == "linux/amd64"
    assert data["metadata"]["cache"] == "/tmp/agentix-runtime-pytest/sha256-abc/"
    assert data["hints"] == {"remove": "rm -rf /foo"}
