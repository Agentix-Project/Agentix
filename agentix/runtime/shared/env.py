"""Bundle runtime env contract — paths, env vars, and helpers.

This module is the single source of truth for the *bundle's runtime
contract*: where the runtime lives inside the container, what each
path-style env var is prepended with, and the `AGENTIX_ADDED_*`
tracking convention that lets user subprocesses subtract Agentix's
contribution back out.

Both sides import from here:

  * host side  — `agentix.cli.build.bundle` bakes the env into the
    manifest; provider backends (`agentix.provider.docker`,
    `agentix.provider.apptainer`, ...) exec the entrypoint and mount
    the runtime tree.
  * sandbox side — `agentix.runtime.server.worker.client` prepends the
    paths when spawning user subprocesses; `agentix.bash` falls back
    to the bundled bash.

Lives under `runtime.shared` because — like `MAX_MESSAGE_BYTES`,
`RemoteCallable`, and the wire `models` — it must be importable
without pulling in `runtime/client/` or `runtime/server/`. Code under
`agentix.cli`, `agentix.utils`, and `agentix.provider` reaches in here
freely; nothing here reaches back.

Non-Python mirrors of the same paths live in:

  * `agentix/builder/Dockerfile`        — bakes the env into the image
  * `agentix/builder/bootstrap.sh`      — prepends the paths at boot
  * `agentix/builder/flake.nix`         — names the merged runtime tree

Those files can't import from this module (they're shell / Nix / a
Dockerfile), so any rename here must update them by hand.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

BUNDLE_NIX_ROOT = "/nix"
"""Fixed in-container mount point for the bundle's Nix runtime tree."""

BUNDLE_RUNTIME_ROOT = f"{BUNDLE_NIX_ROOT}/runtime"
"""Runtime sub-tree under the fixed in-container `/nix` mount."""

BUNDLE_RUNTIME_BIN = f"{BUNDLE_RUNTIME_ROOT}/bin"
"""`symlinkJoin` of every closure's `bin/` — every system binary in the
bundle (bash, claude, rg, ...) resolves under this path."""

BUNDLE_RUNTIME_LIB = f"{BUNDLE_RUNTIME_ROOT}/lib"
"""`symlinkJoin` of every closure's `lib/` — what `LD_LIBRARY_PATH` and
`LIBRARY_PATH` point at."""

BUNDLE_RUNTIME_INCLUDE = f"{BUNDLE_RUNTIME_ROOT}/include"
"""`symlinkJoin` of every closure's `include/` — what `CPATH` and the
language-specific include paths point at."""

BUNDLE_RUNTIME_VENV = f"{BUNDLE_RUNTIME_ROOT}/venv"
"""The uv venv `agentix build` materializes — holds every Python dep
from the project's `[project].dependencies`."""

BUNDLE_RUNTIME_VENV_BIN = f"{BUNDLE_RUNTIME_VENV}/bin"
"""`bin/` of the bundle venv — where `python`, `uvicorn`, and any
console-script entry points from the project's deps land."""

BUNDLE_RUNTIME_BASH = f"{BUNDLE_RUNTIME_BIN}/bash"
"""Concrete bash that ships with every bundle (`runtime-basic`'s
`agentix.nix` entry contributes it). User code that wants a known-good
shell — independent of whatever the task image's `/bin/sh` is — runs
this binary."""

BUNDLE_RUNTIME_PKGCONFIG_DIRS: tuple[str, ...] = (
    f"{BUNDLE_RUNTIME_LIB}/pkgconfig",
    f"{BUNDLE_RUNTIME_ROOT}/share/pkgconfig",
)
"""`pkg-config` lookup dirs the bundle contributes to `PKG_CONFIG_PATH`."""

BUNDLE_RUNTIME_ENTRYPOINT = f"{BUNDLE_RUNTIME_ROOT}/bootstrap.sh"
"""Path inside the bundle that provider backends exec as the
container entry point. The script preps Nix-managed runtime PATHs and
launches uvicorn against `agentix.runtime.server.app:app`. Backends
should never have to know what the script does — only that it exists
at this path."""

BUNDLE_RUNTIME_PATH_ENTRIES: dict[str, tuple[str, ...]] = {
    "PATH": (BUNDLE_RUNTIME_VENV_BIN, BUNDLE_RUNTIME_BIN),
    "LD_LIBRARY_PATH": (BUNDLE_RUNTIME_LIB,),
    "LIBRARY_PATH": (BUNDLE_RUNTIME_LIB,),
    "CPATH": (BUNDLE_RUNTIME_INCLUDE,),
    "C_INCLUDE_PATH": (BUNDLE_RUNTIME_INCLUDE,),
    "CPLUS_INCLUDE_PATH": (BUNDLE_RUNTIME_INCLUDE,),
    "PKG_CONFIG_PATH": BUNDLE_RUNTIME_PKGCONFIG_DIRS,
    "CMAKE_PREFIX_PATH": (BUNDLE_RUNTIME_ROOT,),
}
"""Canonical path-style env vars the bundle contributes to.

Every consumer (`agentix build` bakes the joined form into the
manifest; the worker prepends them when spawning the sandbox subprocess;
`bootstrap.sh` mirrors them in shell) reads from this one table so the
runtime contract has a single source of truth."""

BUNDLE_RUNTIME_ENV: dict[str, str] = {name: ":".join(entries) for name, entries in BUNDLE_RUNTIME_PATH_ENTRIES.items()}
"""`BUNDLE_RUNTIME_PATH_ENTRIES` flattened with `:` — the on-disk
manifest contract. Linux-only by construction (the bundle targets a
Linux container), so the `:` separator is hardcoded rather than
`os.pathsep`."""

BIND_PORT_ENV = "AGENTIX_BIND_PORT"
"""Env var the bundle's bootstrap script reads to choose its listen
port. Backends pick a free host port and pass it via this name."""

BIND_HOST_ENV = "AGENTIX_BIND_HOST"
"""Env var the bundle's bootstrap script reads to choose its listen host."""

AGENTIX_ADDED_PATH = "AGENTIX_ADDED_PATH"
AGENTIX_ADDED_LD_LIBRARY_PATH = "AGENTIX_ADDED_LD_LIBRARY_PATH"

_TRACKING_PREFIX = "AGENTIX_ADDED_"

AGENTIX_SAVED_PREFIX = "AGENTIX_SAVED_"
"""Prefix for vars the worker spawn STRIPPED (rather than prepended to).

The worker removes interpreter-hostile vars (``PYTHONPATH``, ``LD_PRELOAD``,
...) from its own environment, but the task image legitimately owns them —
so the spawn records each one under ``AGENTIX_SAVED_<NAME>`` and
:func:`get_env_without_agentix` restores them for task subprocesses."""


def _split_path(value: str) -> list[str]:
    return value.split(os.pathsep) if value else []


def _remove_path_entries(value: str, entries: str) -> str:
    remove = set(_split_path(entries))
    if not remove:
        return value
    return os.pathsep.join(entry for entry in _split_path(value) if entry not in remove)


def _target_var(tracking_var: str) -> str | None:
    if not tracking_var.startswith(_TRACKING_PREFIX):
        return None
    name = tracking_var.removeprefix(_TRACKING_PREFIX)
    return name or None


def image_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The task image's own environment, ready to pass as a subprocess env.

    :func:`get_env_without_agentix` plus a shell convenience: if the image
    ships a ``~/.bashrc`` (per the RESULT env's ``HOME``) and nothing set
    ``BASH_ENV``, point ``BASH_ENV`` at it so a non-interactive ``bash -c``
    sources the image's shell setup. Canonically re-exported as
    ``agentix.bash.image_env`` — implemented here because this module owns
    the ``AGENTIX_ADDED_*``/``AGENTIX_SAVED_*`` contract and is importable
    from every plugin without a cross-plugin dependency.

    Reads the calling process's live environment, so it is only meaningful
    INSIDE the sandbox; a host driving the sandbox fetches it over the wire
    (``env = await c.remote(image_env)``) rather than evaluating it host-side.
    """
    env = get_env_without_agentix(extra)
    home = env.get("HOME") or os.path.expanduser("~")
    bashrc = os.path.join(home, ".bashrc")
    if "BASH_ENV" not in env and os.path.isfile(bashrc):
        env["BASH_ENV"] = bashrc
    return env


def get_env_without_agentix(
    extra: Mapping[str, str] | None = None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment for user subprocesses without Agentix-added paths.

    The helper only subtracts entries that Agentix explicitly recorded in
    `AGENTIX_ADDED_*` (it intentionally does not remove arbitrary `/nix`
    paths, because a task image may itself be Nix-based), and restores vars
    the worker spawn stripped and recorded under `AGENTIX_SAVED_*`. A live
    value wins over its saved snapshot — presence means something set it
    after the spawn, which is more recent intent than the pre-strip original.
    """

    env = dict(os.environ if base is None else base)

    tracking_vars = [name for name in env if name.startswith(_TRACKING_PREFIX)]
    for tracking_var in tracking_vars:
        target = _target_var(tracking_var)
        if target is None:
            continue
        value = _remove_path_entries(env.get(target, ""), env.get(tracking_var, ""))
        if value:
            env[target] = value
        else:
            env.pop(target, None)

    for name in tracking_vars:
        env.pop(name, None)

    saved_vars = [name for name in env if name.startswith(AGENTIX_SAVED_PREFIX)]
    for saved_var in saved_vars:
        target = saved_var.removeprefix(AGENTIX_SAVED_PREFIX)
        if target and target not in env:
            env[target] = env[saved_var]
    for name in saved_vars:
        env.pop(name, None)

    if extra:
        env.update(extra)
    return env


__all__ = [
    "AGENTIX_ADDED_LD_LIBRARY_PATH",
    "AGENTIX_ADDED_PATH",
    "AGENTIX_SAVED_PREFIX",
    "BIND_HOST_ENV",
    "BIND_PORT_ENV",
    "BUNDLE_NIX_ROOT",
    "BUNDLE_RUNTIME_BASH",
    "BUNDLE_RUNTIME_BIN",
    "BUNDLE_RUNTIME_ENTRYPOINT",
    "BUNDLE_RUNTIME_ENV",
    "BUNDLE_RUNTIME_INCLUDE",
    "BUNDLE_RUNTIME_LIB",
    "BUNDLE_RUNTIME_PATH_ENTRIES",
    "BUNDLE_RUNTIME_PKGCONFIG_DIRS",
    "BUNDLE_RUNTIME_ROOT",
    "BUNDLE_RUNTIME_VENV",
    "BUNDLE_RUNTIME_VENV_BIN",
    "get_env_without_agentix",
    "image_env",
]
