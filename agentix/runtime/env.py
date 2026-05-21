"""Runtime environment helpers.

Agentix may prepend bundle-owned paths while booting the runtime or its
worker. User-facing subprocesses should be able to run without those
bundle paths leaking into normal command lookup.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

AGENTIX_ADDED_PATH = "AGENTIX_ADDED_PATH"
AGENTIX_ADDED_LD_LIBRARY_PATH = "AGENTIX_ADDED_LD_LIBRARY_PATH"

_TRACKING_PREFIX = "AGENTIX_ADDED_"


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


def get_env_without_agentix(
    extra: Mapping[str, str] | None = None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment for user subprocesses without Agentix-added paths.

    The helper only subtracts entries that Agentix explicitly recorded in
    `AGENTIX_ADDED_*`. It intentionally does not remove arbitrary `/nix`
    paths, because a task image may itself be Nix-based.
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

    if extra:
        env.update(extra)
    return env


__all__ = [
    "AGENTIX_ADDED_LD_LIBRARY_PATH",
    "AGENTIX_ADDED_PATH",
    "get_env_without_agentix",
]
