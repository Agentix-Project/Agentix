"""Normalize and translate target platforms for the bundle build.

Agentix bundles always run on Linux containers — `agentix build`
produces a Linux artifact even from a macOS host. The user expresses
the target architecture in Docker's `OS/arch` form (`linux/amd64`,
`linux/arm64`); the Nix builder needs the same architecture in Nix's
`<arch>-<os>` form (`x86_64-linux`, `aarch64-linux`).

This module owns the two-way translation, the alias table for common
shorthands (`amd64`, `arm64`, `x86_64`, …), and the auto-detection of
a sensible default platform from the host's CPU.
"""

from __future__ import annotations

import platform as host_platform

_DOCKER_TO_NIX_SYSTEM = {
    "linux/amd64": "x86_64-linux",
    "linux/arm64": "aarch64-linux",
}

_PLATFORM_ALIASES = {
    "linux/amd64": "linux/amd64",
    "linux/x86-64": "linux/amd64",
    "amd64": "linux/amd64",
    "x86-64": "linux/amd64",
    "linux/arm64": "linux/arm64",
    "linux/arm64/v8": "linux/arm64",
    "linux/aarch64": "linux/arm64",
    "arm64": "linux/arm64",
    "aarch64": "linux/arm64",
}


def normalize_platform(value: str) -> str:
    """Normalize a user platform into Docker's OS/arch form."""
    key = value.strip().lower().replace("_", "-")
    platform = _PLATFORM_ALIASES.get(key)
    if platform is None:
        supported = ", ".join(sorted(_DOCKER_TO_NIX_SYSTEM))
        raise SystemExit(f"--platform {value!r}: supported values are {supported}")
    return platform


def detect_default_platform(machine: str | None = None) -> str:
    """Best-effort default Docker platform for the current build host.

    Agentix builds Linux container images even when invoked from macOS,
    so only the CPU architecture is inherited from the host.
    """
    raw = (machine or host_platform.machine()).strip().lower().replace("_", "-")
    if raw in {"amd64", "x86-64"}:
        return "linux/amd64"
    if raw in {"arm64", "aarch64"}:
        return "linux/arm64"
    raise SystemExit(f"cannot auto-detect Docker platform from machine {raw!r}; pass --platform")


def nix_system_for_platform(platform: str) -> str:
    """Return the Nix system matching a normalized Docker platform."""
    platform = normalize_platform(platform)
    return _DOCKER_TO_NIX_SYSTEM[platform]


__all__ = [
    "detect_default_platform",
    "nix_system_for_platform",
    "normalize_platform",
]
