"""Sandbox SIO namespace for the bridge forwarder."""

from __future__ import annotations

from agentix.bridge.types import NAMESPACE

import agentix


class SandboxNamespace(agentix.Namespace):
    namespace = NAMESPACE


_namespace_singleton: SandboxNamespace | None = None


def get_namespace() -> SandboxNamespace:
    global _namespace_singleton
    if _namespace_singleton is None:
        _namespace_singleton = SandboxNamespace()
        agentix.register_namespace(_namespace_singleton)
    return _namespace_singleton
