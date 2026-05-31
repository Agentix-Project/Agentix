"""SandboxProvider Protocol + backend discovery.

Core ships the `SandboxProvider` Protocol, `Sandbox` dataclass, and backend
registry. Backend wheels (`agentix-provider-docker`, `-daytona`,
`-e2b`, third-party) each install a sibling module under
`agentix.provider`; extending `__path__` lets those siblings co-exist
with the framework files in this directory.
"""

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from agentix.provider.base import (
    BundleDeployer,
    DeployedBundle,
    Sandbox,
    SandboxConfig,
    SandboxId,
    SandboxInfo,
    SandboxProvider,
    SandboxResource,
    providers,
    register_provider,
)

__all__ = [
    "BundleDeployer",
    "DeployedBundle",
    "Sandbox",
    "SandboxConfig",
    "SandboxId",
    "SandboxInfo",
    "SandboxProvider",
    "SandboxResource",
    "providers",
    "register_provider",
]
