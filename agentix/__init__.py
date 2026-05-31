"""agentix — remote calls for sandboxed Python modules.

Integration wheels may contribute modules under `agentix.<short>`
(e.g. `agentix.bash`). Extending `agentix.__path__` lets those modules
co-exist with the framework modules in this package.
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
from agentix.runtime.client import (
    CallTimeout,
    RemoteCallError,
    RuntimeClient,
    RuntimeUnreachable,
    WorkerExited,
)
from agentix.runtime.client._sio_facade import AsyncClientNamespace, request_handler
from agentix.runtime.shared.callables import RemoteCallable
from agentix.sio import Namespace, RemoteSioError, register_namespace
from agentix.utils import context, log, trace
from agentix.utils.log import configure_logging

__version__ = "0.2.7"

__all__ = [
    "AsyncClientNamespace",
    "BundleDeployer",
    "CallTimeout",
    "DeployedBundle",
    "Namespace",
    "RemoteCallable",
    "RemoteCallError",
    "RemoteSioError",
    "RuntimeClient",
    "RuntimeUnreachable",
    "Sandbox",
    "SandboxConfig",
    "SandboxId",
    "SandboxInfo",
    "SandboxProvider",
    "SandboxResource",
    "WorkerExited",
    "__version__",
    "configure_logging",
    "context",
    "log",
    "providers",
    "register_namespace",
    "register_provider",
    "request_handler",
    "trace",
]
