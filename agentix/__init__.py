"""agentix — remote calls for sandboxed Python modules.

Integration wheels may contribute modules under `agentix.<short>`
(e.g. `agentix.bash`). Extending `agentix.__path__` lets those modules
co-exist with the framework modules in this package.
"""

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from agentix._quick import quick_remote
from agentix.deployment.base import (
    BundleMaterializer,
    Deployment,
    MaterializedBundle,
    Sandbox,
    SandboxConfig,
    SandboxId,
    SandboxInfo,
    SandboxResource,
    deployments,
    load_deployment,
    register_deployment,
    session,
)
from agentix.runtime.client import RemoteCallError, RuntimeClient
from agentix.runtime.client._sio_facade import AsyncClientNamespace, request_handler
from agentix.runtime.shared.callables import RemoteCallable
from agentix.sio import Namespace, RemoteSioError, register_namespace
from agentix.utils import log, trace

__version__ = "0.2.7"

__all__ = [
    "AsyncClientNamespace",
    "BundleMaterializer",
    "Deployment",
    "MaterializedBundle",
    "Namespace",
    "RemoteCallable",
    "RemoteCallError",
    "RemoteSioError",
    "RuntimeClient",
    "Sandbox",
    "SandboxConfig",
    "SandboxId",
    "SandboxInfo",
    "SandboxResource",
    "__version__",
    "deployments",
    "load_deployment",
    "log",
    "quick_remote",
    "register_deployment",
    "register_namespace",
    "request_handler",
    "session",
    "trace",
]
