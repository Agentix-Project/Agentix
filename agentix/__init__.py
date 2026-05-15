"""agentix — a Nix-closure runtime for Docker sandboxes."""

# `trace` is imported eagerly so closure impls can `from agentix import trace`
# without circular-import gymnastics. It has no runtime deps and registers an
# emitter only when the server boots, so this is cheap.
from agentix import trace
from agentix.deployment.base import Sandbox
from agentix.deployment.docker import DockerDeployment
from agentix.dispatch import Dispatcher, Registry
from agentix.models import SandboxConfig, SandboxInfo
from agentix.rollout import RolloutPool
from agentix.runtime.client import RemoteCallError, RuntimeClient
from agentix.runtime.models import LogRecord, TraceEvent

__version__ = "0.1.0"

__all__ = [
    "Dispatcher",
    "DockerDeployment",
    "LogRecord",
    "Registry",
    "RemoteCallError",
    "RolloutPool",
    "RuntimeClient",
    "Sandbox",
    "SandboxConfig",
    "SandboxInfo",
    "TraceEvent",
    "__version__",
    "trace",
]
