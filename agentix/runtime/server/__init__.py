"""Sandbox-side runtime server.

Composes FastAPI (for `/health`) and Socket.IO (for remote calls + the
trace broadcast channel) into the ASGI app uvicorn runs. Remote calls
route to one runtime worker subprocess.

Submodules:
  - `app`         — FastAPI app, lifespan, /health
  - `sio`         — Socket.IO server + remote-call event handlers
  - `worker`      — worker client, subprocess entry point, callable invocation

There is no console-script or `__main__` entry point: bundles boot via
`/nix/runtime/bootstrap.sh` (shipped from `agentix/builder/bootstrap.sh`),
which calls `python -c '... import app ...'` directly so it can scrub
`sys.path` before any third-party import — task images sometimes inject
Python-related env that would otherwise shadow the bundle venv.
"""

from agentix.runtime.server.app import app

__all__ = ["app"]
