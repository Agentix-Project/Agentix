#!/nix/runtime/bin/bash
# agentix bundle bootstrap — single source of truth.
#
# Shipped as wheel data under `agentix/builder/` and placed at
# `/nix/runtime/bootstrap.sh` inside every bundle by the in-container
# build (`agentix/builder/bundle-build.sh`).
#
# Deployment backends (docker, apptainer, future k8s/...) use this as
# the container entrypoint — they bake in zero knowledge of Python
# venvs, LD paths, or where `agentix-server` lives. New backend =
# "exec `/nix/runtime/bootstrap.sh` as PID 1, done".
#
# Shebang: `/nix/runtime/bin/bash` is part of every bundle (see
# `agentix-runtime` in `flake.nix`). We use the bundle's own bash so
# the script's behavior is independent of whatever `/bin/sh` the task
# image happens to ship (dash, busybox sh, etc.).
set -euo pipefail

agentix_prepend_path() {
  name="$1"
  added="$2"
  tracking="AGENTIX_ADDED_${name}"
  eval "current=\${$name-}"
  eval "tracked=\${$tracking-}"
  if [ -n "$current" ]; then
    export "$name=$added:$current"
  else
    export "$name=$added"
  fi
  if [ -n "$tracked" ]; then
    export "$tracking=$tracked:$added"
  else
    export "$tracking=$added"
  fi
}

agentix_prepend_path PATH "/nix/runtime/venv/bin:/nix/runtime/bin"
agentix_prepend_path LD_LIBRARY_PATH "/nix/runtime/lib"
agentix_prepend_path LIBRARY_PATH "/nix/runtime/lib"
agentix_prepend_path CPATH "/nix/runtime/include"
agentix_prepend_path C_INCLUDE_PATH "/nix/runtime/include"
agentix_prepend_path CPLUS_INCLUDE_PATH "/nix/runtime/include"
agentix_prepend_path PKG_CONFIG_PATH "/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig"
agentix_prepend_path CMAKE_PREFIX_PATH "/nix/runtime"

HOST="${AGENTIX_BIND_HOST:-0.0.0.0}"
PORT="${AGENTIX_BIND_PORT:-8000}"

# Hand off to uvicorn via `python -c`. Scrub the current working
# directory off `sys.path` BEFORE `import uvicorn`, otherwise a task
# image whose WORKDIR happens to contain a `uvicorn/` / `agentix/` /
# `fastapi/` directory would shadow the bundle venv's modules and we
# get nondeterministic startup failures. `uvicorn.Config.__init__`
# imports the app, so the scrub must happen first — calling the
# console-script entry point `/nix/runtime/venv/bin/agentix-server`
# doesn't give us a window between Python startup and the first
# third-party import.
exec /nix/runtime/venv/bin/python -c "
import os, sys
_cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ('', '.', _cwd)]
from agentix.runtime.server.app import app as asgi_app
from agentix.runtime.shared import MAX_MESSAGE_BYTES
import uvicorn
uvicorn.run(asgi_app, host='$HOST', port=$PORT, ws='wsproto', ws_max_size=MAX_MESSAGE_BYTES)
"
