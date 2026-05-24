#!/nix/runtime/bin/bash
# agentix bundle bootstrap — single source of truth.
#
# Shipped as wheel data under `agentix/builder/` and placed at
# `/nix/runtime/bootstrap.sh` inside every bundle by the in-container
# build (`agentix/builder/bundle-build.sh`).
#
# Deployment backends (docker, apptainer, future k8s/...) use this as
# the container entry point — they bake in zero knowledge of Python
# venvs, LD paths, or where the runtime server lives. New backend =
# "exec `/nix/runtime/bootstrap.sh` as PID 1, done".
#
# Shebang: `/nix/runtime/bin/bash` is part of every bundle (see
# `agentix-runtime` in `flake.nix`). We use the bundle's own bash so
# the script's behaviour is independent of whatever `/bin/sh` the task
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

# Some task images (e.g. swebench) inject their own Python-related env
# (PYTHONPATH, cwd entries on `sys.path`, ...) that would shadow the
# bundle venv's `agentix`, `uvicorn`, or `fastapi`. Scrub `sys.path`
# before the first third-party import, then hand off to uvicorn. Single
# quotes around the `python -c` body: no shell interpolation — host /
# port come from env vars read inside Python.
exec /nix/runtime/venv/bin/python -c '
import os, sys
sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd())]
import uvicorn
from agentix.runtime.server.app import app
from agentix.runtime.shared import MAX_MESSAGE_BYTES
uvicorn.run(
    app,
    host=os.environ.get("AGENTIX_BIND_HOST", "0.0.0.0"),
    port=int(os.environ.get("AGENTIX_BIND_PORT", "8000")),
    ws="wsproto",
    ws_max_size=MAX_MESSAGE_BYTES,
)
'
