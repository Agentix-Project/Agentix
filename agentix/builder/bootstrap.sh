#!/nix/runtime/bin/bash
# agentix bundle bootstrap — single source of truth.
#
# Shipped as wheel data under `agentix/builder/` and placed at
# `/nix/runtime/bootstrap.sh` inside every bundle by the in-container
# build (`agentix/builder/bundle-build.sh`).
#
# Provider backends (docker, apptainer, future k8s/...) use this as
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

# Each plugin lives at /nix/runtime/plugins/<label>/ as its own
# /nix/store tree (no symlinkJoin / buildEnv merge with the toolchain
# or with other plugins). We compose PATH / LD_LIBRARY_PATH / ... by
# globbing this dir at startup — same mental model as `nix-shell -p
# a b c`, where shell selection of a binary is decided by PATH order,
# not by a build-time tree merge. Glob order is lexicographic on the
# plugin labels (`<dist>.<ep>`); first match wins on PATH lookup.
agentix_collect_plugin_paths() {
  subdir="$1"
  result=""
  if [ -d /nix/runtime/plugins ]; then
    for plugin in /nix/runtime/plugins/*/; do
      candidate="${plugin}${subdir}"
      [ -d "$candidate" ] || continue
      candidate="${candidate%/}"
      if [ -n "$result" ]; then
        result="${result}:${candidate}"
      else
        result="${candidate}"
      fi
    done
  fi
  printf '%s' "$result"
}

# Join non-empty items with ':' — accepts a baseline followed by any
# number of (possibly empty) extras. Empty extras are skipped, so a
# missing plugin subdir doesn't introduce empty PATH entries (which
# Unix interprets as CWD).
agentix_colon_append() {
  list="$1"
  shift
  for item in "$@"; do
    [ -n "$item" ] || continue
    if [ -n "$list" ]; then
      list="${list}:${item}"
    else
      list="${item}"
    fi
  done
  printf '%s' "$list"
}

plugin_bins="$(agentix_collect_plugin_paths bin)"
plugin_libs="$(agentix_collect_plugin_paths lib)"
plugin_includes="$(agentix_collect_plugin_paths include)"
plugin_lib_pkgconfig="$(agentix_collect_plugin_paths lib/pkgconfig)"
plugin_share_pkgconfig="$(agentix_collect_plugin_paths share/pkgconfig)"
plugin_roots="$(agentix_collect_plugin_paths "")"

agentix_prepend_path PATH               "$(agentix_colon_append "/nix/runtime/venv/bin:/nix/runtime/bin" "$plugin_bins")"
agentix_prepend_path LD_LIBRARY_PATH    "$(agentix_colon_append "/nix/runtime/lib" "$plugin_libs")"
agentix_prepend_path LIBRARY_PATH       "$(agentix_colon_append "/nix/runtime/lib" "$plugin_libs")"
agentix_prepend_path CPATH              "$(agentix_colon_append "/nix/runtime/include" "$plugin_includes")"
agentix_prepend_path C_INCLUDE_PATH     "$(agentix_colon_append "/nix/runtime/include" "$plugin_includes")"
agentix_prepend_path CPLUS_INCLUDE_PATH "$(agentix_colon_append "/nix/runtime/include" "$plugin_includes")"
agentix_prepend_path PKG_CONFIG_PATH    "$(agentix_colon_append "/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig" "$plugin_lib_pkgconfig" "$plugin_share_pkgconfig")"
agentix_prepend_path CMAKE_PREFIX_PATH  "$(agentix_colon_append "/nix/runtime" "$plugin_roots")"

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
