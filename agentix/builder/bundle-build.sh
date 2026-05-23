#!/bin/sh
# In-container build orchestration for `agentix build`. Run by the
# Dockerfile against the staged context (`repo/`, `flake.nix`,
# `closures/`, `python-version`).
#
#   1. nix build the toolchain  — interpreter + uv
#   2. uv venv + uv sync        — /nix/runtime/venv (all Python deps)
#   3. discover system closures — plugin (agentix.nix entry points)
#                                 + project ([tool.agentix] nix)
#   4. nix build the runtime    — toolchain + closures, merged
#   5. place the merged tree    — /nix/runtime/{bin,lib,...}
#   6. write bootstrap.sh       — bundle's startup contract
set -eu

SUBPATH="${AGENTIX_PROJECT_SUBPATH:-.}"
PROJECT="/build/repo/${SUBPATH}"

if [ ! -f "${PROJECT}/pyproject.toml" ]; then
    echo "bundle-build: no pyproject.toml at ${PROJECT}" >&2
    exit 1
fi

# Flakes only see git-tracked files; the context is not a repo yet.
git init -q
git config user.email build@agentix.local
git config user.name agentix-build
git add -A

echo ">>> [1/5] building Nix toolchain (interpreter + uv)"
nix build .#toolchain -o toolchain-result --print-build-logs
TOOLCHAIN="$(readlink -f toolchain-result)"

echo ">>> [2/5] creating venv + syncing Python deps"
mkdir -p /nix/runtime
"${TOOLCHAIN}/bin/uv" venv /nix/runtime/venv --python "${TOOLCHAIN}/bin/python3"
( cd "${PROJECT}" \
  && VIRTUAL_ENV=/nix/runtime/venv "${TOOLCHAIN}/bin/uv" sync \
        --active --frozen --no-dev --no-editable )

echo ">>> [3/5] discovering system-dep closures"
/nix/runtime/venv/bin/python -m agentix.cli._assemble \
    --project "${PROJECT}" --closures /build/closures
git add -A

echo ">>> [4/5] building Nix runtime closure"
nix build .#runtime -o runtime-result --print-build-logs

echo ">>> [5/6] placing /nix/runtime"
cp -a runtime-result/. /nix/runtime/
rm -f toolchain-result runtime-result

# Drop the bundle's startup contract at /nix/runtime/bootstrap.sh.
# Deployment backends (docker, apptainer, future k8s/...) just exec
# this — they stay agnostic about Python venvs, LD_LIBRARY_PATH, or
# where `agentix-server` lives. The script is shipped verbatim as
# wheel data from `agentix/builder/bootstrap.sh` and staged into the
# build context next to bundle-build.sh.
echo ">>> [6/6] installing /nix/runtime/bootstrap.sh"
install -m 0755 /build/bootstrap.sh /nix/runtime/bootstrap.sh

echo ">>> bundle build complete"
