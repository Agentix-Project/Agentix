# syntax=docker/dockerfile:1
#
# agentix bundle builder. The transient build image contains `/nix`
# with everything a sandbox needs. The host build command copies that
# tree into the portable bundle tar; deployment backends later make the
# same tree visible in the task container at `/nix`.
#
#   /nix/store/...              the closures (interpreter, uv, system deps)
#   /nix/runtime/venv           the uv venv — all Python deps
#   /nix/runtime/{bin,lib,...}  symlinkJoin of every closure
#
# Nix runs only inside this build — the host needs just docker.

ARG AGENTIX_BUILDER_BASE=nixos/nix:latest
FROM ${AGENTIX_BUILDER_BASE}

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
# Flakes + the modern CLI; accept flake `nixConfig` for substituters.
# Syscall filtering can fail under Docker buildx emulation; the Docker
# build sandbox is already the outer isolation boundary here. Per-build nix
# knobs (substituters, trusted keys, cores, ...) come through AGENTIX_NIX_ARGS
# below, forwarded verbatim to `nix build` — no bespoke ARG per option.
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf \
 && echo "accept-flake-config = true" >> /etc/nix/nix.conf \
 && echo "filter-syscalls = false" >> /etc/nix/nix.conf

# Where the project sits inside the copied repo (a uv-workspace member
# or cookbook example lives in a subdirectory). Passed by `agentix build`.
ARG AGENTIX_PROJECT_SUBPATH=.
ENV AGENTIX_PROJECT_SUBPATH=${AGENTIX_PROJECT_SUBPATH}

# Raw passthrough args forwarded verbatim to the in-container build engines,
# set by `agentix build --nix-arg ... --uv-arg ...`. bundle-build.sh word-splits
# them onto `nix build` / `uv sync`.
ARG AGENTIX_NIX_ARGS=
ARG AGENTIX_UV_ARGS=
ENV AGENTIX_NIX_ARGS=${AGENTIX_NIX_ARGS} \
    AGENTIX_UV_ARGS=${AGENTIX_UV_ARGS}

COPY . /build
WORKDIR /build

# nix build (toolchain + runtime), uv venv + uv sync, closure discovery.
RUN export http_proxy="${http_proxy:-$HTTP_PROXY}" \
 && export https_proxy="${https_proxy:-$HTTPS_PROXY}" \
 && export no_proxy="${no_proxy:-$NO_PROXY}" \
 && export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}" \
 && export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}" \
 && export NO_PROXY="${NO_PROXY:-$no_proxy}" \
 && sh ./bundle-build.sh

ENV PATH="/nix/runtime/venv/bin:/nix/runtime/bin:${PATH}" \
    LD_LIBRARY_PATH="/nix/runtime/lib" \
    LIBRARY_PATH="/nix/runtime/lib" \
    CPATH="/nix/runtime/include" \
    C_INCLUDE_PATH="/nix/runtime/include" \
    CPLUS_INCLUDE_PATH="/nix/runtime/include" \
    PKG_CONFIG_PATH="/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig" \
    CMAKE_PREFIX_PATH="/nix/runtime" \
    AGENTIX_ADDED_PATH="/nix/runtime/venv/bin:/nix/runtime/bin" \
    AGENTIX_ADDED_LD_LIBRARY_PATH="/nix/runtime/lib" \
    AGENTIX_ADDED_LIBRARY_PATH="/nix/runtime/lib" \
    AGENTIX_ADDED_CPATH="/nix/runtime/include" \
    AGENTIX_ADDED_C_INCLUDE_PATH="/nix/runtime/include" \
    AGENTIX_ADDED_CPLUS_INCLUDE_PATH="/nix/runtime/include" \
    AGENTIX_ADDED_PKG_CONFIG_PATH="/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig" \
    AGENTIX_ADDED_CMAKE_PREFIX_PATH="/nix/runtime" \
    AGENTIX_BIND_PORT=8000
EXPOSE 8000/tcp
VOLUME ["/nix"]
ENTRYPOINT ["/nix/runtime/bootstrap.sh"]
