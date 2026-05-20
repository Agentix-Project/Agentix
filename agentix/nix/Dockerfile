# syntax=docker/dockerfile:1
#
# agentix bundle builder. The produced image is a *runtime carrier*:
# its `/nix` holds everything a sandbox needs and is mounted into a
# task container at `/nix` (`--mount type=image,subpath=nix`).
#
#   /nix/store/...              the closures (interpreter, uv, system deps)
#   /nix/runtime/venv           the uv venv — all Python deps
#   /nix/runtime/{bin,lib,...}  symlinkJoin of every closure
#
# Nix runs only inside this build — the host needs just docker.

FROM nixos/nix:latest

# Flakes + the modern CLI; accept flake `nixConfig` for substituters.
# Syscall filtering can fail under Docker buildx emulation; the Docker
# build sandbox is already the outer isolation boundary here.
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf \
 && echo "accept-flake-config = true" >> /etc/nix/nix.conf \
 && echo "filter-syscalls = false" >> /etc/nix/nix.conf

# Where the project sits inside the copied repo (a uv-workspace member
# or cookbook example lives in a subdirectory). Passed by `agentix build`.
ARG AGENTIX_PROJECT_SUBPATH=.
ENV AGENTIX_PROJECT_SUBPATH=${AGENTIX_PROJECT_SUBPATH}

COPY . /build
WORKDIR /build

# nix build (toolchain + runtime), uv venv + uv sync, closure discovery.
RUN sh ./bundle-build.sh

ENV PATH="/nix/runtime/bin:/nix/runtime/venv/bin:${PATH}" \
    AGENTIX_BIND_PORT=8000
EXPOSE 8000/tcp
ENTRYPOINT ["/nix/runtime/venv/bin/agentix-server"]
