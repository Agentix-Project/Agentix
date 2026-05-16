# Runtime image — Python slim + framework venv + uv.
#
# Built from the repo root:
#   docker build -t agentix/runtime:<version> -f primitives/_template/Dockerfile .
#
# The runtime image carries:
#   /venvs/runtime/        — framework venv (uv-managed); ENTRYPOINT runs from here
#   /agentix-wheel/        — the framework wheel, stashed for bundle stages
#   uv binary on $PATH     — used by `agentix build`'s generated Dockerfile to
#                            create one venv per namespace
#
# Namespace bundles extend this image: they create `/venvs/<short>/` for each
# namespace, pip-install the namespace + the framework wheel into it, and
# the multiplexer spawns one worker subprocess per namespace's interpreter
# so deps stay isolated across namespaces.

FROM python:3.11-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md ./
COPY agentix ./agentix
RUN python -m build --wheel --outdir /dist

FROM python:3.11-slim
RUN pip install --no-cache-dir uv

# Framework wheel — kept so each bundled namespace's venv can install it
# without reaching PyPI.
COPY --from=builder /dist/*.whl /agentix-wheel/

# Runtime's own venv. /venvs/runtime owns the framework dispatcher;
# the multiplexer is what spawns per-namespace workers.
RUN uv venv /venvs/runtime && \
    /venvs/runtime/bin/pip install --no-cache-dir /agentix-wheel/agentix-*.whl

EXPOSE 8000
ENV AGENTIX_BIND_PORT=8000
ENTRYPOINT ["/venvs/runtime/bin/agentix-server"]
