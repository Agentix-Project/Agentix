# agentix-provider-docker

Docker provider backend for [Agentix](https://github.com/Agentiix/Agentix).

Provisions a sandbox by running a deployed Agentix bundle in a
Docker-compatible runtime, returns the runtime URL the orchestrator's
`RuntimeClient` connects to.

`agentix build` produces a portable bundle tar. `agentix deploy docker`
or `agentix deploy podman` unpacks that tar into a content-addressed
host cache (the local-extract form of the `BundleDeployer` Protocol),
then `SandboxConfig.bundle` uses the returned cache path.

## Install

```bash
pip install agentix-provider-docker
```

## Use

```python
from agentix import RuntimeClient, SandboxConfig, session
from agentix.provider.docker import DockerProvider, DockerProviderConfig

provider = DockerProvider(
    DockerProviderConfig(
        container_engine="podman",
        run_args=["--runtime=crun", "--cgroups=disabled"],
        network="host",
        gpu_args=["--device", "nvidia.com/gpu=all"],
    )
)

async with session(
    provider,
    SandboxConfig(
        image="python:3.13-slim",
        bundle="/home/me/.cache/agentix/bundles/sha256-...",  # printed by `agentix deploy`
        resource={"cpu": 4, "memory": "16g", "gpu": 1},
    )
) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as c:
        ...
```

```bash
agentix build . --output dist/my-agent.bundle.tar
agentix deploy podman dist/my-agent.bundle.tar \
  --run-arg --runtime=crun \
  --run-arg --cgroups=disabled
```

`network="host"` skips port publishing and relies on the runtime binding
directly in the host network namespace. In that mode the backend sets
`AGENTIX_BIND_HOST=127.0.0.1` unless the caller overrides it in
`SandboxConfig.env`.

The `docker` and `podman` names come from the entry points this wheel
declares under `agentix.provider` — once installed, the framework
discovers them automatically. No core-framework changes are required to
register a new backend; the same pattern works for
`agentix-provider-daytona`, `agentix-provider-e2b`, or any third-party
backend.

## License

MIT — see [LICENSE](LICENSE).
