# agentix-deployment-docker

Docker deployment backend for [Agentix](https://github.com/Agentiix/Agentix).

Provisions a sandbox by running an Agentix bundle image in a local
Docker daemon, returns the runtime URL the orchestrator's
`RuntimeClient` connects to.

## Install

```bash
pip install agentix-deployment-docker
```

## Use

```python
from agentix import RuntimeClient, SandboxConfig
from agentix.deployment.docker import DockerDeployment

async with DockerDeployment().lifecycle(
    SandboxConfig(image="python:3.13-slim", bundle="my-agent:0.1.0")
) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as c:
        ...
```

Or via the CLI:

```bash
agentix deploy local --image my-agent:0.1.0
```

The `local` name comes from the entry point this wheel declares under
`agentix.deployment` — once installed, the framework discovers it
automatically. No core-framework changes are required to register a new
backend; the same pattern works for `agentix-deployment-daytona`,
`agentix-deployment-e2b`, or any third-party backend.

## License

MIT — see [LICENSE](LICENSE).
