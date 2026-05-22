# agentix-deployment-e2b

[E2B](https://e2b.dev/) deployment backend for
[Agentix](https://github.com/Agentiix/Agentix).

> Status: CLI surface in place; the managed-sandbox integration is
> still a stub. Tracking parity with `DockerDeployment` (live runtime
> URL, `lifecycle()` context manager) before promoting to a 1.0
> release.

## Install

```bash
pip install agentix-deployment-e2b
```

Set `E2B_API_KEY` in the environment.

## Use

```bash
agentix deploy e2b --image my-agent:0.1.0
```

```python
from agentix import RuntimeClient, SandboxConfig
from agentix.deployment.e2b import E2BDeployment

async with E2BDeployment().lifecycle(
    SandboxConfig(image="python:3.13-slim", bundle="my-agent:0.1.0")
) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as c:
        ...
```

## License

MIT — see [LICENSE](LICENSE).
