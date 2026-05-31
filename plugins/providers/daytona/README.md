# agentix-provider-daytona

[Daytona](https://www.daytona.io/) provider backend for
[Agentix](https://github.com/Agentiix/Agentix).

> Status: CLI surface in place; the managed-sandbox integration is
> still a stub. Tracking parity with `DockerProvider` (live runtime
> URL, `session()` context manager) before promoting to a 1.0
> release.

## Install

```bash
pip install agentix-provider-daytona
```

Set `DAYTONA_API_KEY` in the environment.

## Use

```bash
agentix deploy daytona dist/my-agent.bundle.tar
```

```python
from agentix import RuntimeClient, SandboxConfig
from agentix.provider.daytona import DaytonaProvider

async with DaytonaProvider().session(
    SandboxConfig(image="python:3.13-slim", bundle="<backend bundle ref>")
) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as c:
        ...
```

## License

MIT — see [LICENSE](LICENSE).
