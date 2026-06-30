import pytest


def test_public_import_surface():
    import agentix.tito
    from agentix.tito import TITOGateway, TITOGatewayConfig, SessionServer, get_tito_tokenizer

    assert agentix.tito.TITOGateway is TITOGateway
    assert agentix.tito.TITOGatewayConfig is TITOGatewayConfig
    assert agentix.tito.SessionServer is SessionServer
    assert callable(get_tito_tokenizer)


def test_config_requires_hf_checkpoint():
    from agentix.tito import TITOGatewayConfig

    with pytest.raises(ValueError, match="hf_checkpoint is required"):
        TITOGatewayConfig(hf_checkpoint="")


def test_gateway_constructs_with_explicit_backend(monkeypatch):
    import agentix.tito.gateway as gateway_module
    from agentix.tito import TITOGateway

    class FakeSessionServer:
        def __init__(self, args, backend_url):
            from fastapi import FastAPI

            self.args = args
            self.backend_url = backend_url
            # A real app: the gateway registers a `/healthz` alias on it at construct.
            self.app = FastAPI()

    monkeypatch.setattr(gateway_module, "SessionServer", FakeSessionServer)

    gateway = TITOGateway.from_server(hf_checkpoint="Qwen/Qwen3-0.6B", backend_url="127.0.0.1:8000")

    assert gateway.config.backend_url == "http://127.0.0.1:8000"
    assert gateway.app is gateway.server.app
    assert gateway.server.args.hf_checkpoint == "Qwen/Qwen3-0.6B"
