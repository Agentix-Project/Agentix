import pytest


def test_public_import_surface():
    import tito_gateway
    from tito_gateway import TITOGateway, TITOGatewayConfig, SessionServer, get_tito_tokenizer

    assert tito_gateway.TITOGateway is TITOGateway
    assert tito_gateway.TITOGatewayConfig is TITOGatewayConfig
    assert tito_gateway.SessionServer is SessionServer
    assert callable(get_tito_tokenizer)


def test_config_requires_hf_checkpoint():
    from tito_gateway import TITOGatewayConfig

    with pytest.raises(ValueError, match="hf_checkpoint is required"):
        TITOGatewayConfig(hf_checkpoint="")


def test_gateway_constructs_with_explicit_backend(monkeypatch):
    import tito_gateway.gateway as gateway_module
    from tito_gateway import TITOGateway

    class FakeSessionServer:
        def __init__(self, args, backend_url):
            self.args = args
            self.backend_url = backend_url
            self.app = object()

    monkeypatch.setattr(gateway_module, "SessionServer", FakeSessionServer)

    gateway = TITOGateway.from_server(hf_checkpoint="Qwen/Qwen3-0.6B", backend_url="127.0.0.1:8000")

    assert gateway.config.backend_url == "http://127.0.0.1:8000"
    assert gateway.app is gateway.server.app
    assert gateway.server.args.hf_checkpoint == "Qwen/Qwen3-0.6B"
