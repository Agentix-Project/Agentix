from unittest.mock import patch

import requests

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from tito_gateway import TITOGateway, TITOGatewayConfig


def test_tito_gateway_serves_real_session_routes_with_explicit_backend():
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text=f"echo: {prompt}", finish_reason="stop")

    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"] = {
            "output_token_logprobs": output_token_logprobs,
            "completion_tokens": len(output_token_logprobs),
        }
        return response

    with (
        patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response),
        with_mock_server(process_fn=process_fn) as backend,
    ):
        gateway = TITOGateway(
            TITOGatewayConfig(
                hf_checkpoint="Qwen/Qwen3-0.6B",
                backend_url=backend.url,
                apply_chat_template_kwargs={"enable_thinking": False},
                tito_model="default",
                tito_allowed_append_roles=("tool",),
                miles_router_timeout=30,
            )
        )

        port = find_available_port(33000)
        server = UvicornThreadServer(gateway.app, host="127.0.0.1", port=port)
        server.start()
        url = f"http://127.0.0.1:{port}"

        try:
            health = requests.get(f"{url}/health", timeout=5.0)
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            session_id = requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]
            payload = {
                "messages": [{"role": "user", "content": "What is 1+2?"}],
                "return_logprob": True,
            }
            response = requests.post(
                f"{url}/sessions/{session_id}/v1/chat/completions",
                json=payload,
                timeout=10.0,
            )

            assert response.status_code == 200
            assert response.json()["choices"]
            assert len(backend.request_log) == 1
            proxied_payload = backend.request_log[0]
            assert proxied_payload["messages"] == payload["messages"]
            assert proxied_payload["logprobs"] is True
            assert proxied_payload["return_meta_info"] is True
            assert proxied_payload["no_stop_trim"] is False
            assert isinstance(proxied_payload["input_ids"], list)
            assert proxied_payload["input_ids"]

            session = requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()
            assert len(session["records"]) == 1
            assert session["records"][0]["path"] == "/v1/chat/completions"
            assert session["records"][0]["status_code"] == 200
        finally:
            server.stop()
