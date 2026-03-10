from typing import cast

import httpx
import pytest

from tempus_copilot.llm import baml_adapter as adapter_module
from tempus_copilot.llm.baml_adapter import OllamaGenerationClient, get_default_generation_client


def test_ollama_generation_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OllamaGenerationClient(model="qwen3.5:397b")


def test_ollama_generation_client_rejects_localhost_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    with pytest.raises(ValueError):
        OllamaGenerationClient(model="qwen3.5:397b")


def test_ollama_generation_client_generates_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")

    class FakeResponse:
        def __init__(self, payload: object) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if not isinstance(self._payload, dict):
                raise ValueError("payload must be a dict")
            return cast(dict[str, object], self._payload)

    payloads = [
        {
            "message": {
                "role": "assistant",
                "content": (
                    '{"response":"Use evidence.","supporting_metrics":["8 days"],'
                    '"citations":["kb:1"],"confidence":0.88}'
                ),
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": '{"script":"Intro plan.","citations":["kb:1"],"confidence":0.82}',
            }
        },
    ]

    def fake_post(*_: object, **__: object) -> FakeResponse:
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr(adapter_module.httpx, "post", fake_post)
    client = OllamaGenerationClient(model="qwen3.5:397b")
    objection = client.generate_objection_handler(
        provider_id="P001",
        concern="turnaround_time",
        kb_context="Average turnaround is 8 days.",
        citation_ids=["kb:1"],
        observed_metrics=["8 days"],
    )
    script = client.generate_meeting_script(
        provider_id="P001",
        tumor_focus="Lung",
        kb_context="Average turnaround is 8 days.",
        citation_ids=["kb:1"],
    )
    assert objection.provider_id == "P001"
    assert objection.response == "Use evidence."
    assert script.script == "Intro plan."


def test_ollama_generation_client_retries_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    calls = {"count": 0}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"response":"ok","supporting_metrics":[],"citations":["kb:1"],'
                        '"confidence":0.9}'
                    ),
                }
            }

    def fake_post(*_: object, **__: object) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("temporary")
        return FakeResponse()

    monkeypatch.setattr(adapter_module.httpx, "post", fake_post)
    monkeypatch.setattr(adapter_module, "sleep", lambda _: None)
    client = OllamaGenerationClient(model="qwen3.5:397b", request_retries=1)
    out = client.generate_objection_handler(
        provider_id="P001",
        concern="turnaround_time",
        kb_context="context",
        citation_ids=["kb:1"],
        observed_metrics=["8 days"],
    )
    assert calls["count"] == 2
    assert out.response == "ok"


def test_get_default_generation_client_rejects_non_ollama_provider() -> None:
    with pytest.raises(ValueError):
        get_default_generation_client(generation_provider="google")
