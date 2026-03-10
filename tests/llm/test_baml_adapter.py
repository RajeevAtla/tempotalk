from __future__ import annotations

from typing import cast

import httpx
import pytest

from tempus_copilot.llm import baml_adapter as adapter_module
from tempus_copilot.llm.baml_adapter import (
    OllamaGenerationClient,
    get_default_generation_client,
)


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "https://ollama.com/api/chat"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, object]:
        if not isinstance(self._payload, dict):
            raise ValueError("payload must be a dict")
        return cast(dict[str, object], self._payload)


def make_client(monkeypatch: pytest.MonkeyPatch) -> OllamaGenerationClient:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    return OllamaGenerationClient(model="ministral-3:8b")


def test_normalize_base_url_variants() -> None:
    assert adapter_module._normalize_base_url("https://ollama.com/") == "https://ollama.com"
    with pytest.raises(ValueError):
        adapter_module._normalize_base_url("http://ollama.com")


def test_extract_json_payload_handles_fences_prose_and_sanitization() -> None:
    fenced = '```json\n{"response":"ok"}\n```'
    prose = 'before {"citations":["kb:1"]} after'
    broken = '{\x00"confidence":"0.7"}'
    assert adapter_module._extract_json_payload(fenced)["response"] == "ok"
    assert adapter_module._extract_json_payload(prose)["citations"] == ["kb:1"]
    assert adapter_module._extract_json_payload(broken)["confidence"] == "0.7"
    with pytest.raises(ValueError):
        adapter_module._extract_json_payload("[]")


def test_extract_json_payload_handles_short_fence_form() -> None:
    assert adapter_module._extract_json_payload('```{"fixed": true}') == {"fixed": True}


def test_coerce_helpers_cover_mixed_inputs() -> None:
    assert adapter_module._coerce_string_list("nope") == []
    assert adapter_module._coerce_string_list(["a", 1, "b"]) == ["a", "b"]
    assert adapter_module._coerce_confidence("0.5") == 0.5
    assert adapter_module._coerce_confidence("bad") == 0.0
    assert adapter_module._coerce_confidence({"confidence": "0.6"}) == 0.6
    assert adapter_module._coerce_confidence({"score": 0.7}) == 0.7
    assert adapter_module._coerce_confidence({"value": "0.8"}) == 0.8
    assert adapter_module._coerce_confidence({"other": "nope"}) == 0.0
    assert adapter_module._coerce_confidence(object()) == 0.0


def test_ollama_generation_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OllamaGenerationClient(model="ministral-3:8b")


def test_ollama_generation_client_rejects_localhost_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    with pytest.raises(ValueError):
        OllamaGenerationClient(model="ministral-3:8b")


def test_chat_json_raises_for_missing_message_and_non_string_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(client, "_post_chat", lambda payload: {})
    with pytest.raises(ValueError, match="missing message"):
        client._chat_json("system", "user")

    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda payload: {"message": {"role": "assistant", "content": 123}},
    )
    with pytest.raises(ValueError, match="must be a string"):
        client._chat_json("system", "user")


def test_chat_json_uses_repair_path_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda payload: {"message": {"role": "assistant", "content": "broken"}},
    )
    monkeypatch.setattr(
        client,
        "_repair_json_content",
        lambda broken_content: '{"response":"fixed"}',
    )
    payload = client._chat_json("system", "user")
    assert payload["response"] == "fixed"


def test_post_chat_raises_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(
        adapter_module.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.HTTPError("boom")),
    )
    monkeypatch.setattr(adapter_module, "sleep", lambda seconds: None)
    with pytest.raises(httpx.HTTPError):
        client._post_chat({})


def test_post_chat_runtime_error_when_retry_loop_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(adapter_module, "range", lambda count: [], raising=False)
    with pytest.raises(RuntimeError, match="without response"):
        client._post_chat({})


def test_repair_json_content_validates_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(client, "_post_chat", lambda payload: {})
    with pytest.raises(ValueError, match="missing message"):
        client._repair_json_content("broken")

    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda payload: {"message": {"role": "assistant", "content": 123}},
    )
    with pytest.raises(ValueError, match="must be a string"):
        client._repair_json_content("broken")


def test_repair_json_content_returns_string_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda payload: {"message": {"role": "assistant", "content": '{"fixed":true}'}},
    )
    assert client._repair_json_content("broken") == '{"fixed":true}'


def test_generation_falls_back_when_chat_json_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch)
    monkeypatch.setattr(
        client,
        "_chat_json",
        lambda system_prompt, user_prompt: (_ for _ in ()).throw(ValueError("bad")),
    )
    objection = client.generate_objection_handler(
        provider_id="P001",
        concern="turnaround_time",
        kb_context="Average turnaround is 8 days.",
        citation_ids=["kb:1", "kb:2", "kb:3", "kb:4"],
        observed_metrics=["8 days", "99.1%", "24 hours", "extra"],
    )
    script = client.generate_meeting_script(
        provider_id="P001",
        tumor_focus="Lung",
        kb_context="Average turnaround is 8 days.",
        citation_ids=["kb:1", "kb:2", "kb:3", "kb:4"],
    )
    assert objection.response.startswith("Address turnaround_time")
    assert objection.supporting_metrics == ["8 days", "99.1%", "24 hours"]
    assert objection.citations == ["kb:1", "kb:2", "kb:3"]
    assert objection.confidence == 0.5
    assert script.script.startswith("Open with Lung focus")
    assert script.citations == ["kb:1", "kb:2", "kb:3"]
    assert script.confidence == 0.5


def test_generation_coerces_malformed_success_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(monkeypatch)
    payloads = [
        {
            "response": "Use evidence.",
            "supporting_metrics": ["8 days", 99, "24 hours"],
            "citations": ["kb:1", 2, "kb:2"],
            "confidence": {"score": "0.88"},
        },
        {
            "script": "Intro plan.",
            "citations": ["kb:1", 2, "kb:2"],
            "confidence": "0.82",
        },
    ]
    monkeypatch.setattr(client, "_chat_json", lambda system_prompt, user_prompt: payloads.pop(0))
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
    assert objection.supporting_metrics == ["8 days", "24 hours"]
    assert objection.citations == ["kb:1", "kb:2"]
    assert objection.confidence == 0.88
    assert script.citations == ["kb:1", "kb:2"]
    assert script.confidence == 0.82


def test_ollama_generation_client_generates_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
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

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr(adapter_module.httpx, "post", fake_post)
    client = OllamaGenerationClient(model="ministral-3:8b")
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

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("temporary")
        return FakeResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"response":"ok","supporting_metrics":[],"citations":["kb:1"],'
                        '"confidence":0.9}'
                    ),
                }
            }
        )

    monkeypatch.setattr(adapter_module.httpx, "post", fake_post)
    monkeypatch.setattr(adapter_module, "sleep", lambda seconds: None)
    client = OllamaGenerationClient(model="ministral-3:8b", request_retries=1)
    out = client.generate_objection_handler(
        provider_id="P001",
        concern="turnaround_time",
        kb_context="context",
        citation_ids=["kb:1"],
        observed_metrics=["8 days"],
    )
    assert calls["count"] == 2
    assert out.response == "ok"


def test_get_default_generation_client_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    client = get_default_generation_client(generation_provider="ollama")
    assert isinstance(client, OllamaGenerationClient)
    with pytest.raises(ValueError):
        get_default_generation_client(generation_provider="google")
