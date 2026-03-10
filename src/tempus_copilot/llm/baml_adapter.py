from __future__ import annotations

import json
from os import getenv
from time import sleep
from typing import Protocol, TypedDict, cast
from urllib.parse import urlparse

import httpx

from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact


class GenerationClient(Protocol):
    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact: ...

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact: ...


class OllamaMessage(TypedDict):
    role: str
    content: str


class OllamaChatResponse(TypedDict, total=False):
    message: OllamaMessage


class ObjectionPayload(TypedDict):
    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float


class ScriptPayload(TypedDict):
    script: str
    citations: list[str]
    confidence: float


def _normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1"}:
        raise ValueError("OLLAMA_BASE_URL must point to Ollama Cloud, not localhost")
    if parsed.scheme != "https":
        raise ValueError("OLLAMA_BASE_URL must use https for Ollama Cloud")
    return base_url.rstrip("/")


def _extract_json_payload(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    candidate = stripped[start : end + 1] if start >= 0 and end > start else stripped
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Model output must be a JSON object")
    return cast(dict[str, object], obj)


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
    return out


class OllamaGenerationClient:
    def __init__(
        self,
        model: str,
        request_retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        api_key = getenv("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY is required for Ollama Cloud generation")
        base_url = getenv("OLLAMA_BASE_URL") or "https://ollama.com"
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url)
        self._model = model
        self._request_retries = max(0, request_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        response: httpx.Response | None = None
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        for attempt in range(self._request_retries + 1):
            try:
                response = httpx.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=45.0,
                )
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt >= self._request_retries:
                    raise
                sleep(self._backoff_seconds * (attempt + 1))
        if response is None:
            raise RuntimeError("Ollama chat request failed without response")
        body = cast(OllamaChatResponse, response.json())
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama chat response missing message")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Ollama chat response content must be a string")
        return _extract_json_payload(content)

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        system_prompt = (
            "You are a clinical sales enablement assistant. "
            "Return strict JSON with keys: response, supporting_metrics, citations, confidence."
        )
        user_prompt = (
            f"Provider ID: {provider_id}\n"
            f"Concern: {concern}\n"
            f"Citation IDs: {citation_ids}\n"
            f"Observed metrics: {observed_metrics}\n"
            f"KB context:\n{kb_context}\n"
            "Return only JSON."
        )
        raw = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = cast(ObjectionPayload, raw)
        return ObjectionArtifact(
            provider_id=provider_id,
            concern=concern,
            response=str(payload.get("response", "")),
            supporting_metrics=_coerce_string_list(payload.get("supporting_metrics")),
            citations=_coerce_string_list(payload.get("citations")),
            confidence=float(payload.get("confidence", 0.0)),
        )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        system_prompt = (
            "You are a clinical sales enablement assistant. "
            "Return strict JSON with keys: script, citations, confidence."
        )
        user_prompt = (
            f"Provider ID: {provider_id}\n"
            f"Tumor focus: {tumor_focus}\n"
            f"Citation IDs: {citation_ids}\n"
            f"KB context:\n{kb_context}\n"
            "Return only JSON."
        )
        raw = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = cast(ScriptPayload, raw)
        return MeetingScriptArtifact(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=str(payload.get("script", "")),
            citations=_coerce_string_list(payload.get("citations")),
            confidence=float(payload.get("confidence", 0.0)),
        )


def get_default_generation_client(
    generation_provider: str = "ollama",
    generation_model: str = "qwen3.5:397b",
    request_retries: int = 2,
    backoff_seconds: float = 0.5,
) -> GenerationClient:
    provider = generation_provider.lower().strip()
    if provider != "ollama":
        raise ValueError("Only 'ollama' generation_provider is supported")
    return OllamaGenerationClient(
        model=generation_model,
        request_retries=request_retries,
        backoff_seconds=backoff_seconds,
    )
