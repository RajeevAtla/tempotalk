from __future__ import annotations

import json
from os import getenv
import re
from time import sleep
from typing import Callable, Mapping, Protocol, Sequence, TypeVar, cast, runtime_checkable

import httpx

from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact

_ResultT = TypeVar("_ResultT")


class BamlObjectionResponse(Protocol):
    provider_id: object
    concern: object
    response: object
    supporting_metrics: Sequence[object]
    citations: Sequence[object]
    confidence: float | int | str


class BamlMeetingScriptResponse(Protocol):
    provider_id: object
    tumor_focus: object
    script: object
    citations: Sequence[object]
    confidence: float | int | str


@runtime_checkable
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


class BamlGenerationClient:
    def __init__(
        self,
        client: object | None = None,
        fallback_client: GenerationClient | None = None,
        max_retries: int = 1,
    ) -> None:
        if client is None:
            from baml_client.sync_client import b as default_client

            self._client: object = default_client
        else:
            self._client = client
        self._fallback = fallback_client or RuleBasedGenerationClient()
        self._max_retries = max(0, max_retries)

    def _call_with_retry(
        self, fn: Callable[..., _ResultT], kwargs: Mapping[str, object]
    ) -> _ResultT:
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                return fn(**dict(kwargs))
            except Exception as err:
                last_error = err
        if last_error is not None:
            raise last_error
        raise RuntimeError("BAML generation failed without exception details")

    def _invoke(self, method_name: str, kwargs: Mapping[str, object]) -> object:
        method = getattr(self._client, method_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"Client method {method_name} is unavailable")
        return self._call_with_retry(cast(Callable[..., object], method), kwargs)

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        try:
            raw = self._invoke(
                "GenerateObjectionHandler",
                {
                    "provider_id": provider_id,
                    "concern": concern,
                    "kb_context": kb_context,
                    "citation_ids": citation_ids,
                    "observed_metrics": observed_metrics,
                },
            )
            result = cast(BamlObjectionResponse, raw)
            return ObjectionArtifact(
                provider_id=str(result.provider_id),
                concern=str(result.concern),
                response=str(result.response),
                supporting_metrics=[str(item) for item in result.supporting_metrics],
                citations=[str(item) for item in result.citations],
                confidence=float(result.confidence),
            )
        except Exception:
            return self._fallback.generate_objection_handler(
                provider_id=provider_id,
                concern=concern,
                kb_context=kb_context,
                citation_ids=citation_ids,
                observed_metrics=observed_metrics,
            )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        try:
            raw = self._invoke(
                "GenerateMeetingScript",
                {
                    "provider_id": provider_id,
                    "tumor_focus": tumor_focus,
                    "kb_context": kb_context,
                    "citation_ids": citation_ids,
                },
            )
            result = cast(BamlMeetingScriptResponse, raw)
            return MeetingScriptArtifact(
                provider_id=str(result.provider_id),
                tumor_focus=str(result.tumor_focus),
                script=str(result.script),
                citations=[str(item) for item in result.citations],
                confidence=float(result.confidence),
            )
        except Exception:
            return self._fallback.generate_meeting_script(
                provider_id=provider_id,
                tumor_focus=tumor_focus,
                kb_context=kb_context,
                citation_ids=citation_ids,
            )


class RuleBasedGenerationClient:
    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        return ObjectionArtifact(
            provider_id=provider_id,
            concern=concern,
            response=f"{provider_id}: {concern} response based on {kb_context[:160]}",
            supporting_metrics=observed_metrics[:5],
            citations=citation_ids,
            confidence=0.72 if citation_ids else 0.45,
        )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        return MeetingScriptArtifact(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=f"{provider_id}: {tumor_focus} pitch using {kb_context[:160]}",
            citations=citation_ids,
            confidence=0.7 if citation_ids else 0.4,
        )


class OllamaGenerationClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        fallback_client: GenerationClient | None = None,
        request_retries: int = 1,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._model = model
        self._base_url = (base_url or getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self._fallback = fallback_client or RuleBasedGenerationClient()
        self._request_retries = max(0, request_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    def _request_json(self, prompt: str) -> dict[str, object]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        last_error: Exception | None = None
        for attempt in range(self._request_retries + 1):
            try:
                response = httpx.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                outer = response.json()
                raw = str(outer.get("response", "")).strip()
                if not raw:
                    raise ValueError("Ollama response is empty")
                try:
                    return cast(dict[str, object], json.loads(raw))
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                    if match is None:
                        raise
                    return cast(dict[str, object], json.loads(match.group(0)))
            except Exception as err:
                last_error = err
                if attempt >= self._request_retries:
                    break
                sleep(self._backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama generation failed without error")

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        prompt = (
            "Return JSON only with keys provider_id, concern, response, supporting_metrics, citations, confidence.\n"
            f"provider_id={provider_id}\n"
            f"concern={concern}\n"
            f"allowed_citations={citation_ids}\n"
            f"observed_metrics={observed_metrics}\n"
            f"kb_context={kb_context}\n"
        )
        try:
            obj = self._request_json(prompt)
            return ObjectionArtifact(
                provider_id=str(obj.get("provider_id", provider_id)),
                concern=str(obj.get("concern", concern)),
                response=str(obj.get("response", "")),
                supporting_metrics=[str(v) for v in cast(list[object], obj.get("supporting_metrics", []))],
                citations=[str(v) for v in cast(list[object], obj.get("citations", []))],
                confidence=float(obj.get("confidence", 0.65)),
            )
        except Exception:
            return self._fallback.generate_objection_handler(
                provider_id=provider_id,
                concern=concern,
                kb_context=kb_context,
                citation_ids=citation_ids,
                observed_metrics=observed_metrics,
            )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        prompt = (
            "Return JSON only with keys provider_id, tumor_focus, script, citations, confidence.\n"
            f"provider_id={provider_id}\n"
            f"tumor_focus={tumor_focus}\n"
            f"allowed_citations={citation_ids}\n"
            f"kb_context={kb_context}\n"
        )
        try:
            obj = self._request_json(prompt)
            return MeetingScriptArtifact(
                provider_id=str(obj.get("provider_id", provider_id)),
                tumor_focus=str(obj.get("tumor_focus", tumor_focus)),
                script=str(obj.get("script", "")),
                citations=[str(v) for v in cast(list[object], obj.get("citations", []))],
                confidence=float(obj.get("confidence", 0.65)),
            )
        except Exception:
            return self._fallback.generate_meeting_script(
                provider_id=provider_id,
                tumor_focus=tumor_focus,
                kb_context=kb_context,
                citation_ids=citation_ids,
            )


def get_default_generation_client(
    generation_provider: str,
    generation_model: str,
    request_retries: int = 1,
    backoff_seconds: float = 0.5,
) -> GenerationClient:
    provider = generation_provider.lower().strip()
    if provider == "ollama":
        return OllamaGenerationClient(
            model=generation_model,
            request_retries=request_retries,
            backoff_seconds=backoff_seconds,
        )
    if provider in {"google", "baml"} and getenv("GOOGLE_API_KEY"):
        return BamlGenerationClient(max_retries=request_retries)
    return RuleBasedGenerationClient()
