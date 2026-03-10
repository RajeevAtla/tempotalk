"""Typed Ollama Cloud generation client used by the pipeline."""

from __future__ import annotations

import json
from os import getenv
from time import sleep
from typing import Protocol, TypedDict, cast
from urllib.parse import urlparse

import httpx

from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact


class GenerationClient(Protocol):
    """Protocol for generation clients used during pipeline execution."""

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        """Generate an objection-handling artifact for a provider.

        Args:
            provider_id: Provider identifier.
            concern: CRM concern to address.
            kb_context: Retrieved KB text supplied to the model.
            citation_ids: Allowed citation identifiers from retrieval.
            observed_metrics: Metrics extracted from the retrieval context.

        Returns:
            A typed objection-handling artifact.
        """
        ...

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        """Generate a meeting script artifact for a provider.

        Args:
            provider_id: Provider identifier.
            tumor_focus: Provider tumor focus.
            kb_context: Retrieved KB text supplied to the model.
            citation_ids: Allowed citation identifiers from retrieval.

        Returns:
            A typed meeting script artifact.
        """
        ...


class OllamaMessage(TypedDict):
    """Chat message payload exchanged with Ollama."""

    role: str
    content: str


class OllamaChatResponse(TypedDict, total=False):
    """Subset of the Ollama chat response used by the client."""

    message: OllamaMessage


class ObjectionPayload(TypedDict):
    """Expected model payload for objection handling output."""

    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float


class ScriptPayload(TypedDict):
    """Expected model payload for meeting script output."""

    script: str
    citations: list[str]
    confidence: float


def _normalize_base_url(base_url: str) -> str:
    """Validate and normalize the Ollama Cloud base URL.

    Args:
        base_url: Candidate base URL for the generation service.

    Returns:
        The normalized base URL without a trailing slash.

    Raises:
        ValueError: If the URL points to localhost or does not use HTTPS.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1"}:
        raise ValueError("OLLAMA_BASE_URL must point to Ollama Cloud, not localhost")
    if parsed.scheme != "https":
        raise ValueError("OLLAMA_BASE_URL must use https for Ollama Cloud")
    return base_url.rstrip("/")


def _extract_json_payload(text: str) -> dict[str, object]:
    """Extract the first JSON object from model output text.

    Args:
        text: Raw model output.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the parsed payload is not a JSON object.
        json.JSONDecodeError: If the content cannot be repaired into JSON.
    """
    stripped = text.strip()
    # Models sometimes wrap JSON in fences or preamble text even when asked not to.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    candidate = stripped[start : end + 1] if start >= 0 and end > start else stripped
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        sanitized = "".join(ch if ord(ch) >= 32 else " " for ch in candidate)
        obj = json.loads(sanitized)
    if not isinstance(obj, dict):
        raise ValueError("Model output must be a JSON object")
    return cast(dict[str, object], obj)


def _coerce_string_list(value: object) -> list[str]:
    """Coerce a dynamic value into a list of strings.

    Args:
        value: Dynamic model field value.

    Returns:
        The string members of the input list, or an empty list.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
    return out


def _coerce_confidence(value: object) -> float:
    """Coerce a dynamic value into a confidence score.

    Args:
        value: Dynamic model field value.

    Returns:
        A floating-point confidence score.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        for key in ("confidence", "score", "value"):
            candidate = mapping.get(key)
            if isinstance(candidate, (int, float, str)):
                return _coerce_confidence(candidate)
    return 0.0


class OllamaGenerationClient:
    """Generation client backed by Ollama Cloud chat completions."""

    def __init__(
        self,
        model: str,
        request_retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        """Initialize the generation client.

        Args:
            model: Generation model name.
            request_retries: Number of retry attempts after the initial request.
            backoff_seconds: Base backoff delay between retries.

        Raises:
            RuntimeError: If the Ollama API key is missing.
            ValueError: If the configured base URL violates runtime policy.
        """
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
        """Request a JSON response from the model.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User payload passed to the model.

        Returns:
            Parsed JSON object from the model response.

        Raises:
            ValueError: If the Ollama response shape is invalid.
            json.JSONDecodeError: If neither the original nor repaired content is valid JSON.
        """
        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = self._post_chat(payload)
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama chat response missing message")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Ollama chat response content must be a string")
        try:
            return _extract_json_payload(content)
        except (json.JSONDecodeError, ValueError):
            # One repair pass keeps the happy path strict without masking persistent failures.
            repaired = self._repair_json_content(content)
            return _extract_json_payload(repaired)

    def _post_chat(self, payload: dict[str, object]) -> OllamaChatResponse:
        """Send a chat request with retry handling.

        Args:
            payload: JSON payload to send to Ollama.

        Returns:
            Parsed Ollama chat response.

        Raises:
            RuntimeError: If all attempts fail without returning a response object.
            httpx.HTTPError: If the final request fails.
        """
        response: httpx.Response | None = None
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
        return cast(OllamaChatResponse, response.json())

    def _repair_json_content(self, broken_content: str) -> str:
        """Ask the model to repair malformed JSON.

        Args:
            broken_content: Model output that failed JSON parsing.

        Returns:
            Repaired JSON string content.

        Raises:
            ValueError: If the repair response shape is invalid.
        """
        payload: dict[str, object] = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Repair malformed JSON. Return only valid JSON preserving intent and keys."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Fix this to strict JSON only:\n"
                        f"{broken_content}"
                    ),
                },
            ],
        }
        body = self._post_chat(payload)
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama repair response missing message")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Ollama repair response content must be a string")
        return content

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        """Generate an objection-handling artifact for a provider.

        Args:
            provider_id: Provider identifier.
            concern: CRM concern to address.
            kb_context: Retrieved KB text supplied to the model.
            citation_ids: Allowed citation identifiers from retrieval.
            observed_metrics: Metrics extracted from the retrieval context.

        Returns:
            A typed objection-handling artifact.
        """
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
        try:
            raw = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
            payload = cast(ObjectionPayload, raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fall back to a deterministic template so the pipeline still
            # produces reviewable output.
            return ObjectionArtifact(
                provider_id=provider_id,
                concern=concern,
                response=(
                    f"Address {concern} for {provider_id} using validated metrics and KB evidence."
                ),
                supporting_metrics=observed_metrics[:3],
                citations=citation_ids[:3],
                confidence=0.5,
            )
        return ObjectionArtifact(
            provider_id=provider_id,
            concern=concern,
            response=str(payload.get("response", "")),
            supporting_metrics=_coerce_string_list(payload.get("supporting_metrics")),
            citations=_coerce_string_list(payload.get("citations")),
            confidence=_coerce_confidence(payload.get("confidence", 0.0)),
        )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        """Generate a meeting script artifact for a provider.

        Args:
            provider_id: Provider identifier.
            tumor_focus: Provider tumor focus.
            kb_context: Retrieved KB text supplied to the model.
            citation_ids: Allowed citation identifiers from retrieval.

        Returns:
            A typed meeting script artifact.
        """
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
        try:
            raw = self._chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
            payload = cast(ScriptPayload, raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fall back to a deterministic template so the pipeline still
            # produces reviewable output.
            return MeetingScriptArtifact(
                provider_id=provider_id,
                tumor_focus=tumor_focus,
                script=(
                    "Open with "
                    f"{tumor_focus} focus, cite strongest evidence, and close on next step."
                ),
                citations=citation_ids[:3],
                confidence=0.5,
            )
        return MeetingScriptArtifact(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=str(payload.get("script", "")),
            citations=_coerce_string_list(payload.get("citations")),
            confidence=_coerce_confidence(payload.get("confidence", 0.0)),
        )


def get_default_generation_client(
    generation_provider: str = "ollama",
    generation_model: str = "ministral-3:8b",
    request_retries: int = 2,
    backoff_seconds: float = 0.5,
) -> GenerationClient:
    """Construct the default generation client for configured runtime policy.

    Args:
        generation_provider: Generation provider name.
        generation_model: Generation model name.
        request_retries: Number of retry attempts after the initial request.
        backoff_seconds: Base backoff delay between retries.

    Returns:
        A configured generation client.

    Raises:
        ValueError: If the provider is unsupported.
    """
    provider = generation_provider.lower().strip()
    if provider != "ollama":
        raise ValueError("Only 'ollama' generation_provider is supported")
    return OllamaGenerationClient(
        model=generation_model,
        request_retries=request_retries,
        backoff_seconds=backoff_seconds,
    )
