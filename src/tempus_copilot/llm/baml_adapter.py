from __future__ import annotations

from os import getenv
from typing import Any, Protocol, runtime_checkable

from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact


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
        client: Any | None = None,
        fallback_client: GenerationClient | None = None,
        max_retries: int = 1,
    ) -> None:
        if client is None:
            from baml_client.sync_client import b as default_client

            self._client: Any = default_client
        else:
            self._client = client
        self._fallback = fallback_client or RuleBasedGenerationClient()
        self._max_retries = max(0, max_retries)

    def _call_with_retry(self, fn: Any, kwargs: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                return fn(**kwargs)
            except Exception as err:
                last_error = err
        if last_error is not None:
            raise last_error
        raise RuntimeError("BAML generation failed without exception details")

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        try:
            result = self._call_with_retry(
                self._client.GenerateObjectionHandler,
                {
                    "provider_id": provider_id,
                    "concern": concern,
                    "kb_context": kb_context,
                    "citation_ids": citation_ids,
                    "observed_metrics": observed_metrics,
                },
            )
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
            result = self._call_with_retry(
                self._client.GenerateMeetingScript,
                {
                    "provider_id": provider_id,
                    "tumor_focus": tumor_focus,
                    "kb_context": kb_context,
                    "citation_ids": citation_ids,
                },
            )
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


def get_default_generation_client() -> GenerationClient:
    if getenv("GOOGLE_API_KEY"):
        return BamlGenerationClient()
    return RuleBasedGenerationClient()
