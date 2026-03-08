from __future__ import annotations

from os import getenv
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GenerationClient(Protocol):
    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
    ) -> str: ...

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
    ) -> str: ...


class BamlGenerationClient:
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from baml_client.sync_client import b as default_client

            self._client: Any = default_client
        else:
            self._client = client

    def generate_objection_handler(self, provider_id: str, concern: str, kb_context: str) -> str:
        result = self._client.GenerateObjectionHandler(
            provider_id=provider_id,
            concern=concern,
            kb_context=kb_context,
        )
        return str(result.response)

    def generate_meeting_script(self, provider_id: str, tumor_focus: str, kb_context: str) -> str:
        result = self._client.GenerateMeetingScript(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            kb_context=kb_context,
        )
        return str(result.script)


class RuleBasedGenerationClient:
    def generate_objection_handler(self, provider_id: str, concern: str, kb_context: str) -> str:
        return f"{provider_id}: {concern} response based on {kb_context[:160]}"

    def generate_meeting_script(self, provider_id: str, tumor_focus: str, kb_context: str) -> str:
        return f"{provider_id}: {tumor_focus} pitch using {kb_context[:160]}"


def get_default_generation_client() -> GenerationClient:
    if getenv("GOOGLE_API_KEY"):
        return BamlGenerationClient()
    return RuleBasedGenerationClient()
