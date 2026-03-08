from types import SimpleNamespace

from tempus_copilot.llm.baml_adapter import BamlGenerationClient


class FakeBamlClient:
    def GenerateObjectionHandler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> object:
        return SimpleNamespace(
            provider_id=provider_id,
            concern=concern,
            response=f"{provider_id}:{concern}:{kb_context[:12]}",
            supporting_metrics=observed_metrics,
            citations=citation_ids,
            confidence=0.88,
        )

    def GenerateMeetingScript(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> object:
        return SimpleNamespace(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=f"{provider_id}:{tumor_focus}:{kb_context[:12]}",
            citations=citation_ids,
            confidence=0.83,
        )


def test_baml_adapter_uses_generated_client_shape() -> None:
    adapter = BamlGenerationClient(client=FakeBamlClient())
    objection = adapter.generate_objection_handler(
        "P001",
        "turnaround_time",
        "context here",
        citation_ids=["kb:1"],
        observed_metrics=["8 days"],
    )
    script = adapter.generate_meeting_script("P001", "Lung", "context here", citation_ids=["kb:1"])
    assert objection.response.startswith("P001:turnaround_time")
    assert script.script.startswith("P001:Lung")
    assert objection.citations == ["kb:1"]
