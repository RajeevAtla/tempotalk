from tempus_copilot.llm.baml_adapter import BamlGenerationClient


class FakeBamlClient:
    def GenerateObjectionHandler(self, provider_id: str, concern: str, kb_context: str) -> object:
        class Resp:
            response = f"{provider_id}:{concern}:{kb_context[:12]}"

        return Resp()

    def GenerateMeetingScript(self, provider_id: str, tumor_focus: str, kb_context: str) -> object:
        class Resp:
            script = f"{provider_id}:{tumor_focus}:{kb_context[:12]}"

        return Resp()


def test_baml_adapter_uses_generated_client_shape() -> None:
    adapter = BamlGenerationClient(client=FakeBamlClient())
    objection = adapter.generate_objection_handler("P001", "turnaround_time", "context here")
    script = adapter.generate_meeting_script("P001", "Lung", "context here")
    assert objection.startswith("P001:turnaround_time")
    assert script.startswith("P001:Lung")
