from os import getenv

import pytest

from tempus_copilot.llm.baml_adapter import OllamaGenerationClient


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not getenv("RUN_OLLAMA_INTEGRATION"), reason="RUN_OLLAMA_INTEGRATION not set")
def test_live_ollama_generation_round_trip() -> None:
    client = OllamaGenerationClient(model="qwen3.5:0.8b")
    objection = client.generate_objection_handler(
        provider_id="P001",
        concern="turnaround_time",
        kb_context="Average turnaround is 8 days and sensitivity is 99.1%.",
        citation_ids=["product_kb.md:0"],
        observed_metrics=["8 days", "99.1%"],
    )
    assert objection.provider_id == "P001"
    assert 0.0 <= objection.confidence <= 1.0
