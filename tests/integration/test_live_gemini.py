from os import getenv

import pytest

from tempus_copilot.llm.baml_adapter import BamlGenerationClient

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_live_baml_gemini_generation_round_trip() -> None:
    client = BamlGenerationClient()
    objection = client.generate_objection_handler(
        provider_id="P001",
        concern="turnaround_time",
        kb_context="Average turnaround is 8 days and sensitivity is 99.1%.",
        citation_ids=["product_kb.md:0"],
        observed_metrics=["8 days", "99.1%"],
    )
    assert objection.provider_id == "P001"
    assert 0.0 <= objection.confidence <= 1.0
