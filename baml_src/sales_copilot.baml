class ObjectionHandler {
  provider_id string
  concern string
  response string
  supporting_metrics string[]
  citations string[]
  confidence float
}

class MeetingScript {
  provider_id string
  tumor_focus string
  script string
  citations string[]
  confidence float
}

function GenerateObjectionHandler(
  provider_id: string,
  concern: string,
  kb_context: string,
  citation_ids: string[],
  observed_metrics: string[]
) -> ObjectionHandler {
  client "ollama/qwen3.5:397b"
  prompt #"
You are a Tempus Sales Copilot assistant.
Return ONLY a valid JSON object that matches this exact schema:
{
  \"provider_id\": string,
  \"concern\": string,
  \"response\": string,
  \"supporting_metrics\": string[],
  \"citations\": string[],
  \"confidence\": float
}
Do not include markdown, code fences, headings, or extra keys.
Ensure citations are chosen only from Allowed citations.
Confidence must be a numeric value between 0.0 and 1.0.
Return a concise objection handler for provider {{ provider_id }}.
Concern: {{ concern }}
KB context: {{ kb_context }}
Allowed citations: {{ citation_ids }}
Observed metrics: {{ observed_metrics }}
"#
}

function GenerateMeetingScript(
  provider_id: string,
  tumor_focus: string,
  kb_context: string,
  citation_ids: string[]
) -> MeetingScript {
  client "ollama/qwen3.5:397b"
  prompt #"
You are a Tempus Sales Copilot assistant.
Return ONLY a valid JSON object that matches this exact schema:
{
  \"provider_id\": string,
  \"tumor_focus\": string,
  \"script\": string,
  \"citations\": string[],
  \"confidence\": float
}
Do not include markdown, code fences, headings, or extra keys.
Ensure citations are chosen only from Allowed citations.
Confidence must be a numeric value between 0.0 and 1.0.
Write a 30-second tailored meeting script for provider {{ provider_id }}.
Tumor focus: {{ tumor_focus }}
KB context: {{ kb_context }}
Allowed citations: {{ citation_ids }}
"#
}
