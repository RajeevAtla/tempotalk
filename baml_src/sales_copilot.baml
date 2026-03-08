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
  client "google-ai/gemini-2.5-flash"
  prompt #"
You are a Tempus Sales Copilot assistant.
Return a concise objection handler for provider {{ provider_id }} using structured output.
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
  client "google-ai/gemini-2.5-flash"
  prompt #"
You are a Tempus Sales Copilot assistant.
Write a 30-second tailored meeting script for provider {{ provider_id }}.
Tumor focus: {{ tumor_focus }}
KB context: {{ kb_context }}
Allowed citations: {{ citation_ids }}
"#
}
