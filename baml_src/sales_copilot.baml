class ObjectionHandler {
  provider_id string
  response string
}

class MeetingScript {
  provider_id string
  script string
}

function GenerateObjectionHandler(provider_id: string, concern: string, kb_context: string) -> ObjectionHandler {
  client "google-ai/gemini-2.5-flash"
  prompt #"
You are a Tempus Sales Copilot assistant.
Return a concise objection handler for provider {{ provider_id }}.
Concern: {{ concern }}
KB context: {{ kb_context }}
"#
}

function GenerateMeetingScript(provider_id: string, tumor_focus: string, kb_context: string) -> MeetingScript {
  client "google-ai/gemini-2.5-flash"
  prompt #"
You are a Tempus Sales Copilot assistant.
Write a 30-second tailored meeting script for provider {{ provider_id }}.
Tumor focus: {{ tumor_focus }}
KB context: {{ kb_context }}
"#
}
