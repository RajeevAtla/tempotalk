from __future__ import annotations

from tempus_copilot.models import CRMNote, KBDocument, ProviderRecord


class BamlAdapter:
    def generate_objection_handler(
        self,
        provider: ProviderRecord,
        notes: list[CRMNote],
        kb: list[KBDocument],
    ) -> str:
        note = notes[0].concern_type if notes else "general concerns"
        kb_blurb = kb[0].text[:120].replace("\n", " ") if kb else ""
        return (
            f"For {provider.physician_name}, address '{note}' with evidence from KB: "
            f"{kb_blurb}"
        )

    def generate_meeting_script(self, provider: ProviderRecord, kb: list[KBDocument]) -> str:
        kb_blurb = kb[0].text[:120].replace("\n", " ") if kb else ""
        last_name = provider.physician_name.split(" ", 1)[-1]
        return (
            f"Dr. {last_name}, based on your {provider.tumor_focus} volume, "
            f"Tempus can support decision speed with validated performance. {kb_blurb}"
        )
