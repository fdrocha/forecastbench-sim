"""Pluggable template registry.

Each world builds a TemplateRegistry from its own QuestionTemplates and hands it
to the resolver/generator. Core never imports a world's templates directly — this
replaces the old global `get_template`/`TEMPLATES_BY_ID` lookup that hardcoded the
FreeCiv template set.
"""

from .schema import QuestionTemplate


class TemplateRegistry:
    def __init__(self, templates: list[QuestionTemplate] | None = None):
        self._by_id: dict[str, QuestionTemplate] = {}
        for t in templates or []:
            self.register(t)

    def register(self, template: QuestionTemplate) -> None:
        self._by_id[template.template_id] = template

    def get(self, template_id: str) -> QuestionTemplate:
        # Tolerate the "h0_" comprehension-question prefix.
        tid = template_id[3:] if template_id.startswith("h0_") else template_id
        if tid not in self._by_id:
            raise ValueError(f"Unknown template_id: {template_id!r}")
        return self._by_id[tid]

    def __contains__(self, template_id: str) -> bool:
        tid = template_id[3:] if template_id.startswith("h0_") else template_id
        return tid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)
