# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import dataclasses
import random

# Local
from fms_dgt.base.registry import get_datastore
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.databuilders.conversation.data_objects import (
    BigFiveProfile,
    ConversationDataPoint,
    PersonaSpec,
    PersonaStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage


def render_persona(step: PersonaStep) -> str:
    """Render a PersonaStep's structured fields into a prompt-ready string.

    Consumer stages call this to get the text they pass to the LLM. Keeping
    rendering here (rather than in each consumer) avoids duplication while
    allowing callers to render on demand rather than being locked into whatever
    string was pre-rendered at write time.
    """
    parts = []
    if step.persona_role:
        parts.append(f"Role: {step.persona_role}.")
    if step.expertise:
        parts.append(f"Expertise: {step.expertise}.")
    if step.domain:
        parts.append(f"Domain: {step.domain}.")
    if step.goals:
        parts.append("Goals: " + "; ".join(step.goals) + ".")
    if step.style_override:
        parts.append(f"Style: {step.style_override}.")
    else:
        p = step.personality
        traits = []
        if p.extraversion > 0.3:
            traits.append("expressive and enthusiastic")
        elif p.extraversion < -0.3:
            traits.append("reserved and terse")
        if p.agreeableness < -0.3:
            traits.append("skeptical and challenging")
        elif p.agreeableness > 0.3:
            traits.append("cooperative and accommodating")
        if p.neuroticism > 0.3:
            traits.append("anxious and sometimes frustrated")
        if traits:
            parts.append("Personality: " + ", ".join(traits) + ".")
    return " ".join(parts)


def _persona_spec_from_dict(record: Dict) -> PersonaSpec:
    """Deserialize a JSONL record into a PersonaSpec.

    Handles the nested BigFiveProfile dict under the ``personality`` key.
    Unknown keys are silently ignored so the JSONL schema can evolve without
    breaking older stage versions.
    """
    known_fields = {f.name for f in dataclasses.fields(PersonaSpec)}
    kwargs = {k: v for k, v in record.items() if k in known_fields}

    if "personality" in kwargs and isinstance(kwargs["personality"], dict):
        p = kwargs["personality"]
        known_p = {f.name for f in dataclasses.fields(BigFiveProfile)}
        kwargs["personality"] = BigFiveProfile(**{k: v for k, v in p.items() if k in known_p})

    return PersonaSpec(**kwargs)


@register_stage("sample/persona")
class SamplePersonaStage(Stage):
    """Initialization stage that samples a PersonaSpec and appends a PersonaStep.

    Zero LM calls. Loads personas from a JSONL file via a datastore. The
    PersonaStep is written with target="user".

    Structured PersonaSpec fields are stored as first-class fields on the step.
    ``content`` is set to the rendered string that would be passed to an LLM,
    useful for debugging and tracing what downstream stages actually saw.

    Config kwargs:
        persona_store: Datastore config dict with at least a ``type`` key and
            a ``data_path`` pointing to a JSONL file of persona records. This
            field is required; omitting it raises ValueError at init time.
    """

    def __init__(
        self,
        *,
        name: str,
        persona_store: Dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)

        if not persona_store:
            raise ValueError(
                f"Stage '{name}' requires a 'persona_store' config dict with a "
                "'data_path' pointing to a JSONL file of persona records. "
                "See data/public/examples/chit_chat/personas.jsonl for an example."
            )

        store = get_datastore(
            persona_store.get(TYPE_KEY, "default"),
            store_name=name,
            **{k: v for k, v in persona_store.items() if k != TYPE_KEY},
        )
        records = store.load_data()

        if not records:
            raise ValueError(
                f"Stage '{name}': persona_store loaded 0 records from "
                f"'{persona_store.get('data_path', '(no data_path)')}'. "
                "Ensure the file exists and is non-empty."
            )

        self._personas: List[PersonaSpec] = [_persona_spec_from_dict(r) for r in records]

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        for data_point in data_points:
            spec = random.choice(self._personas)
            step = PersonaStep(
                content="",  # filled below after render_persona has the step
                stage_name=self.name,
                target="user",
                persona_role=spec.role,
                expertise=spec.expertise,
                domain=spec.domain,
                goals=list(spec.goals),
                personality=spec.personality,
                style_override=spec.style_override,
            )
            step.content = render_persona(step)
            data_point.steps.append(step)
        return data_points
