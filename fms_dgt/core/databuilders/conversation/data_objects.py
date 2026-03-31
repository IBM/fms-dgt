# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
import dataclasses
import uuid

# Local
from fms_dgt.base.data_objects import DataPoint


# ===========================================================================
#                       STEP (base)
# ===========================================================================
@dataclass
class Step:
    """A single unit of stage output appended to a conversation.

    One stage invocation appends one Step to ConversationDataPoint.steps. A
    conversational turn (user + assistant exchange) spans multiple steps:
    typically a FlowControllerStep, a UserStep, an AssistantStep, and zero
    or more ToolCallStep/ToolResultStep objects. Steps are not turns.

    Each reserved role has a typed subclass with first-class fields for that
    role's structured data. Subclasses register themselves with @register_step
    so that Step.from_dict() can reconstruct the correct type on deserialization.
    The `role` field is the discriminator — it must match the key passed to
    @register_step. All subclass fields must have defaults so that old seeds
    (which may predate those fields) deserialize safely.

    Validator stages write scores into `scores` on the step they evaluated:
        [s.scores for s in context.steps if s.role == "assistant"]
    """

    role: str
    """Semantic role of this step. Each reserved role maps to a typed subclass:
      "user"            — UserStep
      "assistant"       — AssistantStep
      "tool_call"       — ToolCallStep (one step per call)
      "tool_result"     — ToolResultStep (paired with preceding tool_call)
      "flow_controller" — FlowControllerStep
      "scenario"        — ScenarioStep (written once during initialization)
      "persona"         — PersonaStep
    Recipe authors may define additional roles and register custom subclasses
    with @register_step; unrecognized roles fall back to base Step.

    There is no framework-level grounding step. Static initialization context
    (tool schemas, document corpora) belongs on a recipe-specific ScenarioStep
    subclass. Mid-conversation retrieval results are modeled as ToolCallStep +
    ToolResultStep pairs, which produces the correct training signal for
    retrieval-augmented and tool-calling models alike.
    """

    content: str | list | dict
    """The message text or structured output produced by the stage.

    str   — natural-language turns (user, assistant, scenario, persona, etc.)
    dict  — single structured object (one tool call, one tool result, etc.)
    list  — sequence of structured objects (parallel tool calls, multi-doc grounding)
    """

    stage_name: str | None = None
    """Internal stage registry identifier that produced this step
    (e.g. "lm/user/guided", "sample/persona"). Useful for debugging and
    telemetry. Not a semantic field — use `role` to classify the step."""

    scores: Dict[str, float] = field(default_factory=dict)
    """Quality scores written by validator stages. Keyed by validator name.
    Example: {"tool_call_validator": 0.91, "safety_judge": 1.0}"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary stage-specific metadata that does not warrant a first-class
    field (e.g. parse details, raw LM output). Prefer typed subclass fields
    over metadata for anything stages query by name."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this step to a plain dict for storage.

        `role` is the type discriminator: Step.from_dict() uses it to
        reconstruct the correct subclass via the step registry.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step":
        """Deserialize a step dict, reconstructing the correct subclass.

        Dispatches on `role` via the step registry. If the resolved subclass
        overrides `from_dict` (e.g. to handle nested dataclasses), that
        override is called. Unrecognized roles fall back to the base Step
        class so seed data with unknown custom roles deserializes safely.
        """
        # Deferred import: data_objects → registry → stages.base → data_objects is a
        # circular dependency. Keeping this import inside the method breaks the cycle
        # because it only executes after all three modules have finished loading.
        # Do NOT move this import to the top of the file.
        # Local
        from fms_dgt.core.databuilders.conversation.registry import (  # noqa: PLC0415
            get_step,
        )

        role = d.get("role", "")
        target_cls = get_step(role)
        # Typed subclasses have `role` as init=False so it must be stripped
        # before passing kwargs to their constructor. Base Step takes role as
        # a normal init arg so it is kept.
        kwargs = {k: v for k, v in d.items() if k != "role"} if target_cls is not Step else d
        if target_cls is not Step and "from_dict" in target_cls.__dict__:
            return target_cls.from_dict(kwargs)
        return target_cls(**kwargs)


# ===========================================================================
#                       PERSONA PRIMITIVES
# ===========================================================================
# Defined before the step subclasses because PersonaStep references BigFiveProfile.


@dataclass
class BigFiveProfile:
    """OCEAN personality dimensions.

    Each dimension is a float in [-1.0, 1.0]. 0.0 means unspecified: the
    stage does not constrain that dimension. Set only the dimensions that
    matter for the recipe; unset dimensions are unconstrained.
    """

    openness: float = 0.0
    """−1: conventional, resistant to novelty. +1: curious, creative, open."""

    conscientiousness: float = 0.0
    """−1: spontaneous, flexible, disorganized. +1: methodical, thorough."""

    extraversion: float = 0.0
    """−1: reserved, terse, low-energy. +1: expressive, verbose, enthusiastic."""

    agreeableness: float = 0.0
    """−1: challenging, skeptical, adversarial. +1: cooperative, accommodating."""

    neuroticism: float = 0.0
    """−1: calm, stable, confident. +1: anxious, frustrated, volatile."""


@dataclass
class PersonaSpec:
    """Structured persona definition used by persona stages.

    Persona stages (sample/persona, lm/persona) build a PersonaSpec and
    populate a PersonaStep with its fields directly.

    Appending a new PersonaStep mid-conversation shifts the active persona;
    user and assistant stages always read the last step matching their target.
    """

    role: str | None = None
    """e.g. "enterprise IT manager", "frustrated customer"."""

    expertise: str | None = None
    """e.g. "intermediate", "low", "expert"."""

    domain: str | None = None
    """e.g. "cloud infrastructure", "e-commerce"."""

    goals: List[str] = field(default_factory=list)

    personality: BigFiveProfile = field(default_factory=BigFiveProfile)
    """OCEAN dimensions. Set only the dimensions that matter; others default to 0.0."""

    style_override: str | None = None
    """Free-text style description. Takes precedence over personality dimensions
    when both are set. Use for nuanced descriptions that don't map to OCEAN
    (e.g. "speaks in short sentences, uses bullet points")."""


# ===========================================================================
#                       TYPED STEP SUBCLASSES
# ===========================================================================
# Import deferred to after class definitions to avoid circular imports at
# module load time; registration happens at the bottom of this file.


@dataclass
class UserStep(Step):
    """A natural-language user turn."""

    role: str = field(default="user", init=False)


@dataclass
class AssistantStep(Step):
    """A natural-language assistant turn."""

    role: str = field(default="assistant", init=False)


@dataclass
class ToolCallStep(Step):
    """A single tool call produced by an assistant stage.

    One ToolCallStep per tool call. Parallel tool calls from one assistant
    turn produce multiple consecutive ToolCallStep objects.
    """

    role: str = field(default="tool_call", init=False)


@dataclass
class ToolResultStep(Step):
    """Tool execution result paired with the preceding ToolCallStep."""

    role: str = field(default="tool_result", init=False)


@dataclass
class FlowControllerStep(Step):
    """Interaction pattern and termination signal from the flow controller.

    The iteration loop checks `terminate` after each stage in the iteration
    sequence. The user stage reads `hint` to guide turn generation.
    """

    role: str = field(default="flow_controller", init=False)

    terminate: bool = False
    """When True the iteration loop yields and closes this conversation."""

    hint: str | None = None
    """Optional guidance text passed to the user stage for this iteration."""


@dataclass
class ScenarioStep(Step):
    """Conversation scenario written once during initialization.

    `scenario_family_id` groups conversations sharing the same initialization
    seed. The serializer reads it when grouping DPO pairing and ROLLOUT
    trajectory candidates.
    """

    role: str = field(default="scenario", init=False)

    scenario_family_id: str | None = None


@dataclass
class PersonaStep(Step):
    """User or assistant persona description.

    First-class fields carry the structured PersonaSpec so that downstream
    stages (user turn generator, persona consistency validator, etc.) can
    render what they need rather than being coupled to a single pre-rendered
    string. `content` holds the rendered text that was actually passed to the
    LLM — useful for debugging and tracing what the model saw.

    Stages query by both role and target:
        persona_steps = [
            s for s in context.steps
            if s.role == "persona" and s.target == "user"
        ]

    Appending a new PersonaStep mid-conversation shifts the active persona;
    stages always read the last matching step.

    Overrides to_dict/from_dict to handle the nested BigFiveProfile dataclass.
    Custom Step subclasses with other nested dataclasses should do the same.
    """

    role: str = field(default="persona", init=False)

    target: Literal["user", "assistant"] = "user"
    """Which conversational participant this persona describes."""

    # PersonaSpec fields promoted to first class so stages can query and
    # render them selectively rather than parsing a pre-rendered string.
    persona_role: str | None = None
    expertise: str | None = None
    domain: str | None = None
    goals: List[str] = field(default_factory=list)
    personality: BigFiveProfile = field(default_factory=BigFiveProfile)
    style_override: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        # BigFiveProfile serializes correctly via asdict (it's a plain dataclass),
        # but we record its type so from_dict can reconstruct it unambiguously.
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaStep":
        kwargs = dict(d)
        kwargs.pop("role", None)
        personality_raw = kwargs.pop("personality", None)
        if isinstance(personality_raw, dict):
            personality = BigFiveProfile(**personality_raw)
        elif isinstance(personality_raw, BigFiveProfile):
            personality = personality_raw
        else:
            personality = BigFiveProfile()
        return cls(personality=personality, **kwargs)


# ===========================================================================
#                       BRANCH POINT
# ===========================================================================
@dataclass
class BranchPoint:
    """Provenance record for a rejected (degraded) conversation branch.

    Set on a rejected ConversationDataPoint when preference data generation is
    active. None on all SFT conversations.
    """

    parent_conversation_id: str
    """conversation_id of the chosen (root) path this branch diverged from."""

    branch_turn_index: int
    """Which assistant turn was degraded (0-indexed over assistant steps)."""

    degradation_type: str
    """e.g. "wrong_tool", "missing_arg", "safety_violation"."""

    chosen_response: Step
    """The original high-quality assistant step at branch_turn_index.
    Stored here so the serializer does not need to look up the parent."""


# ===========================================================================
#                       CONVERSATION CONTEXT
# ===========================================================================
@dataclass(kw_only=True)
class ConversationDataPoint(DataPoint):
    """Central state object passed through every stage in the conversation pipeline.

    Every stage receives the full context and reads only what it needs. Stages
    mutate the context in place (appending Steps) and return it. A stage that
    fails to process a data point drops it by returning an empty list.

    All conversation state lives in `steps`. Scenario description, persona,
    flow control signals, tool calls, and tool results are encoded as typed
    Step subclasses and accessed by querying steps by role:

        # Current scenario
        scenario_steps = [s for s in context.steps if s.role == "scenario"]
        scenario = scenario_steps[-1].content if scenario_steps else None

        # Scenario family id — first-class field on ScenarioStep
        family_id = scenario_steps[-1].scenario_family_id if scenario_steps else None

        # Active user persona (empty string if no persona stage in the recipe)
        persona_steps = [s for s in context.steps if s.role == "persona" and s.target == "user"]
        persona_text = persona_steps[-1].content if persona_steps else ""

        # Flow controller termination — first-class field on FlowControllerStep
        fc_steps = [s for s in context.steps if s.role == "flow_controller"]
        terminate = bool(fc_steps and fc_steps[-1].terminate)

        # Flow controller hint for user stage
        hint = fc_steps[-1].hint if fc_steps else None

    Inherits `task_name: str` and `is_seed: bool` from DataPoint.
    """

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Step] = field(default_factory=list)

    # Branch provenance: None on all SFT conversations.
    branch_point: BranchPoint | None = None

    # Conversation-level outcome reward: populated by an outcome validator at
    # conversation completion. Used primarily in ROLLOUT mode; None in SFT mode.
    outcome_reward: float | None = None
    outcome_reward_breakdown: Dict[str, float] = field(default_factory=dict)

    def assistant_step_scores(self) -> List[Dict[str, float]]:
        """Returns scores for each assistant step in order.

        Convenience method for serializers constructing per-turn preference
        signal. Index i corresponds to the i-th assistant step in `steps`.
        """
        return [s.scores for s in self.steps if s.role == "assistant"]
