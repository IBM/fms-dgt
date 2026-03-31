# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Callable, Dict, Type

# Local
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    FlowControllerStep,
    PersonaStep,
    ScenarioStep,
    Step,
    ToolCallStep,
    ToolResultStep,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.stages.base import Stage

_STAGE_REGISTRY: Dict[str, Type[Stage]] = {}
_CONTEXT_GENERATOR_REGISTRY: Dict[str, type] = {}
_STEP_REGISTRY: Dict[str, type] = {}


def register_stage(name: str) -> Callable[[Type[Stage]], Type[Stage]]:
    """Register a Stage subclass under the given name.

    Usage:
        @register_stage("lm/user/guided")
        class GuidedUserStage(Stage):
            ...
    """

    def decorator(cls: Type[Stage]) -> Type[Stage]:
        if name in _STAGE_REGISTRY:
            raise ValueError(
                f"Stage '{name}' is already registered to {_STAGE_REGISTRY[name].__qualname__}. "
                f"Cannot register {cls.__qualname__} under the same name."
            )
        _STAGE_REGISTRY[name] = cls
        return cls

    return decorator


def get_stage(name: str) -> Type[Stage]:
    """Retrieve a registered Stage class by name.

    Raises KeyError if the name has not been registered.
    """
    if name not in _STAGE_REGISTRY:
        raise KeyError(
            f"Stage '{name}' is not registered. " f"Available stages: {sorted(_STAGE_REGISTRY)}"
        )
    return _STAGE_REGISTRY[name]


def register_context_generator(name: str) -> Callable[[type], type]:
    """Register a ContextGenerator subclass under the given name.

    Usage:
        @register_context_generator("tc/neighbor_tools")
        class NeighborToolContextGenerator(ContextGenerator):
            ...
    """

    def decorator(cls: type) -> type:
        if name in _CONTEXT_GENERATOR_REGISTRY:
            raise ValueError(
                f"ContextGenerator '{name}' is already registered to "
                f"{_CONTEXT_GENERATOR_REGISTRY[name].__qualname__}."
            )
        _CONTEXT_GENERATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_context_generator(name: str) -> type:
    """Retrieve a registered ContextGenerator class by name."""
    if name not in _CONTEXT_GENERATOR_REGISTRY:
        raise KeyError(
            f"ContextGenerator '{name}' is not registered. "
            f"Available: {sorted(_CONTEXT_GENERATOR_REGISTRY)}"
        )
    return _CONTEXT_GENERATOR_REGISTRY[name]


def register_step(role: str) -> Callable[[type], type]:
    """Register a Step subclass under the given role name.

    The role string is the discriminator used by Step.from_dict() to
    reconstruct the correct subclass during deserialization. It must match
    the `role` value the subclass writes into ConversationDataPoint.steps.

    Built-in step types are pre-registered by the framework. Recipe authors
    who define custom Step subclasses for custom roles call this decorator
    in their recipe package — no core changes required.

    Raises ValueError if the role is already registered.

    Usage:
        @register_step("persona")
        @dataclass
        class PersonaStep(Step):
            target: Literal["user", "assistant"] = "user"
    """

    def decorator(cls: type) -> type:
        if role in _STEP_REGISTRY:
            raise ValueError(
                f"Step role '{role}' is already registered to "
                f"{_STEP_REGISTRY[role].__qualname__}. "
                f"Cannot register {cls.__qualname__} under the same role."
            )
        _STEP_REGISTRY[role] = cls
        return cls

    return decorator


def get_step(role: str) -> type:
    """Retrieve a registered Step subclass by role name.

    Falls back to the base Step class for unrecognized roles rather than
    raising, so seed data with unknown custom roles deserializes safely
    as flat Step objects.
    """
    return _STEP_REGISTRY.get(role, Step)


def _register_builtin_steps() -> None:
    """Register the framework's built-in Step subclasses.

    Called once at the bottom of this module. The import is deferred to here
    (rather than at module top) to break the circular dependency:
        data_objects -> registry -> stages.base -> data_objects
    """
    for role, cls in [
        ("user", UserStep),
        ("assistant", AssistantStep),
        ("tool_call", ToolCallStep),
        ("tool_result", ToolResultStep),
        ("flow_controller", FlowControllerStep),
        ("scenario", ScenarioStep),
        ("persona", PersonaStep),
    ]:
        if role not in _STEP_REGISTRY:
            _STEP_REGISTRY[role] = cls


_register_builtin_steps()
