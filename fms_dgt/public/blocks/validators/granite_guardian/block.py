# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re

# Local
from fms_dgt.base.block import ValidatorBlock, get_row_name
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.constants import TYPE_KEY
from fms_dgt.public.blocks.validators.granite_guardian.constants import (
    GUARDIAN_MODEL_FAMILY,
    REQUIRED_LOGPROBS,
    REQUIRED_MAX_TOKENS,
    REQUIRED_MAX_TOKENS_WITH_THINK,
    REQUIRED_TEMPERATURE,
    V32_LABEL_RE,
    V33_SCORE_RE,
)

# ===========================================================================
#                       DATA OBJECT
# ===========================================================================


@dataclass(kw_only=True)
class GraniteGuardianData(ValidatorBlockData):
    """Input/output data type for GraniteGuardianValidator.

    Attributes:
        text (str): The text to assess (user message, assistant response, or
            full conversation). Passed as the ``assistant`` role so Guardian
            evaluates the content rather than treating it as instructions.
        risk_policy (Dict[str, Any]): Risk policy dict. For 3.3 it must contain
            at least one of ``criteria_id`` (named built-in risk, e.g.
            ``"harm"``) or ``custom_criteria`` (free-text policy description).
            For 3.2 it must contain ``risk_name``.  Additional metadata fields
            (e.g. ``risk_description``, ``version``) are allowed and are
            forwarded into the output metadata unchanged.
        reasoning (Optional[str]): The think-trace extracted from the model
            response when ``think=True`` is set on the block.  Populated by
            the block; callers should leave this ``None``.
    """

    text: str
    risk_policy: Dict[str, Any]
    reasoning: Optional[str] = None


# ===========================================================================
#                       BLOCK
# ===========================================================================


@register_block("validators/granite_guardian")
class GraniteGuardianValidator(ValidatorBlock):
    """Wraps IBM Granite Guardian for safety / risk assessment.

    The block supports Granite Guardian 3.3 (default) and 3.2, handling the
    API differences between versions transparently.  It enforces greedy
    decoding regardless of the ``lm_config`` supplied, overriding any
    conflicting values with a warning.

    Rating is always derived from the model's text output (``<score>yes</score>``
    / ``<score>no</score>`` for 3.3; bare ``Yes``/``No`` for 3.2).

    Confidence is derived from logprobs only for model_version ``"3.2"``, where
    the first output token is ``Yes`` or ``No`` and the logprob at that position
    is a well-defined probability over the rating.  For 3.3 the first token is
    ``<score>`` (not the rating token), so logprob-based confidence is not
    meaningful and is omitted from the metadata.

    Args:
        lm_config (Dict): LMProvider configuration.  Must contain a ``type``
            key (e.g. ``"vllm"`` or ``"openai"``).  ``model_id_or_path``
            should reference a ``granite-guardian`` model; a warning is
            emitted if it does not.
        model_version (str): ``"3.3"`` (default) or ``"3.2"``.  Controls
            which chat-template fields and output-parsing logic are used.
        think (bool): When ``True`` (only valid for 3.3), requests
            think-mode generation.  The ``<think>...</think>`` trace is
            extracted and stored in ``GraniteGuardianData.reasoning`` and
            surfaced in the ``_validate`` metadata dict.  Defaults to
            ``False``.
    """

    DATA_TYPE = GraniteGuardianData

    def __init__(
        self,
        lm_config: Dict,
        model_version: str = "3.3",
        think: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model_version not in ("3.3", "3.2"):
            raise ValueError(
                f"Unsupported model_version '{model_version}'. Must be '3.3' or '3.2'."
            )
        self._model_version = model_version
        self._think = think

        if think and model_version != "3.3":
            raise ValueError("think=True is only supported for model_version='3.3'.")

        if TYPE_KEY not in lm_config:
            raise ValueError(
                f"Must specify '{TYPE_KEY}' in 'lm_config' for {self.__class__.__name__}."
            )

        model_id = lm_config.get("model_id_or_path", "")
        if model_id and GUARDIAN_MODEL_FAMILY not in model_id:
            self.logger.warning(
                "GraniteGuardianValidator: 'model_id_or_path' ('%s') does not appear to be a "
                "granite-guardian model. Unexpected results may occur.",
                model_id,
            )

        lm_config = dict(lm_config)
        self._enforce_inference_settings(lm_config)

        self._lm = get_block(lm_config[TYPE_KEY], **lm_config)
        self._blocks.append(self._lm)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _enforce_inference_settings(self, lm_config: Dict) -> None:
        """Overrides temperature, max_tokens, and logprobs in-place, warning on conflicts."""
        temp = lm_config.get("temperature")
        if temp is not None and temp != REQUIRED_TEMPERATURE:
            self.logger.warning(
                "GraniteGuardianValidator: overriding temperature %s -> %s (greedy decoding required).",
                temp,
                REQUIRED_TEMPERATURE,
            )
        lm_config["temperature"] = REQUIRED_TEMPERATURE

        max_tok = lm_config.get("max_tokens")
        if self._think:
            # In think mode the model generates a reasoning trace before the
            # score tag; 20 tokens would truncate it. Use the caller-supplied
            # value if present, otherwise fall back to REQUIRED_MAX_TOKENS_THINK.
            if max_tok is None:
                lm_config["max_tokens"] = REQUIRED_MAX_TOKENS_WITH_THINK
            # else: keep whatever the user specified
        else:
            if max_tok is not None and max_tok != REQUIRED_MAX_TOKENS:
                self.logger.warning(
                    "GraniteGuardianValidator: overriding max_tokens %s -> %s.",
                    max_tok,
                    REQUIRED_MAX_TOKENS,
                )
            lm_config["max_tokens"] = REQUIRED_MAX_TOKENS

        # Logprobs are only meaningful for 3.2 where the first output token is
        # Yes/No. For 3.3 the first token is <score>, so logprob-based confidence
        # is not valid and we do not request logprobs.
        if self._model_version == "3.2":
            logprobs = lm_config.get("logprobs")
            if logprobs is not None and logprobs != REQUIRED_LOGPROBS:
                self.logger.warning(
                    "GraniteGuardianValidator: overriding logprobs %s -> %s.",
                    logprobs,
                    REQUIRED_LOGPROBS,
                )
            lm_config["logprobs"] = REQUIRED_LOGPROBS

    def _build_guardian_config(self, risk_policy: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the guardian_config dict for the chat template kwargs."""
        if self._model_version == "3.3":
            if "criteria_id" in risk_policy:
                return {"risk": risk_policy["criteria_id"]}
            elif "custom_criteria" in risk_policy:
                return {"custom_criteria": risk_policy["custom_criteria"]}
            else:
                raise ValueError(
                    "For model_version='3.3', risk_policy must contain 'criteria_id' or 'custom_criteria'."
                )
        else:  # 3.2
            if "risk_name" not in risk_policy:
                raise ValueError("For model_version='3.2', risk_policy must contain 'risk_name'.")
            return {"risk_definition": risk_policy["risk_name"]}

    def _confidence_from_logprobs(self, token_logprobs: List[Dict[str, Any]]) -> Optional[float]:
        """Derives calibrated confidence from top-k token log-probs.

        Looks at the first output token's top-k log-probabilities and extracts
        the probability mass on the yes/no (or Yes/No) tokens, normalising to
        produce a confidence score in [0, 1] where 1 == maximally unsafe.

        Returns:
            Optional[float]: Normalised probability of "yes" (unsafe), or
            ``None`` if logprobs are absent or unusable.
        """
        if not (isinstance(token_logprobs, list) and token_logprobs):
            return None

        first_token_lp = token_logprobs[0]
        logp_safe = first_token_lp.get("no", first_token_lp.get("No", -math.inf))
        logp_unsafe = first_token_lp.get("yes", first_token_lp.get("Yes", -math.inf))

        p_safe = math.exp(logp_safe)
        p_unsafe = math.exp(logp_unsafe)
        total = p_safe + p_unsafe
        if total <= 0:
            return None

        return round(p_unsafe / total, 4)

    def _parse_response_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts (rating, reasoning) from a raw model response string.

        For 3.3: parses ``<score>yes</score>`` / ``<score>no</score>`` and
        optionally a ``<think>...</think>`` block.
        For 3.2: parses a bare ``Yes`` / ``No`` label.

        Returns:
            Tuple[Optional[str], Optional[str]]: (rating, reasoning) where
            reasoning is ``None`` unless think-mode produced a trace.
        """
        if not text:
            return None, None

        reasoning = None
        if self._model_version == "3.3":
            # Extract think trace if present
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
            if think_match:
                reasoning = think_match.group(1).strip()

            score_match = V33_SCORE_RE.search(text)
            rating = score_match.group(1).lower() if score_match else None
        else:
            label_match = V32_LABEL_RE.search(text)
            rating = label_match.group(1).lower() if label_match else None

        return rating, reasoning

    # -----------------------------------------------------------------------
    # ValidatorBlock interface
    # -----------------------------------------------------------------------

    def _validate_rating(
        self,
        rating: Optional[str],
        confidence: Optional[float],
        reasoning: Optional[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Converts a Guardian rating into (is_valid, metadata).

        ``is_valid=True`` means the content is safe (rating == "no").
        """
        metadata: Dict[str, Any] = {"rating": rating}
        if confidence is not None:
            metadata["confidence"] = confidence
        if reasoning is not None:
            metadata["reasoning"] = reasoning

        if rating is None:
            # Could not parse output; treat as failed validation
            metadata["reason"] = "Guardian output could not be parsed."
            return False, metadata

        is_safe = rating == "no"
        return is_safe, metadata

    def _validate_batch(
        self, inputs: List[GraniteGuardianData]
    ) -> List[Tuple[bool, Dict[str, Any]]]:
        """Run Guardian on a batch of instances in a single LM call."""
        # All items in a batch share the same risk_policy (task-homogeneous input
        # from the safety databuilder), but we build per-item chat_template_kwargs
        # in case risk_policy differs across items.
        lm_inputs = []
        for x in inputs:
            guardian_config = self._build_guardian_config(x.risk_policy)
            chat_template_kwargs: Dict[str, Any] = {"guardian_config": guardian_config}
            if self._model_version == "3.3":
                chat_template_kwargs["think"] = self._think
            lm_inputs.append(
                {
                    "input": [{"role": "user", "content": x.text}],
                    "task_name": get_row_name(x),
                    "gen_kwargs": {
                        "extra_body": {"chat_template_kwargs": chat_template_kwargs},
                    },
                }
            )

        lm_outputs = self._lm(lm_inputs, method=self._lm.CHAT_COMPLETION)

        results = []
        for x, lm_output in zip(inputs, lm_outputs):
            result = lm_output.get("result") or {}
            response_text: str = result.get("content", "") if isinstance(result, dict) else result
            rating, reasoning = self._parse_response_text(response_text)

            # Confidence is only meaningful for 3.2 where the first output token
            # is Yes/No. For 3.3 the first token is <score>, so we skip it.
            confidence: Optional[float] = None
            if self._model_version == "3.2":
                token_logprobs = (lm_output.get("addtl") or {}).get("token_logprobs", [[]])[0]
                confidence = self._confidence_from_logprobs(token_logprobs)

            x.reasoning = reasoning
            results.append(
                self._validate_rating(rating=rating, confidence=confidence, reasoning=reasoning)
            )

        return results

    def _validate(self, x: GraniteGuardianData) -> Tuple[bool, Dict[str, Any]]:
        """Not used — GraniteGuardianValidator overrides _validate_batch instead."""
        raise NotImplementedError(
            "GraniteGuardianValidator uses _validate_batch for batch LM calls. "
            "Do not call _validate directly."
        )
