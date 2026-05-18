# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any, Dict, List
import random
import re

# Local
from fms_dgt.base.databuilder import GenerationDataBuilder
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.blocks.validators.rouge import RougeDedupValidator
from fms_dgt.public.blocks.validators.granite_guardian.block import (
    GraniteGuardianValidator,
)
from fms_dgt.public.databuilders.safety.refusal.data_objects import SafetyRefusalData
from fms_dgt.public.databuilders.safety.refusal.task import SafetyRefusalTask
from fms_dgt.utils import group_by

# ===========================================================================
#                       HELPERS
# ===========================================================================


def _parse_generated_instructions(text: str) -> List[str]:
    """Split LM output into individual instructions using the ``### Question N:`` delimiter."""
    parts = re.split(r"### Question \d+:", text)
    return [p.strip() for p in parts if p.strip()]


# ===========================================================================
#                       DATABUILDER
# ===========================================================================


@register_data_builder("public/safety/refusal")
class SafetyRefusalDataBuilder(GenerationDataBuilder):
    """Generates (harmful prompt, calibrated refusal) pairs for safety alignment training.

    Pipeline:
        1. Generate new harmful instructions from seed examples via ICL.
        2. Deduplicate with ROUGE to remove near-duplicates.
        3. Filter with Granite Guardian: keep only prompts Guardian rates as
           harmful (asymmetric filter -- we want the harmful ones here).
        4. Generate a calibrated refusal response for each validated instruction.
        5. Filter with Granite Guardian: keep only responses Guardian rates as
           safe (rating == ``"no"``).

    The asymmetric filtering pattern is the defining characteristic of this
    recipe: harmful instructions are *kept* (not the safe ones), and safe
    responses are *kept* (not the harmful ones).

    Risk policy configuration lives on the task (``task.yaml``). This
    databuilder is policy-agnostic; one ``refusal.yaml`` serves all risk
    categories.
    """

    TASK_TYPE = SafetyRefusalTask

    instruction_generator: LMProvider
    dedup: RougeDedupValidator
    response_generator: LMProvider
    granite_guardian: GraniteGuardianValidator

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        templates_dir = Path(__file__).parent / "prompt_templates"
        self._gen_instruction_prompt = JinjaPromptTemplate(
            template_path=templates_dir / "generate_instructions.txt"
        )
        self._gen_response_prompt = JinjaPromptTemplate(
            template_path=templates_dir / "generate_response.txt"
        )

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[SafetyRefusalData],
    ) -> List[SafetyRefusalData]:
        if not instruction_data:
            return []

        results: List[SafetyRefusalData] = []
        for task_name, group in group_by(instruction_data, key=lambda dp: dp.task_name).items():
            task: SafetyRefusalTask = self.get_task(task_name)
            policy = task.policy_dict

            # Step 1: Generate new harmful instructions from seed via ICL
            candidates = self._generate_instructions(
                group, policy, task.num_icl_examples, task.num_samples_per_batch
            )

            # Step 2: Deduplicate against all known instructions in this group
            all_known = [dp.instruction for dp in group]
            deduped = self._deduplicate(candidates, all_known)

            # Step 3: Guardian filter -- keep only genuinely harmful instructions
            validated = self._guardian_filter_harmful(deduped, policy)
            if not validated:
                continue

            # Step 4: Generate refusal responses
            with_responses = self._generate_responses(validated, policy)

            # Step 5: Guardian filter -- keep only safe responses
            results.extend(self._guardian_filter_safe(with_responses, policy))

        return results

    # -----------------------------------------------------------------------
    # Pipeline steps
    # -----------------------------------------------------------------------

    def _generate_instructions(
        self,
        data_points: List[SafetyRefusalData],
        policy: Dict[str, Any],
        num_icl_examples: int,
        num_samples_per_batch: int,
    ) -> List[SafetyRefusalData]:
        # data_points is already task-homogeneous (grouped in __call__)
        task_name = data_points[0].task_name
        icl_examples = random.choices(data_points, k=num_icl_examples)

        prompt = self._gen_instruction_prompt.encode(
            render_dict={
                "num_samples": num_samples_per_batch,
                "task_description": icl_examples[0].task_description,
                "icl_examples": icl_examples,
                "risk_policy": policy,
            }
        )
        lm_inputs = [
            {
                "input": [{"role": "user", "content": prompt}],
                "references": icl_examples,
                "task_name": task_name,
            }
        ]

        lm_outputs = self.instruction_generator(lm_inputs, method=LMProvider.CHAT_COMPLETION)

        outputs: List[SafetyRefusalData] = []
        for lm_output in lm_outputs:
            result = lm_output.get("result", "")
            if isinstance(result, dict):
                result = result.get("content") or ""
            for instruction in _parse_generated_instructions(result):
                outputs.append(
                    SafetyRefusalData(
                        **{
                            **lm_output["references"][0].to_dict(),
                            "is_seed": False,
                            "instruction": instruction,
                            "response": None,
                            "source": "machine",
                        }
                    )
                )

        return outputs

    def _deduplicate(
        self,
        candidates: List[SafetyRefusalData],
        context: List[str],
    ) -> List[SafetyRefusalData]:
        dedup_inputs = [{"input": dp.instruction} for dp in candidates]
        dedup_outputs = self.dedup(dedup_inputs, context=context)
        valid_texts = {out["input"] for out in dedup_outputs if out.get("is_valid", True)}
        return [dp for dp in candidates if dp.instruction in valid_texts]

    def _guardian_filter_harmful(
        self,
        data_points: List[SafetyRefusalData],
        policy: Dict[str, Any],
    ) -> List[SafetyRefusalData]:
        """Keep data points whose instruction Guardian rates as harmful (``"yes"``).

        Guardian ``is_valid=True`` means safe. We invert: keep ``is_valid=False``.
        """
        risk_policy = self._build_guardian_risk_policy(policy)
        guardian_outputs = self.granite_guardian(
            [
                {
                    "text": dp.instruction,
                    "risk_policy": risk_policy,
                    "reference": dp,
                    "task_name": dp.task_name,
                }
                for dp in data_points
            ]
        )
        kept = []
        for gout in guardian_outputs:
            if not gout.get("is_valid", True):
                dp = gout["reference"]
                dp.metadata = self._build_metadata(policy=policy, mode="instruction")
                kept.append(dp)
        return kept

    def _generate_responses(
        self,
        data_points: List[SafetyRefusalData],
        policy: Dict[str, Any],
    ) -> List[SafetyRefusalData]:
        lm_inputs: List[Dict] = [
            {
                "input": [
                    {
                        "role": "user",
                        "content": self._gen_response_prompt.encode(
                            render_dict={
                                "question": dp.instruction,
                                "risk_policy": policy,
                            }
                        ),
                    }
                ],
                "reference": dp,
                "task_name": dp.task_name,
            }
            for dp in data_points
        ]

        lm_outputs = self.response_generator(lm_inputs, method=LMProvider.CHAT_COMPLETION)

        outputs: List[SafetyRefusalData] = []
        for lm_output in lm_outputs:
            dp = lm_output["reference"]
            result = lm_output.get("result", "")
            if isinstance(result, dict):
                result = result.get("content") or ""
            dp.response = result.strip()
            dp.is_seed = False
            dp.source = "machine"
            outputs.append(dp)

        return outputs

    def _guardian_filter_safe(
        self,
        data_points: List[SafetyRefusalData],
        policy: Dict[str, Any],
    ) -> List[SafetyRefusalData]:
        """Keep data points whose response Guardian rates as safe (``"no"``).

        Guardian ``is_valid=True`` means safe -- exactly what we want for responses.
        """
        risk_policy = self._build_guardian_risk_policy(policy)
        guardian_outputs = self.granite_guardian(
            [
                {
                    "text": dp.response,
                    "risk_policy": risk_policy,
                    "reference": dp,
                    "task_name": dp.task_name,
                }
                for dp in data_points
            ]
        )
        kept = []
        for gout in guardian_outputs:
            if gout.get("is_valid", False):
                dp = gout["reference"]
                dp.metadata = self._build_metadata(policy=policy, mode="response")
                kept.append(dp)
        return kept

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_guardian_risk_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
        """Build the ``risk_policy`` dict for ``GraniteGuardianValidator`` (3.3 API)."""
        constraints = policy.get("reply_cannot_contain", [])
        description = policy.get("risk_description", policy["risk"])
        criteria_text = description
        if constraints:
            criteria_text += "\n\nA response is unsafe if it contains any of:\n" + "\n".join(
                f"- {c}" for c in constraints
            )
        return {"custom_criteria": criteria_text}

    @staticmethod
    def _build_metadata(policy: Dict[str, Any], mode: str) -> Dict[str, Any]:
        return {
            "risk": policy["risk"],
            "risk_description": policy["risk_description"],
            "risk_group": policy.get("risk_group", ""),
            "risk_group_id": policy.get("risk_group_id", ""),
            "risk_id": policy.get("risk_id", ""),
            "reason_denial": policy.get("reason_denial", ""),
            "short_reply_type": policy.get("short_reply_type", ""),
            "exception": policy.get("exception", ""),
            "policy_version": policy.get("policy_version", "v0.1"),
            "mode": mode,
        }
