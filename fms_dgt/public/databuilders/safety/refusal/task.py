# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict
import os

# Local
from fms_dgt.base.task import GenerationTask
from fms_dgt.public.databuilders.safety.refusal.data_objects import SafetyRefusalData
from fms_dgt.public.databuilders.safety.refusal.policy_loader import (
    load_policy_metadata,
    load_risk_name,
)


class SafetyRefusalTask(GenerationTask):
    """Task type for the safety/refusal databuilder.

    ``task.yaml`` must specify a ``risk_policy`` block:

    .. code-block:: yaml

        risk_policy:
          path: data/public/safety/resources/policies/discrimination.yaml
          risk: discrimination_at_work  # optional; defaults to first risk in file
    """

    INPUT_DATA_TYPE = SafetyRefusalData
    OUTPUT_DATA_TYPE = SafetyRefusalData

    def __init__(
        self,
        *args: Any,
        risk_policy: Dict[str, Any],
        num_icl_examples: int = 3,
        num_samples_per_batch: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._policy_dict = self._load_policy(risk_policy)

        if num_icl_examples < 1:
            raise ValueError("num_icl_examples must be at least 1.")
        if num_samples_per_batch < 1:
            raise ValueError("num_samples_per_batch must be at least 1.")

        self._num_icl_examples = num_icl_examples
        self._num_samples_per_batch = num_samples_per_batch

    @property
    def policy_dict(self) -> Dict[str, Any]:
        return self._policy_dict

    @property
    def num_icl_examples(self) -> int:
        return self._num_icl_examples

    @property
    def num_samples_per_batch(self) -> int:
        return self._num_samples_per_batch

    def instantiate_input_example(self, **kwargs: Any) -> SafetyRefusalData:
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            task_description=self.task_description,
            instruction=kwargs.get("instruction", kwargs.get("question", "")),
            response=kwargs.get("response", kwargs.get("answer", None)),
            source=kwargs.get("source", "seed"),
            metadata=kwargs.get("metadata", None),
        )

    @staticmethod
    def _load_policy(risk_policy: Dict[str, Any]) -> Dict[str, Any]:
        path = os.path.expandvars(risk_policy.get("path", ""))
        if not os.path.exists(path):
            raise ValueError(
                f"SafetyRefusalTask: cannot locate risk policy file {path!r}. "
                "Set 'risk_policy.path' to an absolute path or a path relative "
                "to the working directory."
            )
        risk_name = risk_policy.get("risk") or load_risk_name(path)
        return load_policy_metadata(path, risk_name)
