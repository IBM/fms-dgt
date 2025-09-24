# Standard
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

# Local
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider


@dataclass(kw_only=True)
class LMJudgeData(ValidatorBlockData, LMBlockData):
    success_func: Callable[..., bool | Tuple[bool, Dict | None]]


@register_block("lm_judge")
class LMJudgeValidator(ValidatorBlock):
    """LLM-based Validator"""

    DATA_TYPE: LMJudgeData = LMJudgeData

    def __init__(self, lm_config: Dict = None, **kwargs: Any):
        super().__init__(**kwargs)
        assert TYPE_KEY in lm_config, f"Must specify {TYPE_KEY} in 'lm' field of {self.name} block"

        self._llm_generator: LMProvider = get_block(lm_config.get(TYPE_KEY), **lm_config)
        self._blocks.append(self._llm_generator)

    def execute(
        self,
        inputs: Iterable[LMJudgeData],
        **kwargs,
    ):

        # simplify generation here
        llm_outputs: List[LMJudgeData] = self._llm_generator(
            inputs,
            **kwargs,
        )

        judge_outputs, to_save = [], []
        for llm_output in llm_outputs:
            llm_output.is_valid, llm_output.metadata = self._validate(judge_output=llm_output)
            if llm_output.is_valid or not self._filter_invalids:
                judge_outputs.append(llm_output)

            if not llm_output.is_valid:
                to_save.append(llm_output)

        self.save_data(to_save)

        return judge_outputs

    def _validate(self, judge_output: LMJudgeData) -> bool | Tuple[bool, Dict | None]:
        output = judge_output.success_func(judge_output.result)
        if isinstance(output, bool):
            return output, {"judge_output": judge_output.result}
        elif (
            isinstance(output, tuple)
            and len(output) == 2
            and (isinstance(output[-1], dict) or not output[-1])
        ):
            if not output[-1]:
                return output[0], {"judge_output": judge_output.result}
            else:
                return output[0], {"judge_output": judge_output.result, **output[-1]}
        else:
            raise RuntimeError(
                f'"{judge_output.success_func}" success function does not return a tuple of [bool, Dict| None] type.',
            )
