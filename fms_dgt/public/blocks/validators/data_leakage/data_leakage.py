# Standard
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Local
from fms_dgt.base.block import ValidatorBlock

# from fms_dgt.constants import DATASET_TYPE
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import register_block
from fms_dgt.public.blocks.validators.data_leakage.rougelmod_score_handler import (
    RougeLmodScoreHandler,
    RougeLmodScoreHandlerConfig,
)


@dataclass(kw_only=True)
class DLValidatorData(ValidatorBlockData):
    input: str
    local_context: Optional[List[str]] = None

    """
    Represents a single validation example for the DataLeakageValidator.
    Each instance contains one generated string (`input`) and an optional list of seed strings (`local_context`) used during its creation.
    If a global context is used instead, it should be passed directly to the `execute()` method.

    Attributes:
        input (str): The generated string to check for data leakage.
        local_context (List[str]): Optional. A list of seed strings that may have influenced the generated input.
                                    If a global context was used for all inputs, pass it via the `context` argument to `execute()`.
                                    Either local_context here or (global) context argument in execute() is required.
    """

    def __post_init__(self) -> None:
        if self.local_context is not None:
            if isinstance(self.local_context, str):
                self.local_context = [self.local_context]


@register_block("validators/data_leak")
class DataLeakageValidator(ValidatorBlock):

    DATA_TYPE: DLValidatorData = DLValidatorData

    def __init__(self, threshold: float = 1.1, **kwargs: Any) -> None:
        """
        Validator block to detect potential data leakage.

        Args:
            threshold (float): Similarity score threshold above which a sample is considered an invalid data leak
        """
        super().__init__(**kwargs)
        self._threshold = threshold
        score_handler_config = RougeLmodScoreHandlerConfig()
        self.score_handler = RougeLmodScoreHandler(config=score_handler_config)

    def execute(
        self,
        inputs: Iterable[DLValidatorData],
        *,
        filter: bool = True,
        context: Optional[List[str]] = None,
    ):
        """
        Validates generated samples by detecting data leakage from seed examples.
        This validator removes any elements from `inputs` that appear to contain fragments copied from the seed examples.
        It uses a custom data-leakage metric developed by Shlomit Shachor and Natalia Razinkov.

        Args:
            inputs (Iterable[DLValidatorData]):   The generated samples to validate. Each may include a `local_context`.
            context ([List[Str]]): Optional. A list of global seed examples used as references text.
                                    Each input is compared against these using the leakage metric, and overly similar ones are filtered out.
                                    Either global context here or local_context for each input is required.

        Returns:
            DATASET_TYPE: Input dataset with validity w.r.t data leakage added (invalid inputs possibly filtered according to _filter_invalids field)
        """
        filter = filter and self._filter_invalids
        outputs, to_save = [], []
        for x in inputs:
            _context = context or x.local_context
            if not _context:
                raise ValueError(
                    "Missing context: either 'x.local_context' or a global 'context' must be provided."
                )
            x.is_valid, x.metadata = self._validate(x.input, _context)
            if x.is_valid or not filter:
                outputs.append(x)
            if not x.is_valid:
                to_save.append(x)
        self.save_data(to_save)
        return outputs

    def _scorer(self, candidate_text, reference_texts):
        result = self.score_handler.compute_text_to_texts_match(
            ref_texts=reference_texts, pred_text=candidate_text
        )

        score = result.aggregated_score_for_all_max_match_text_sentences
        return score

    def _validate(self, candidate_text, reference_texts) -> Tuple[bool, Dict | None]:
        if self._threshold is None or self._threshold > 1.0:
            return True, None

        score = self._scorer(candidate_text, reference_texts)

        is_valid = score < self._threshold
        return is_valid, (
            {"score": score}
            if is_valid
            else {
                "score": score,
                "reason": f"Score ({score}) equals or exceeds threshold ({self._threshold})",
            }
        )
