# Standard
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

# Local
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import register_block

try:
    # Third Party
    from rouge_score import rouge_scorer
except ModuleNotFoundError:
    pass


@dataclass(kw_only=True)
class RougeDedupData(ValidatorBlockData):
    input: str


@register_block("rouge_scorer")
class RougeDedupValidator(ValidatorBlock):
    """Base Class for all Validators"""

    DATA_TYPE: RougeDedupData = RougeDedupData

    def __init__(
        self,
        threshold: float = 1.1,
        fast_rouge: bool = True,
        score_func: Literal["f1", "precision", "recall"] = "f1",
        rouge_types: List[Literal["lcs", "n_1", "n_2", "n_3"]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if threshold is None:
            # if threshold is set to None, we'll put it as an unreachably high value
            threshold = 1.1

        self._fast_rouge = fast_rouge
        self._threshold = threshold
        if score_func not in ["f1", "precision", "recall"]:
            raise ValueError(
                f"'score_func' passed to RougeDedupValidator must be one of [f1, precision, recall], instead got {score_func}"
            )
        self._score_func = score_func

        if rouge_types is None:
            rouge_types = ["lcs"]
        if not isinstance(rouge_types, list):
            raise ValueError(
                f"'rouge_types' must be provided as list, instead got {rouge_types}"
            )
        rouge_types = [x.strip() for x in rouge_types]
        for rouge_type in rouge_types:
            if rouge_type not in ["lcs", "n_1", "n_2", "n_3"]:
                raise ValueError(
                    f"'rouge_type' passed to RougeDedupValidator must only contain [lcs, n_1, n_2, n_3], instead got {rouge_type}"
                )
        self._rouge_types = rouge_types

        self._cache = dict()
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def tokenize(self, inp: List | str):
        if isinstance(inp, list):
            return [self.tokenize(el) for el in inp]
        else:
            if inp not in self._cache:
                self._cache[inp] = self._scorer._tokenizer.tokenize(inp)
            return self._cache[inp]

    def execute(
        self,
        inputs: Iterable[RougeDedupData],
        *,
        context: Optional[List[str]] = None,
        score_func: str = None,
    ):
        """Deduplicator that removes elements of `inputs` that are too rouge-similar. By default it will pick the one that is maximally dissimilar from `context` to keep"""
        score_func = score_func or self._score_func
        if score_func not in ["f1", "precision", "recall"]:
            raise ValueError(
                f"'score_func' passed to RougeDedupValidator must be one of [f1, precision, recall], instead got {score_func}"
            )

        # tokenize context
        tokenized_context = self.tokenize(context) if context else []

        tokenized = []
        for inp in inputs:
            # TODO: Safety check this
            tokenized.append((self.tokenize(inp.input), inp))

        # first score inputs by rouge similarity to context
        ranked_inputs = []
        for new_tokens, inp in tokenized:
            worst_rouge_score = (
                max(
                    map(
                        partial(
                            _score_seqs,
                            new_tokens,
                            max_thresh=self._threshold if self._fast_rouge else None,
                            score_func=self._score_func,
                            rouge_types=self._rouge_types,
                        ),
                        tokenized_context,
                    )
                )
                if tokenized_context and self._threshold <= 1.0
                else -1
            )

            is_valid_wrt_context = worst_rouge_score < self._threshold

            if is_valid_wrt_context or not self._filter_invalids:
                ranked_inputs.append(
                    (
                        worst_rouge_score,
                        is_valid_wrt_context,
                        new_tokens,
                        inp,
                    )
                )

        ranked_inputs.sort(key=lambda x: x[0])

        # now add
        added_tokens, outputs, to_save = [], [], []
        for _, is_valid_wrt_context, new_tokens, inp in ranked_inputs:
            # only check against elements we've already added
            validation_output = self._validate(new_tokens, added_tokens)
            if isinstance(validation_output, bool):
                valid = validation_output
            elif isinstance(validation_output, tuple) and len(validation_output) == 2:
                valid, inp.metadata = validation_output
            else:
                raise RuntimeError(
                    '"_validate" function must return a bool or tuple of [bool, Dict]',
                )

            inp.is_valid = is_valid_wrt_context and valid

            if inp.is_valid or not self._filter_invalids:
                added_tokens.append(new_tokens)
                outputs.append(inp)

            if not inp.is_valid:
                to_save.append(inp)

        self.save_data(to_save)

        return outputs

    def _validate(
        self, new_tokens: List[int], check_tokens: List[List[int]]
    ) -> bool | Tuple[bool, Dict | None]:
        """Runs through all the validators if data list is None. Otherwise just runs through the validators specified for data in the List"""

        if (
            self._threshold > 1 or len(check_tokens) == 0
        ):  # if threshold greater than 1 or there is nothing to check against, no need to bother computing this
            return True

        rouge_scores = map(
            partial(
                _score_seqs,
                new_tokens,
                max_thresh=self._threshold if self._fast_rouge else None,
                score_func=self._score_func,
                rouge_types=self._rouge_types,
            ),
            check_tokens,
        )

        # NOTE: we do < self._threshold so that if self._threshold == 1.0 we still remove duplicates
        max_rouge_score = max(rouge_scores)
        is_valid = max_rouge_score < self._threshold
        return is_valid, (
            {"score": max_rouge_score}
            if is_valid
            else {
                "score": max_rouge_score,
                "reason": f"Maximum rouge score ({max_rouge_score}) is equal or greater than threshold ({self._threshold})",
            }
        )


def _score_seqs(
    target_tokens: List[int],
    prediction_tokens: List[int],
    max_thresh: Optional[float] = None,
    score_func: Literal["f1", "precision", "recall"] = "f1",
    rouge_types: List[Literal["lcs", "n_1", "n_2", "n_3"]] = None,
):
    if rouge_types is None:
        rouge_types = ["lcs"]

    scores = []
    for rouge_type in rouge_types:
        if rouge_type == "lcs":
            scores.append(
                _score_lcs(target_tokens, prediction_tokens, max_thresh, score_func)
            )
        elif rouge_type in ["n_1", "n_2", "n_3"]:
            n = int(rouge_type.split("_")[-1])
            scores.append(_score_n(target_tokens, prediction_tokens, n, score_func))
        else:
            raise ValueError(f"Unhandled rouge type {rouge_type}")

    return sum(scores) / len(scores)


def _score_n(
    target_tokens: List[int],
    prediction_tokens: List[int],
    n: int,
    score_func: Literal["f1", "precision", "recall"] = "f1",
):
    tgt_n_grams, pred_n_grams = dict(), dict()
    for i in range(len(target_tokens) - n):
        tgt_n_grams[str(target_tokens[i : i + n])] = 1
    for i in range(len(prediction_tokens) - n):
        pred_n_grams[str(prediction_tokens[i : i + n])] = 1

    overlap = sum(
        [pred_n_grams[tgt_k] for tgt_k in tgt_n_grams if tgt_k in pred_n_grams]
    )
    return _score(overlap, len(pred_n_grams), len(tgt_n_grams), score_func)


# The below code is a modification of rouge_scorer._score_lcs, which is
# available under the following License:
#
# Copyright 2022 The rouge_score Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def _score_lcs(
    target_tokens: List[int],
    prediction_tokens: List[int],
    max_thresh: Optional[float] = None,
    score_func: Literal["f1", "precision", "recall"] = "f1",
) -> float:
    """Computes Rouge-L, i.e., rouge score based on longest common subsequence

    Args:
        max_thresh: threshold at which process ends if current fmeasure is greater than this OR cannot reach this
        target_tokens: Tokens from the target text
        prediction_tokens: Tokens from the predicted text

    Returns:
        score for ratio of the lengths of the LCS over the number of unigrams in each sequence
    """

    def _lcs_table(ref, can):
        rows, cols, max_lcs = len(ref), len(can), -1
        tbl = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):

                # get table entry
                if ref[i - 1] == can[j - 1]:
                    tbl[i][j] = tbl[i - 1][j - 1] + 1
                else:
                    tbl[i][j] = max(tbl[i - 1][j], tbl[i][j - 1])

                # decide whether or not to exit early, there's two possible outcomes here
                #   1. Optimistically, if the current lcs produces an fmeasure that's already at least the threshold
                #       you can just return
                #   2. Pessimistically, if the fmeasure of the lcs + the remainder of the sequence (i.e., assume every single
                #       token matches along the remainder of the string) cannot meet the threshold, then you don't need
                #       to check any further
                if max_thresh is not None and tbl[i][j] > max_lcs:
                    if (
                        # (OPTIMISTIC) exit if current outcome meets threshold
                        _score(
                            tbl[i][j],
                            len(prediction_tokens),
                            len(target_tokens),
                            score_func,
                        )
                        >= max_thresh
                        or
                        # (PESSIMISTIC) exit if highest possible outcome cannot meet threshold
                        _score(
                            tbl[i][j] + min(rows - i + 1, cols - j + 1),
                            len(prediction_tokens),
                            len(target_tokens),
                            score_func,
                        )
                        < max_thresh
                    ):
                        return tbl[i][j]

                # update max_lcs
                max_lcs = max(tbl[i][j], max_lcs)

        return max_lcs

    if not target_tokens or not prediction_tokens:
        return 0

    lcs_length = _lcs_table(target_tokens, prediction_tokens)

    return _score(lcs_length, len(prediction_tokens), len(target_tokens), score_func)


def _score(ct, pred_ct, tgt_ct, score_func):
    # first determine current fmeasure
    prec = ct / (pred_ct or 1)
    rec = ct / (tgt_ct or 1)
    if score_func == "f1":
        return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    elif score_func == "precision":
        return prec
    elif score_func == "recall":
        return rec
    else:
        raise ValueError(f"Unhandled score function {score_func}")
