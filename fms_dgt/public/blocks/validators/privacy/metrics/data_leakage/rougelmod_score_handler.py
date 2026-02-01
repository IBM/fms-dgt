# Standard
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import math

# Third Party
import numpy as np

# Local
from fms_dgt.public.blocks.validators.privacy.metrics.data_leakage.metric_msgs import (
    ERR_POSITIVE,
    ERR_SIZE,
)
from fms_dgt.public.blocks.validators.privacy.metrics.data_leakage.score_handler import (
    AbstractScoreHandler,
    AbstractStemmer,
    MatchScoreResult,
    TextMatchScoreResult,
)
from fms_dgt.public.blocks.validators.privacy.metrics.text import (
    IDentiText,
    TextIdentity,
)
from fms_dgt.public.blocks.validators.privacy.metrics.text_util import (
    find_common_non_overlapping_ngrams,
    is_stop_word,
    normalize_text,
    remove_ending_punc,
    sent_tokenize,
)
from fms_dgt.public.blocks.validators.privacy.metrics.util import (
    calculate_aggregated_score,
    is_positive,
    is_range_correct,
    type_check,
    value_check,
)

M_A_S_K_R_E_F = "maskref"
M_A_S_K_P_R_E_D = "maskpred"


@dataclass
class RougeLmodScoreHandlerConfig:
    """
    A RougeLmodScoreHandlerConfig - configuration to guide RougeL modified scorer

    adjacency_penalty_weight : float
        Used with the adjacency score to get weight for adjacency penalty
        (default value is 0.2).
    distance_consistency_penalty_weight : float
        Used with the distance consistency score to get weight for distance
        consistency penalty (default is 0.2).
    word_leak_count_baseline : float
        Amount of words that is considered baseline leakage (default is 6).
        Used for computing a normalized leak score based on the number of
        matched words using a non-linear curve.The score increases gradually
        up to a baseline threshold, then grows more slowly toward a cap of 1.0.
        This ensures that small matches yield low scores, while larger matches
        asymptotically approach 1.0. See Also  score_leak_baseline, curve_power
        and growth_rate.
    score_leak_baseline: float
        Threshold for transitioning from power curve to exponential growth (default is 0.7).
    curve_power : float, optional
        Power used in the initial curve for small matches (default is 1.25).
    growth_rate : float, optional
        Rate of exponential growth beyond the baseline (default is 0.3).
    prompt_pred_overlap_mask_th: int
        Treshold for sequence of words that overlap between prompt and pred that will be masked.
    min_allowed_index_distance: int
        Used to prune stop-words in LCS that are too far apart (distance larger than this number)
    alpha: float
        A scaling parameter controlling the influence of the matched sequences count
    base_stop_word_w:
        Base weight for stop-words, can be between 0.0-1.0 (default is 0.5).
        Calculated dynamically, final weight will be equal or greater than
        base_stop_word_w.

    Raises
    ----------
        ValueError :
            If any of the inputs does not allow to score

        TypeError :
            if one of the parameters has wrong type
    """

    adjacency_penalty_weight: float = field(default=0.2)
    distance_consistency_penalty_weight: float = field(default=0.2)
    word_leak_count_baseline: float = field(default=6.0)
    score_leak_baseline: float = field(default=0.7)
    curve_power: float = field(default=1.25)
    growth_rate: float = field(default=0.3)

    prompt_pred_overlap_mask_th: int = field(default=2)

    min_allowed_index_distance: int = field(default=1)
    alpha: float = field(default=1.0)
    base_stop_word_w: float = field(default=0.5)

    def __post_init__(self):

        type_check(float, base_stop_word_w=self.base_stop_word_w)
        type_check(float, alpha=self.alpha)
        type_check(float, adjacency_weight=self.adjacency_penalty_weight)
        type_check(float, distance_consistency_weight=self.distance_consistency_penalty_weight)
        type_check(float, word_leak_count_baseline=self.word_leak_count_baseline)
        type_check(float, score_leak_baseline=self.score_leak_baseline)
        type_check(float, curve_power=self.curve_power)
        type_check(float, grwoth_rate=self.growth_rate)
        type_check(int, prompt_pred_overlap_mask_th=self.prompt_pred_overlap_mask_th)
        type_check(int, min_allowed_index_distance=self.min_allowed_index_distance)

        value_check(
            is_positive(self.alpha),
            ERR_SIZE,
            "alpha",
        )

        value_check(
            is_range_correct(self.adjacency_penalty_weight),
            ERR_SIZE,
            "adjacency_weight",
            "[0.0-1.0]",
        )
        value_check(
            is_range_correct(self.distance_consistency_penalty_weight),
            ERR_SIZE,
            "distance_consistency_weight",
            "[0.0-1.0]",
        )
        value_check(
            is_range_correct(self.score_leak_baseline),
            ERR_SIZE,
            "score_leak_baseline",
            "[0.0-1.0]",
        )
        value_check(
            is_positive(self.word_leak_count_baseline),
            ERR_POSITIVE,
            "word_leak_count_baseline",
        )

        value_check(
            is_positive(self.curve_power),
            ERR_POSITIVE,
            "curve_power",
        )

        value_check(
            is_positive(self.growth_rate),
            ERR_POSITIVE,
            "curve_power",
        )

        value_check(
            is_positive(self.prompt_pred_overlap_mask_th),
            ERR_POSITIVE,
            "prompt_pred_overlap_mask_th",
        )

        value_check(
            is_positive(self.min_allowed_index_distance),
            ERR_POSITIVE,
            "min_allowed_index_distance",
        )


@dataclass
class RougeLmodScoreHandler(AbstractScoreHandler):
    """
    Implements a modified Rouge-L scoring handler for evaluating textual similarity
    between predicted and reference sentences, with enhancements for privacy-aware
    data leakage detection.

    This handler extends the standard Rouge-L metric by incorporating:
    - Prompt masking to reduce score inflation from prompt repetition.
    - Stop-word weighting and pruning based on adjacency and distance.
    - Non-linear score normalization that down-scores short matches to avoid
      overestimating minor leaks, while allowing longer matches to approach 1.0.
    - Sentence-level matching with configurable thresholds and penalties.

    Attributes
    ----------
    config : RougeLmodScoreHandlerConfig
        Configuration object that controls scoring behavior, including penalties,
        thresholds, and curve parameters.

    Key Features
    ------------
    - Sentence-level matching using longest common subsequence (LCS).
    - Penalization for non-adjacent or inconsistent word matches.
    - Optional stemming for flexible word comparison.
    - Prompt-aware masking to avoid false positives in leakage detection.
    - Aggregated scoring across multiple reference sentences.

    Methods
    -------
    compute_text_to_texts_match(...)
        Computes match scores between a single prediction and multiple references.

    compute_identiText_to_identiTexts_match(...)
        Computes match scores using IDentiText objects with identity metadata.


    Returns
    -------
    TextMatchScoreResult
        Contains detailed match results, scores, and metadata.
    """

    config: RougeLmodScoreHandlerConfig = field(default_factory=RougeLmodScoreHandlerConfig)

    def __post_init__(self):
        self.config = RougeLmodScoreHandlerConfig() if self.config is None else self.config

    def _append_max(
        self,
        scores_sent: List[MatchScoreResult],
        scores: List[MatchScoreResult],
    ):
        max_item = self._get_max(scores_sent)
        if max_item is not None:
            scores.append(max_item)

    def _get_max(
        self,
        scores_sent: List[MatchScoreResult],
    ):
        if not scores_sent:
            return None

        return max(scores_sent, key=lambda x: (x.score is not None, x.score))

    def _allow_to_append(self, mod_longest_seq_indices_pred, mod_pred):
        if not mod_longest_seq_indices_pred:
            return False

        return True

    def _preprocess_ref_texts(self, ref_texts: List[IDentiText], split=True):
        for ref_content in ref_texts:
            ref_content.split_to_sentences(split=split)

    def _preprocess_pred_text(self, pred_text: IDentiText, prompt_text: str, split=True):

        if not prompt_text:
            pred_text.split_to_sentences(split=split)
            return

        prompt_sentences = set(sent_tokenize(prompt_text))
        left_prompt_sentences = set()

        # 1.a Try to remove full prompt from pred
        for prompt_sentence in prompt_sentences:
            prompt_sentence = prompt_sentence.strip().lstrip()
            prompt_sentence = remove_ending_punc(prompt_sentence)
            found = pred_text.text.find(prompt_sentence)
            if found > 0:
                words_to_mask = prompt_sentence.split()
                masked_sentence = " ".join([M_A_S_K_P_R_E_D for _ in words_to_mask])
                pred_text.text = pred_text.text.replace(prompt_sentence, masked_sentence)
            else:
                left_prompt_sentences.add(prompt_sentence)

        # For all left_prompt_sentences mask with pred, normalize before
        pred_text.split_to_sentences(split=split)
        if not left_prompt_sentences:
            return

        prompt_sentences = [
            normalize_text(prompt, include_stop_words=True) for prompt in left_prompt_sentences
        ]

        # For pred masking we will not use stemmer - mask only identical
        # We will mask repetitions
        masked_pred_sentences = []
        for pred_sent in pred_text._norm_sentences:
            masked_pred_sent, _ = self._mask_overlaps_between_texts(
                pred_sent,
                prompt_sentences,
                M_A_S_K_P_R_E_D,
            )
            masked_pred_sentences.append(masked_pred_sent)
        pred_text._norm_masked_sentences = masked_pred_sentences
        return

    def _trace_back_lcs(
        self,
        ref_stemmed: list[str | Any],
        pred_stemmed: list[str | Any],
        lcs: list[list[int]],
    ) -> Tuple[list[int], list[int]]:
        """
        This method traces back to find the longest common subsequence indices

        ref_stemmed : list[str | Any]
            a list of stemmed words from the ground truth text (reference)
        pred_stemmed : list[str | Any]
            a list of stemmed words from the model generated output text (prediction)
        lcs : list[list[int]]
            the longest common sequence matrix

        Returns
        --------
            a tuple of two lists of int - longest common sequence indices for reference and prediction
        """
        m = len(ref_stemmed)
        n = len(pred_stemmed)

        # Trace back to find the longest common subsequence indices
        i, j = 0, 0
        longest_seq_indices_ref = []
        longest_seq_indices_pred = []

        while i < m and j < n:
            if ref_stemmed[i] == pred_stemmed[j]:
                longest_seq_indices_ref.append(i)
                longest_seq_indices_pred.append(j)
                i += 1
                j += 1
            else:
                if lcs[i + 1][j] >= lcs[i][j + 1]:
                    i += 1
                else:
                    j += 1

        return longest_seq_indices_ref, longest_seq_indices_pred

    def _calculate_lcs(
        self, ref_stemmed: list[str | Any], pred_stemmed: list[str | Any]
    ) -> list[list[int]]:
        """
        This method calculates longest common sequence

        ref_stemmed : list[str | Any]
            a list of stemmed words from the ground truth text
        pred_stemmed : list[str | Any]
            a list of stemmed words from the model generated output text

        Returns
        --------
            a list of lists of int - longest common sequence matrix
        """
        m = len(ref_stemmed)
        n = len(pred_stemmed)

        # Initialize the LCS table with zeros
        lcs = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill the LCS table
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if ref_stemmed[i] == pred_stemmed[j]:
                    lcs[i][j] = lcs[i + 1][j + 1] + 1
                else:
                    lcs[i][j] = max(lcs[i + 1][j], lcs[i][j + 1])

        return lcs

    def _rouge_l_longest_sequence_mod(
        self,
        ref_text: str,
        pred_text: str,
        mask_repeatitions=False,
        stemmer: AbstractStemmer = None,
    ) -> Tuple[list[list[int]], list[list[int]], list[str], list[str]]:
        """
        This method used to find matched words and their indexes in ref_text and pred_text

        ref_text : str
            the ground truth text
        pred_text : str
            the model generated output text

        Returns
        --------
            tuple of 1) a list of  int indexes of matched words in ref_text, 2) a list of int indexes of matched words
            in pred_text, 3) a list of words in ref_text and 4) a list of words in pred_text
        """

        # Tokenize the reference and predicted texts
        ref = ref_text.split()
        pred = pred_text.split()

        # Stem the tokenized words and convert to lower case

        ref_stemmed = ref
        pred_stemmed = pred
        if stemmer:
            ref_stemmed = [stemmer.stem(word.lower()) for word in ref]
            pred_stemmed = [stemmer.stem(word.lower()) for word in pred]

        matching_indices_pred_g_g = []
        matching_indices_ref_g_g = []
        while True:
            _, _, done = self._find_matching_indices(
                mask_repeatitions,
                matching_indices_pred_g_g,
                matching_indices_ref_g_g,
                pred_stemmed,
                ref_stemmed,
            )

            if done:
                break

        return matching_indices_ref_g_g, matching_indices_pred_g_g, ref, pred

    def _find_matching_indices(
        self, include_repeatitions, indices_pred_g_g, indices_ref_g_g, pred, ref
    ):

        # mask ref if not include_repeatitions
        mod_ref = ref
        if not include_repeatitions:
            mod_ref = ref[:]
            for group in indices_ref_g_g:
                for idx in group:
                    mod_ref[idx] = M_A_S_K_R_E_F

        # mask pred
        mod_pred = pred[:]
        for group in indices_pred_g_g:
            for idx in group:
                mod_pred[idx] = M_A_S_K_P_R_E_D

        # Calculate LCS for modified versions
        mod_lcs = self._calculate_lcs(mod_ref, mod_pred)
        # Trace back LCS for modified versions to find indices
        (
            mod_longest_seq_indices_ref,
            mod_longest_seq_indices_pred,
        ) = self._trace_back_lcs(mod_ref, mod_pred, mod_lcs)

        # in a certain group if there is a matched word which is too far
        # from the rest remove it
        self._remove_non_adjacent_stop_words(
            mod_longest_seq_indices_pred, mod_longest_seq_indices_ref, mod_ref
        )
        self._remove_non_adjacent_stop_words(
            mod_longest_seq_indices_pred, mod_longest_seq_indices_ref, mod_ref, False
        )

        # Combine original and modified LCS indices
        if self._allow_to_append(mod_longest_seq_indices_pred, mod_pred):
            indices_ref_g_g.append(mod_longest_seq_indices_ref)
            indices_pred_g_g.append(mod_longest_seq_indices_pred)
            done = False
        else:
            done = True

        return mod_pred, mod_ref, done

    def _remove_non_adjacent_stop_words(
        self,
        mod_longest_seq_indices_pred,
        mod_longest_seq_indices_ref,
        mod_ref,
        from_end=True,
    ):
        if len(mod_longest_seq_indices_ref) == 1:
            ind_r = mod_longest_seq_indices_ref[0]
            if is_stop_word(mod_ref[ind_r]) and len(mod_ref[ind_r]) <= 3:
                mod_longest_seq_indices_pred.clear()
                mod_longest_seq_indices_ref.clear()

                return

        ind1 = -1 if from_end else 0
        ind2 = -2 if from_end else 1
        while True:
            if len(mod_longest_seq_indices_pred) > 1:
                diff_p = (
                    abs(mod_longest_seq_indices_pred[ind1] - mod_longest_seq_indices_pred[ind2]) - 1
                )
                diff_r = (
                    abs(mod_longest_seq_indices_ref[ind1] - mod_longest_seq_indices_ref[ind2]) - 1
                )
                if (
                    diff_p > self.config.min_allowed_index_distance
                    or diff_r > self.config.min_allowed_index_distance
                ) and is_stop_word(mod_ref[mod_longest_seq_indices_ref[ind1]]):
                    del mod_longest_seq_indices_pred[ind1]
                    del mod_longest_seq_indices_ref[ind1]
                else:
                    break
            else:
                break

    def _mask_words_list(self, group_lst, words, mask_word):
        for group in group_lst:
            for index in group:
                words[index] = mask_word

    def _normalize_data_leak_score_non_linear_to_cap(
        self,
        num_matched_words: float,
        baseline_word_count: float = 8,
        baseline_score: float = 0.8,
        curve_power: float = 1.25,
        growth_rate: float = 0.3,
    ) -> float:
        """
        Computes a normalized leak score based on the number of matched words using a non-linear curve.

        The score increases gradually up to a baseline threshold, then grows more slowly toward a cap of 1.0.
        This method ensures that small matches yield low scores, while larger matches asymptotically approach 1.0.

        Parameters
        ----------
        num_matched_words : float
            Number of words matched between prediction and reference.

        baseline_word_count : float, optional
            Threshold for transitioning from power curve to exponential growth (default is 6).

        baseline_score : float, optional
            Score value at the baseline word count (default is 0.7).

        curve_power : float, optional
            Power used in the initial curve for small matches (default is 1.25).

        growth_rate : float, optional
            Rate of exponential growth beyond the baseline (default is 0.3).

        Returns
        -------
        float
            A normalized score between 0.0 and 1.0.
        """

        if num_matched_words <= 0:
            return 0.0

        if num_matched_words <= baseline_word_count:
            t = num_matched_words / baseline_word_count
            expected_score = baseline_score * (t**curve_power)
        else:
            t = num_matched_words - baseline_word_count
            expected_score = 1.0 - (1.0 - baseline_score) * math.exp(-growth_rate * t)

        return min(1.0, max(0.0, expected_score))

    def _score_reference_prediction_pair(
        self,
        ref_norm_sentence: str,
        pred_norm_masked_sentence: str,
        pred_norm_sentence: str,
        stemmer: AbstractStemmer = None,
    ) -> MatchScoreResult:
        """
        Scores a reference-prediction pair based on absolute number of matched words, adjacency, and consistency scores.
        If prompt is not None, reduce score if the prediction repeats parts of the prompt

        ref_norm_sentence : str
            A reference sentence.
        pred_norm_masked_sentence : str
            A prediction sentence.
        prompt_sentences : str
            The LLM prompt sentences.

        Returns:
        -------
            MatchScoreResult object with data related to the scoring
        """

        pred_norm_masked_words = pred_norm_masked_sentence.split()
        pred_norm_words = pred_norm_sentence.split()

        # match pred and ref
        (
            longest_seq_indices_ref_g_g,
            longest_seq_indices_pred_g_g,
            ref_norm_words,
            _,
        ) = self._rouge_l_longest_sequence_mod(
            ref_norm_sentence, pred_norm_masked_sentence, False, stemmer=stemmer
        )

        # Calculate the number of matched words
        scores = []

        for index, longest_seq_indices_pred_g in enumerate(longest_seq_indices_pred_g_g):
            longest_seq_indices_ref_g = longest_seq_indices_ref_g_g[index]

            norm_num_matched_words = self._count_words(
                longest_seq_indices_pred_g, pred_norm_masked_words, self.config.base_stop_word_w
            )

            matched_words_score = self._normalize_data_leak_score_non_linear_to_cap(
                norm_num_matched_words,
                self.config.word_leak_count_baseline,
                self.config.score_leak_baseline,
                self.config.curve_power,
                self.config.growth_rate,
            )

            # Calculate the distance consistency score
            distance_consistency_score = self._compute_distance_consistency_score(
                longest_seq_indices_ref_g, longest_seq_indices_pred_g
            )

            # Calculate the adjacency score
            adjacency_score = self._compute_adjacency_score(longest_seq_indices_pred_g)

            lcs_penalty = self.config.adjacency_penalty_weight * (
                1 - adjacency_score
            ) + self.config.distance_consistency_penalty_weight * (1 - distance_consistency_score)

            score = matched_words_score - lcs_penalty
            score = max(min(score, 1.0), 0.0)
            scores.append(score)

        # Final - aggregate and build result
        final_score = calculate_aggregated_score(scores, self.config.alpha)
        result = MatchScoreResult()
        result.score = final_score
        result.norm_ref_words = ref_norm_words
        result.norm_pred_words = pred_norm_words
        result.norm_pred_words_masked = pred_norm_masked_words
        result.norm_ref_indices = longest_seq_indices_ref_g_g
        result.norm_pred_indices = longest_seq_indices_pred_g_g

        result.norm_pred = pred_norm_sentence
        result.norm_ref = ref_norm_sentence

        return result

    def _count_words(self, group, pred_words, base_stop_word_w):
        stop_words_count = sum(1 for w_index in group if is_stop_word(pred_words[w_index]))
        non_stop_words_count = len(group) - stop_words_count

        stop_word_w = (
            base_stop_word_w + (base_stop_word_w * (non_stop_words_count / len(group)))
            if len(group) > 0
            else 1
        )

        wsum = 0
        for w_index in group:
            word = pred_words[w_index]
            wsum += stop_word_w if is_stop_word(word) else 1

        return wsum

    def _mask_overlaps_between_texts(
        self,
        target_sentence,
        source_sentences,
        mask_word,
    ):
        ngram_indexes_to_mask = []

        target_words = target_sentence.split()
        masked_target_words = target_words[:]
        if source_sentences is not None:

            for source_sentence in source_sentences:
                source_words = source_sentence.split()

                while True:
                    target_indices_g_g = find_common_non_overlapping_ngrams(
                        masked_target_words,
                        source_words,
                        min_len=self.config.prompt_pred_overlap_mask_th,
                    )

                    if len(target_indices_g_g) == 0:
                        break

                    self._mask_words_list(target_indices_g_g, masked_target_words, mask_word)

                ngram_indexes_to_mask.append(target_indices_g_g)

        masked_target_sentence = " ".join(masked_target_words)

        return masked_target_sentence, target_words

    def _compute_distance_consistency_score(
        self, ref_indices: list[int], pred_indices: list[int]
    ) -> float:
        """
        This method measures how consistent the distances between matched words are in the reference and prediction

        ref_indices : list[int]
            a list of indices of the matched words in the ground truth text ()
        pred_indices : list[int]
            a list of indices of the matched words in the model generated output

        Returns
        --------
            a float value of the adjacency score
        """
        if len(ref_indices) != len(pred_indices):
            raise ValueError("Both arrays must have the same length")

        if len(pred_indices) == 1:
            return 1.0

        ref_distances = np.diff(ref_indices)
        pred_distances = np.diff(pred_indices)

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(ref_distances - pred_distances))

        # Normalize: score is 1 when MAE is 0, and approaches 0 as MAE increases
        consistency_score = 1 / (1 + mae)

        return consistency_score

    def _compute_adjacency_score(self, pred_indices: list[int]) -> float:
        """
        This method measures the weighted adjacency score of the matched words
        in the prediction text, where smaller distances are given higher weight.

        pred_indices : list[int]
            A list of indices of the matched words in the model-generated output.

        Returns
        --------
            A float value of the weighted adjacency score.
        """
        pred_indices.sort()

        if len(pred_indices) <= 1:
            return 1.0

        # Compute the distances between consecutive indices
        pred_distances = np.diff(pred_indices)

        # Assign weights to distances (inverse of the distance)
        weights = 1 / pred_distances

        # Compute the weighted adjacency score
        adjacency_score = sum(weights) / len(pred_distances)

        return adjacency_score

    def compute_text_to_texts_match(
        self,
        ref_texts: List[str],
        pred_text: str,
        prompt_text: str = None,
        stemmer: AbstractStemmer = None,
        score_leak_th: float = 0.0,
        split=True,
    ) -> TextMatchScoreResult | None:
        """
        This method used to calculate similarity between a string and a list of strings by
        RougeL modified metric. Use it when the id of the text is not important.
        The system will assign an id based on the order of the text in the list
        for ref_texts, and id 0 for pred_text.

        ref_texts : List[str]
           reference texts ((multi-sentence))
        pred_text : str
            source text (e.g. LLM generated text) to compare to the ref text
            (can have multiple sentences)
        prompt_text : str, optional
             the LLM prompt (multi-sentence)
        stemmer : AbstractStemmer
            Use it if you want stemmed words to be matched as identical (e.g. discuss, discussing)
        score_leak_th : float
             only matches above this threshold are returned
        split: Boolean
             Indicates if to split the text to sentences before matching. Default is True.

        Returns
        --------

        TextMatchScoreResult - result of the match

        """

        identi_refs = [
            IDentiText(ref_text, TextIdentity(index)) for index, ref_text in enumerate(ref_texts)
        ]

        return self.compute_identiText_to_identiTexts_match(
            identi_refs,
            IDentiText(pred_text, TextIdentity(0)),
            prompt_text,
            stemmer=stemmer,
            score_leak_th=score_leak_th,
            split=split,
        )

    def compute_identiText_to_identiTexts_match(
        self,
        ref_texts: List[IDentiText],
        pred_text: IDentiText,
        prompt_text: str = None,
        stemmer: AbstractStemmer = None,
        score_leak_th: float = 0.0,
        split=True,
    ) -> TextMatchScoreResult | None:
        """
        This method used to calculate similarity between list of reference texts and pred_text
        a TextContent is a text with id information.

        ref_texts : List[IDentiText]
           a list of reference texts
        pred_text : IDentiText
            the LLM generated output  (multi-sentence)
        prompt_text : str
            the LLM prompt (multi-sentence)
         stemmer : AbstractStemmer
            Use it if you want stemmed words to be matched as identical (e.g. discuss, discussing)
        score_leak_th : float
            Only matches above this threshold are returned
        split: Boolean
            Indicates if to split the text to sentences before matching. Default is True.


        Returns
        --------

        TextMatchScoreResult - result of the match

        """

        # Step 1: preprocess pred and ref - normalization,masking
        self._preprocess_pred_text(pred_text, prompt_text, split)
        self._preprocess_ref_texts(ref_texts, split)

        # Step #2 - Match & Score
        score_leak_th = score_leak_th if score_leak_th is not None else 0.0
        all_above_th = []
        best_score_for_each_pred_sentence_lst = []
        for pred_sent_id, pred_norm_masked_sent in enumerate(pred_text._norm_masked_sentences):
            scores_sent = []
            for ref_text in ref_texts:
                for ref_sent_id, ref_norm_sent in enumerate(ref_text._norm_sentences):
                    result = self._score_reference_prediction_pair(
                        ref_norm_sent,
                        pred_norm_masked_sent,
                        pred_text._norm_sentences[pred_sent_id],
                        stemmer,
                    )

                    if result is not None and result.score >= 0.0:
                        result.prompt = prompt_text
                        result.ref = ref_text._sentences[ref_sent_id]
                        result.pred = pred_text._sentences[pred_sent_id]
                        result.ref_identity = TextIdentity(**ref_text.identity.__dict__)
                        result.pred_identity = TextIdentity(**pred_text.identity.__dict__)
                        result.ref_identity.secondary_id = ref_sent_id
                        result.pred_identity.secondary_id = pred_sent_id

                        scores_sent.append(result)

                    if result and result.score >= score_leak_th:
                        all_above_th.append(result)

            self._append_max(scores_sent, best_score_for_each_pred_sentence_lst)

        max_score = self._get_max(best_score_for_each_pred_sentence_lst)

        return TextMatchScoreResult(
            score_leak_th,
            prompt_text,
            pred_text,
            ref_texts,
            max_score,
            all_above_th,
            score_aggregation_alpha=self.config.alpha,
        )
