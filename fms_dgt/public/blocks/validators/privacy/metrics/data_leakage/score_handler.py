# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

# Third Party
from nltk import PorterStemmer

# Local
from fms_dgt.public.blocks.validators.privacy.metrics.text import (
    IDentiText,
    TextIdentity,
)
from fms_dgt.public.blocks.validators.privacy.metrics.util import (
    calculate_aggregated_score,
    select_highest_scores_from_group,
)


class AbstractStemmer:
    """
    An abstract class for Stemmer. Use it to wrap your preferred stemmer.
    """

    def stem(self, word: str) -> str:
        raise NotImplementedError()


class DefaultStemmer(AbstractStemmer):
    """
    A basic stemmer implementation using the Porter stemming algorithm.

    This class provides a simple interface for stemming individual words
    using the widely-used PorterStemmer from the NLTK library.

    Methods
    -------
    stem(word: str) -> str
        Returns the stemmed form of the input word.

    Example
    -------
    >>> stemmer = DefaultStemmer()
    >>> stemmer.stem("running")
    'run'
    """

    porter_stemmer = PorterStemmer()

    def stem(self, word: str) -> str:
        return self.porter_stemmer.stem(word)


@dataclass
class MatchScoreResult:
    """
    Represents a match between a pair of pred and ref sentences.
    All the fields are initialized internally by the code,
    should not be passed during instantiation.

        score:  float
            The leak score for the pred and ref sentences in this result

        norm_pred: str
            The prediction sentence in its normalized form (used by the metric)

        norm_ref: str
            The reference sentence in its normalized form (used by the metric)

        norm_pred_words: List[str]
            The prediction sentence words

        norm_ref_words: List[str]
            The reference sentence words

        norm_pred_words_masked: List[str]
            The prediction words where part of the words might be masked due to prompt overlap.

        norm_pred_indices:List[List[int]]
            The indices of the words in pred that matched the ref

        norm_ref_indices:List[List[int]]
            The indices of the words in ref that matched the pred

        pred_identity:TextIdentity
            Identity of the pred

        ref_identity:TextIdentity
            Identify of the ref

        pred: str
            The prediction sentence in its original form (non-normalized)

        ref: str
            The reference sentence in its original form (non-normalized)

        prompt: str
            The prompt text in its original form (non-normalized)
    """

    score: float = field(init=False, default=0.0)
    norm_pred: str = field(init=False, default=None)
    norm_ref: str = field(init=False, default=None)
    norm_pred_words: List[str] = field(init=False, default=None)
    norm_ref_words: List[str] = field(init=False, default=None)
    norm_pred_words_masked: List[str] = field(init=False, default=None)
    norm_pred_indices: List[List[int]] = field(init=False, default=None)
    norm_ref_indices: List[List[int]] = field(init=False, default=None)

    ref_identity: TextIdentity = field(init=False, default=None)
    pred_identity: TextIdentity = field(init=False, default=None)
    ref: str = field(init=False, default=None)
    pred: str = field(init=False, default=None)
    prompt: str = field(init=False, default=None)


@dataclass
class TextMatchScoreResult:
    """
    Represents a match between text content and a set of ref contents.
    Each content may contain several sentences. The system breaks each to sentences
    and detects the maximum scores between any sentence from the text to any sentence
    from the references.

        score_th : float
            Only matches with score above this number are included in this result

        prompt : str
            The prompt that was used for the LLM's generated text.

        text: str
            the LLM's generated text

        references: List[IDentiText]
            The texts you wish to evaluate for leakage wrt to the generated text.
            Examples: for synthetic data this can be the seed. For RAG this can be the RAG context,
            or it can be training data samples if you wish to identify leaks from the training dataset.

        max_match: MatchScoreResult
            A match (leak) got the maximum score (above score_th)

        all_matches_above_th: List[MatchScoreResult]
            all the matches (leaks) that got a score above score_th

        score_aggregation_alpha: float
            Used to calculate an aggregated score for all the input text sentences.
            The aggregated score is always higher than the max score.
            score_aggregation_alpha is a scaling parameter that controls the influence of additional scores,
            higher means more influence, higher aggregated score.

        aggregated_score_for_all_max_match_text_sentences: float
            An aggregated score for all generated text sentences.
            Should not be initialized. Calculated by the code.

        max_match_for_each_text_sentence: List[MatchScoreResult]
            The max match for every generated text sentence, if above score_th.
            Should not be initialized. Calculated by the code.

        max_match_for_each_ref_sentence: List[MatchScoreResult]
            The max match for every reference text sentence, if above score_th.
            Should not be initialized. Calculated by the code.

        aggregated_score_for_all_max_match_ref_sentences: float
            An aggregated score for all reference text sentences.
            Should not be initialized. Calculated by the code.
    """

    score_th: float = field(init=True, default=None)
    prompt: str = field(init=True, default=None)
    text: IDentiText = field(init=True, default=None)
    references: List[IDentiText] = field(init=True, default=None)
    max_match: MatchScoreResult = field(init=True, default=None)
    all_matches_above_th: List[MatchScoreResult] = field(init=True, default_factory=list)
    score_aggregation_alpha: float = field(init=True, default=1.0)

    # Not at init

    # For Text (generated) side
    aggregated_score_for_all_max_match_text_sentences: float = field(init=False, default=0.0)

    max_match_for_each_text_sentence: List[MatchScoreResult] = field(
        init=False, default_factory=list
    )

    # Ref side
    max_match_for_each_ref_sentence: List[MatchScoreResult] = field(
        init=False, default_factory=list
    )

    aggregated_score_for_all_max_match_ref_sentences: float = field(init=False, default=0.0)

    def __post_init__(self):
        self._calculate_aggregated_score_for_all_max_match_text_sentences()
        self._calculate_aggregated_score_for_all_max_match_ref_sentences()

    def _calculate_aggregated_score_for_all_max_match_text_sentences(
        self,
    ) -> tuple[float, List[MatchScoreResult]]:
        max_scores_per_pred = select_highest_scores_from_group(
            self.all_matches_above_th, "score", ["norm_pred"]
        )
        # remove duplicate match for same ref & ref-indices
        max_scores_per_pred = select_highest_scores_from_group(
            max_scores_per_pred, "score", ["norm_ref", "norm_ref_indices"]
        )
        self.max_match_for_each_text_sentence = sorted(
            max_scores_per_pred, key=lambda x: x.score, reverse=True
        )

        scores = [item.score for item in self.max_match_for_each_text_sentence]
        self.aggregated_score_for_all_max_match_text_sentences = calculate_aggregated_score(
            scores, alpha=self.score_aggregation_alpha
        )

    def _calculate_aggregated_score_for_all_max_match_ref_sentences(
        self,
    ) -> tuple[float, List[MatchScoreResult]]:
        max_scores_per_ref = select_highest_scores_from_group(
            self.all_matches_above_th, "score", ["ref_identity"]
        )

        # sort by index to get the ref sentences in order for next search of
        # best max average sequence
        self.max_match_for_each_ref_sentence = sorted(
            max_scores_per_ref, key=lambda x: x.ref_identity, reverse=False
        )

        scores = [item.score for item in self.max_match_for_each_ref_sentence]
        self.aggregated_score_for_all_max_match_ref_sentences = calculate_aggregated_score(
            scores, alpha=self.score_aggregation_alpha
        )


class AbstractScoreHandler(ABC):
    """
    An abstract class to be implemented by specific score handlers.
    """

    @abstractmethod
    def compute_identiText_to_identiTexts_match(
        self,
        ref_texts: List[IDentiText],
        pred_text: IDentiText,
        prompt_text: str = None,
        stemmer: AbstractStemmer = None,
        score_leak_th: float = 0.0,
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

        Returns
        --------

        TextMatchScoreResult - result of the match

        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
