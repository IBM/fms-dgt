# Standard

# Local
from fms_dgt.public.blocks.validators.privacy.metrics.data_leakage.rougelmod_score_handler import (
    RougeLmodScoreHandler,
)
from fms_dgt.public.blocks.validators.privacy.metrics.data_leakage.score_handler import (
    AbstractScoreHandler,
    AbstractStemmer,
    TextMatchScoreResult,
)
from fms_dgt.public.blocks.validators.privacy.metrics.text import (
    IDentiText,
    TextIdentity,
)


def compute_leak_from_context(
    context: str,
    pred_text: str,
    prompt_wo_context_text: str,
    score_handler: AbstractScoreHandler = RougeLmodScoreHandler(),
    stemmer: AbstractStemmer = None,
    score_leak_th: float = None,
) -> TextMatchScoreResult | None:
    """
    This method used to calculate similarity by RougeL modified metric,
    between text and reference texts that are searched using a search engine.
    Note that in this use case, if you provide a prompt, then leaks from the prompt are ignored,
    in this use case, if the LLM model repeated the prompt, it is not considered a leak,
    only leaks from the refs that are not in the prompt are reported.
    If this is not wanted, provide an empty prompt or only the parts that you wish the metric to ignore.
    For a use case where you wish to understand if the context was leaked use compute_leak_from_context

    context : str
        The context given to the LLM  (each can have multiple sentences)
    pred_text : str
            source text (e.g. LLM generated text) to compare to the ref text
            (can have multiple sentences)
    prompt_wo_context_text : str, optional
                the LLM prompt, not including the context (multi-sentence)
    stemmer : AbstractStemmer
            Use it if you want stemmed words to be matched as identical (e.g. discuss, discussing)
    score_leak_th : float
                only matches above this threshold are returned

    Returns
    --------

    TextMatchScoreResult - result of the match

    """
    # prepare windows for search

    return score_handler.compute_text_to_texts_match(
        {IDentiText(context)},
        IDentiText(pred_text),
        prompt_wo_context_text,
        stemmer=stemmer,
        score_leak_th=score_leak_th,
    )


def compute_leak_from_refs(
    ref_texts: list[str],
    pred_text: str,
    prompt_text: str,
    score_handler: AbstractScoreHandler = RougeLmodScoreHandler(),
    stemmer: AbstractStemmer = None,
    score_leak_th: float = None,
) -> TextMatchScoreResult | None:
    """
    This method used to calculate similarity by RougeL modified metric,
    between text and provided reference texts.
    Note that in this use case, if you provide a prompt, then leaks from the prompt are ignored,
    in this use case, if the LLM model repeated the prompt, it is not considered a leak,
    only leaks from the refs that are not in the prompt are reported.
    If this is not wanted, provide an empty prompt or only the parts that you wish the metric to ignore.
    For a use case where you wish to understand if the context was leaked use compute_leak_from_context


    ref_texts : set[IDentiText]
        candidate leaking reference texts  (each can be multi-sentence)
    pred_text : str
            source text (e.g. LLM generated text) to compare to the ref text
            (can have multiple sentences)
    prompt_text : str, optional
                the LLM prompt, if provided then leaks from the prompt are not reported (can be multi-sentence)
    stemmer : AbstractStemmer
            Use it if you want stemmed words to be matched as identical (e.g. discuss, discussing)
    score_leak_th : float
                only matches above this threshold are returned

    Returns
    --------

    TextMatchScoreResult - result of the match

    """
    # prepare windows for search

    identi_refs = [
        IDentiText(ref_text, TextIdentity(index)) for index, ref_text in enumerate(ref_texts)
    ]
    return score_handler.compute_text_to_texts_match(
        set(identi_refs),
        IDentiText(pred_text),
        prompt_text,
        stemmer=stemmer,
        score_leak_th=score_leak_th,
    )
