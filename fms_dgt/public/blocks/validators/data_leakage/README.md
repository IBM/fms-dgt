# Data Leakage Validator Block

A validator block that measures lexical similarity between model-generated text and reference texts (e.g., training data or seed examples). It detects when generated text contains fragments that appear to be copied from the source data.

Registered as `validators/data_leak`.

## Data specification

### Required

- `input` (str): The generated text to check for data leakage.

### Optional

- `local_context` (List[str]): Reference texts specific to this input. Either `local_context` per input or a global `context` passed to `execute()` is required.

### Output

- `is_valid` (bool): `True` if the leakage score is below the threshold.
- `metadata` (dict): Contains `score` and, if invalid, a `reason` string.

## How it works

The metric is based on a **modified Rouge-L** algorithm. Standard Rouge-L finds a single Longest Common Subsequence (LCS) between two texts and scores based on its length. This implementation extends it in several ways:

**Iterative multi-LCS matching**: Instead of finding just one LCS, the algorithm iteratively finds multiple common subsequences. After finding the first LCS, matched words are masked and the search repeats until no more matches are found. This captures cases where generated text copies multiple separate fragments from the source.

**Scoring beyond length**: Each matched subsequence is scored considering word importance (stop words receive lower weight), adjacency (how close together matched words are), and distance consistency (whether gaps between matched words are similar in reference and prediction). The raw word count is converted to a score (0.0–1.0) using a non-linear curve.

**Prompt-aware masking**: When a prompt is provided, overlapping words between the prompt and the prediction are masked before scoring. This prevents score inflation when the model repeats parts of the prompt.

**Sentence-level matching with aggregation**: Both texts are split into sentences and matching is done sentence-by-sentence. Scores are aggregated across all sentences using a configurable `alpha` parameter.

## Configuration

- `threshold` (float, default 1.1): Score above which an input is considered leaked. A threshold above 1.0 effectively disables filtering.
- `adjacency_penalty_weight` (float, default 0.2): Weight for the adjacency penalty.
- `distance_consistency_penalty_weight` (float, default 0.2): Weight for the distance consistency penalty.
- `word_leak_count_baseline` (float, default 6.0): Baseline word count for the non-linear scoring curve.
- `score_leak_baseline` (float, default 0.7): Score value at the baseline word count.
- `base_stop_word_w` (float, default 0.5): Base weight for stop words (0.0–1.0).
- `alpha` (float, default 1.0): Controls influence of additional sentence scores during aggregation.

## Dependencies

Requires `nltk` and `ftfy` (included in the `[public]` optional dependency group).
