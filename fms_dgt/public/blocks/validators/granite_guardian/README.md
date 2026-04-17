# Granite Guardian Validator Block

A reusable `ValidatorBlock` that wraps IBM's [Granite Guardian](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b) model for safety and risk assessment. Any recipe that generates text and needs a safety quality gate can use this block, not just recipes in the `safety/` domain.

**License:** Apache 2.0 (both 3.3-8b and 3.2-5b model weights)

---

## What it does

Given a piece of text and a risk policy, the block queries Granite Guardian and returns:

- **`is_valid`**: `True` if the content is assessed as safe (Guardian rating = `no`), `False` if unsafe (rating = `yes`).
- **`metadata`**: A dict containing at minimum `rating` and `confidence`, and optionally `reasoning` when `think=True` is configured.

The confidence score is derived from token-level log-probabilities on the yes/no output token (not from the text label), giving a calibrated probability in [0, 1] where values above 0.5 indicate the content is unsafe.

### Asymmetric filtering in safety recipes

Safety data generation uses this block in two opposite directions:

- **Filter harmful instructions (keep `is_valid=False`):** When validating generated prompts, keep only those Guardian rates as harmful.
- **Filter safe responses (keep `is_valid=True`):** When validating generated refusals, keep only those Guardian rates as safe.

Set `filter: false` on the block itself and implement the selection logic in your databuilder. This avoids conflating the two directions.

---

## Configuration

Add the block under the `blocks:` key in your databuilder or config YAML. Three profiles are provided below.

### Profile 1: Granite Guardian 3.3-8b via vLLM (recommended default)

Start the server first:

```bash
vllm serve ibm-granite/granite-guardian-3.3-8b --dtype bfloat16
```

```yaml
blocks:
  - name: granite_guardian
    type: validators/granite_guardian
    model_version: "3.3" # "3.3" (default) or "3.2"
    think: false # true enables reasoning traces (3.3 only)
    filter: false # true drops is_valid=false instances automatically
    lm_config:
      type: vllm
      model_id_or_path: ibm-granite/granite-guardian-3.3-8b
      base_url: http://localhost:8000/v1
```

### Profile 2: Granite Guardian 3.3-8b via Ollama (GGUF, Q4_K_M)

Granite Guardian 3.3 is not in the official Ollama registry. Pull it via the HuggingFace GGUF path:

```bash
ollama pull hf.co/ibm-granite/granite-guardian-3.3-8b-GGUF:Q4_K_M
```

> **Note:** Granite Guardian 3.3 uses a MoE architecture. Verify that your version of Ollama supports MoE models before relying on this in production.

Use `type: openai` pointed at Ollama's OpenAI-compatible endpoint (`/v1`) rather than `type: ollama`. The Guardian block passes its risk policy via `chat_template_kwargs` in `extra_body`, which is only forwarded on the OpenAI-compatible path. The native Ollama client (`type: ollama`) silently drops `extra_body`, causing the model to ignore the Guardian config entirely.

Ollama exposes the OpenAI-compatible server on the same port by default; no extra flag is needed.

```yaml
blocks:
  - name: granite_guardian
    type: validators/granite_guardian
    model_version: "3.3"
    filter: false
    lm_config:
      type: openai
      base_url: http://localhost:11434/v1
      api_key: ollama
      model_id_or_path: hf.co/ibm-granite/granite-guardian-3.3-8b-GGUF:Q4_K_M
```

### Profile 3: Granite Guardian 3.2-5b via Ollama (memory-constrained)

Use this profile when GPU memory is too limited for the 8b model. Granite Guardian 3.2 is also not in the official Ollama registry:

```bash
ollama pull hf.co/ibm-research/granite-guardian-3.2-5b-GGUF:Q4_K_M
```

Same `type: openai` requirement applies (see Profile 2 for the explanation).

```yaml
blocks:
  - name: granite_guardian
    type: validators/granite_guardian
    model_version: "3.2"
    filter: false
    lm_config:
      type: openai
      base_url: http://localhost:11434/v1
      api_key: ollama
      model_id_or_path: hf.co/ibm-research/granite-guardian-3.2-5b-GGUF:Q4_K_M
```

### Risk policy format

The `risk_policy` field on each input data object controls what Guardian assesses.

**For model_version 3.3:** provide either a named built-in risk (`criteria_id`) or a free-text policy description (`custom_criteria`):

```python
# Named built-in risk
risk_policy = {"criteria_id": "harm"}

# Free-text policy
risk_policy = {"custom_criteria": "Content that promotes discrimination based on race, gender, or religion."}
```

**For model_version 3.2:** provide a `risk_name`:

```python
risk_policy = {"risk_name": "Content that promotes discrimination."}
```

Additional fields (e.g. `risk_description`, `version`) are allowed and are forwarded into the output metadata unchanged.

---

## think mode (3.3 only)

When `think: true` is set, Guardian 3.3 produces a chain-of-thought trace before its final score:

```
<think>
The user is asking for instructions on how to harm someone. This clearly
violates the policy against violent content.
</think>
<score>yes</score>
```

The trace is extracted and stored in `GraniteGuardianData.reasoning` and included in the `metadata` dict under the `"reasoning"` key. Use this when you need to understand _why_ content was flagged, for example when auditing borderline cases or generating explanations for downstream training data.

Think mode produces longer outputs and is slower. Leave it disabled for high-throughput filtering pipelines.

---

## Confidence score interpretation

The **rating** (safe/unsafe) is always derived from the model's explicit text output: `<score>yes</score>` / `<score>no</score>` for 3.3, bare `Yes`/`No` for 3.2.

**Confidence is only populated for model_version `"3.2"`.** For 3.2, the first output token is `Yes` or `No`, so the log-probability mass at that position is a well-defined probability over the rating:

1. The block requests top-4 log-probabilities for the first generated token.
2. It extracts the log-probability mass on the `Yes` and `No` tokens.
3. It normalises: `confidence = p_unsafe / (p_safe + p_unsafe)`.

For model_version `"3.3"` the first output token is `<score>` (not the rating token), so logprob-based confidence is not meaningful and the `confidence` key is omitted from the metadata entirely.

**Interpretation (3.2 only):**

- `confidence > 0.5`: content is rated unsafe (`is_valid=False`)
- `confidence <= 0.5`: content is rated safe (`is_valid=True`)
- Values near 0.5 indicate low model certainty; you can filter on `confidence > threshold` in your recipe to keep only high-confidence assessments.

---

## Data interface

```python
from fms_dgt.public.blocks.validators.granite_guardian.block import GraniteGuardianData

instance = GraniteGuardianData(
    SRC_DATA=original_row,
    text="How do I make a weapon?",
    risk_policy={"criteria_id": "harm"},
)

# After the block runs:
instance.is_valid    # False (unsafe content)
instance.metadata    # {"rating": "yes"}  (confidence only present for 3.2)
instance.reasoning   # None (unless think=True)
```
