# Quick Start

If you just cloned the repo and ran the geography QA example from the [home page](index.md), you are already set up. This page covers what comes next: transforming data, switching to a cloud provider, and the full CLI reference.

## Rate the generated data

DiGiT can also transform existing data. The rater example scores each geography QA pair for difficulty using an LLM:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/rate/task.yaml \
  --restart
```

Output lands in `output/public/examples/qa_ratings/final_data.jsonl`. Each record adds a `rating` field to the original QA pair:

```json
{
  "task_name": "public/examples/qa_ratings",
  "is_seed": false,
  "question": "What is the deepest lake in the world?",
  "answer": "Lake Baikal in Siberia, Russia, is the deepest lake in the world.",
  "rating": 2
}
```

## Using a cloud provider

Pass a `--config-path` to override the LM engine without editing any YAML files. For OpenAI:

```bash
export OPENAI_API_KEY=your-api-key

python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --config-path ./configs/public/examples/openai_qa.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

See [Changing the Language Model Engine](tutorials/changing_lm_engine.md) for WatsonX, Anthropic, and other providers.

## CLI reference

```bash
python -m fms_dgt.public --help
```

### Flags

| Flag                        | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `--task-paths`              | Path(s) to task YAML files                                     |
| `--config-path`             | Override LM engine or model without editing the builder YAML   |
| `--num-outputs-to-generate` | Number of synthetic examples to produce per generation task    |
| `--restart`                 | Discard previous output and start fresh                        |
| `--output-dir`              | Directory to write generated data (overrides `DGT_OUTPUT_DIR`) |
| `--data-dir`                | Directory to load input data from (overrides `DGT_DATA_DIR`)   |
| `--include-namespaces`      | Additional databuilder namespaces to load (e.g. `public`)      |

### Environment variables

| Variable                          | Default      | Description                                                                                                       |
| --------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------- |
| `DGT_DATA_DIR`                    | `data/`      | Root directory for input data files referenced by `${DGT_DATA_DIR}` in task YAMLs                                 |
| `DGT_OUTPUT_DIR`                  | `output/`    | Root directory for all generated output, logs, and task cards                                                     |
| `DGT_CACHE_DIR`                   | `.cache/`    | Root directory for enrichment cache files (tool schemas, embeddings, neighbor graphs). See [Tool Enrichment Cache](#tool-enrichment-cache) |
| `DGT_TELEMETRY_DIR`               | `telemetry/` | Directory for `events.jsonl` and `traces.jsonl` telemetry files                                                   |
| `DGT_TELEMETRY_DISABLE`           | _(unset)_    | Set to any non-empty value to disable telemetry file writing entirely                                             |
| `DGT_TELEMETRY_RECORD_PAYLOADS`   | _(unset)_    | Set to `1` to include prompts and completions in telemetry spans (see [Observability](concepts/observability.md)) |
| `DGT_TELEMETRY_PAYLOAD_MAX_CHARS` | `4096`       | Maximum characters per payload field when payload recording is enabled                                            |

## Tool Enrichment Cache

When a task declares tool enrichments (`output_parameters`, `embeddings`, `neighbors`), DiGiT caches the results so subsequent runs do not re-pay the cost of LLM calls or embedding passes.

Cache files are written under:

```
{DGT_CACHE_DIR}/enrichments/{type}/{fingerprint}.json
```

The fingerprint is a SHA-256 hash of the tool set and enrichment config (model ID, `keep_k`, etc.). Two tasks that use the same tools and the same enrichment config will hit the same cache file automatically — no explicit sharing configuration is needed. REST and MCP tools are cached identically to file-backed tools; the cache is keyed by qualified tool name, not by source URL or file path.

**Delta-merge:** if you add new tools to a registry, only the new tools are computed and appended. Existing cache entries are not discarded.

**Force refresh:** set `force: true` on an enrichment in the task YAML to bypass the cache load and recompute from scratch (useful when tool descriptions have changed without qualified names changing):

```yaml
enrichments:
  - type: output_parameters
    force: true
    lm_config:
      type: ollama
      model_id_or_path: granite3.3:8b
```

Override the cache root by setting `DGT_CACHE` in your environment or `.env` file.

## Next steps

- Read the [Architecture](concepts/architecture.md) overview to understand Tasks, Databuilders, and Blocks.
- Follow [Building a Generation Databuilder](tutorials/generate_data.md) to build your own databuilder from scratch.
- Browse the [Examples](examples/simple_reasoning.md) to see what is included out of the box.
