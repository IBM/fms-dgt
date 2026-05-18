# Enrichments

Enrichments are optional post-load passes that augment tools with additional information. They run once at task initialization, after all loaders complete and before the registry is handed to samplers and engines.

## Why enrichments exist

Tool definitions loaded from static files rarely carry `output_parameters`. MCP and REST tools are more likely to have them (via MCP schemas and OpenAPI response objects), but this is not guaranteed.

Without output schemas, sampling strategies that reason about dataflow between tools cannot produce their intended training signal. Enrichments fill this gap in a controlled, cacheable, and explicitly declared way.

**Enrichments are opt-in.** The subsystem works without any enrichments using `tc/random` sampling. Enabling them unlocks higher-quality strategies. A recipe author declares exactly which enrichments run and pays exactly the cost of running them.

## Base class and registration

```python
class ToolEnrichment(ABC):
    depends_on: list[str] = []       # artifact keys that must exist before this runs
    artifact_key: str | None = None  # key written to registry.artifacts; None if enrichment only modifies Tool objects

    @abstractmethod
    def enrich(self, registry: ToolRegistry) -> None: ...
```

`enrich()` receives the full registry. It can read tool definitions, read `registry.artifacts`, modify `Tool` objects in place, and write to `registry.artifacts`. Enrichments are registered with `@register_tool_enrichment("name")`.

`depends_on` declares which artifact keys must already be present before this enrichment runs. The framework resolves execution order by topological sort. Declaration order in the YAML does not need to match dependency order. A cycle or missing dependency is a hard error at task construction time.

## Enrichment types

| Registered name | Modifies tools | Writes artifact | Depends on |
|---|---|---|---|
| `output_parameters` | Yes (adds `output_parameters`) | None | None |
| `embeddings` | No | `"embeddings"` | None |
| `dataflow` | No | `"dataflow"` | None |
| `neighbors` | No | `"neighbors"` | `"embeddings"` |

### `output_parameters`

Calls an LLM to infer an output schema for every tool that lacks `output_parameters`. Skips tools that already have them, so MCP and REST tools with native output schemas pay no LLM cost.

Requires `lm_config` using the same shape as `LMJudgeValidator` and `LMToolEngine`:

```yaml
enrichments:
  - type: output_parameters
    lm_config:
      type: ollama
      model_id_or_path: granite3.3:8b
      temperature: 0.1
      max_new_tokens: 256
```

Required by `tc/chain`, `tc/fan_out`, `tc/fan_in`, and `tc/dag` when the source tools lack native output schemas.

### `embeddings`

Embeds each tool (name, description, input parameters, output parameters if present) using a sentence-transformer model. Stores the result in `registry.artifacts["embeddings"]` as `dict[qualified_tool_name, embedding_vector]`. No LLM call.

```yaml
enrichments:
  - type: embeddings
    model: sentence-transformers/all-mpnet-base-v2
```

Required by embedding-based sampling strategies.

### `dataflow`

Computes genuine output-to-input parameter-level edges between tool pairs. An edge A→B exists when at least one output parameter of A can serve as an input parameter of B, based on both type compatibility and semantic similarity. The edge score reflects the strength of the best matching parameter pair rather than aggregate schema similarity.

The enrichment writes `registry.artifacts["dataflow"]` with two sub-indexes:

```python
{
    "out": { src_qname: { ... } },   # forward index: for each tool, its successors
    "in":  { tgt_qname: { ... } },   # reverse index: for each tool, its predecessors
}
```

Both directions are built in one pass and cached together. `tc/chain`, `tc/fan_out`, and `tc/dag` read the forward index; `tc/fan_in` reads the reverse index.

`DataflowEnrichment` embeds per-parameter sentences independently of `EmbeddingsEnrichment`. Declaring a dependency on `"embeddings"` would impose a mandatory `EmbeddingsEnrichment` run for something `DataflowEnrichment` does not actually use, so `depends_on = []`.

```yaml
enrichments:
  - type: dataflow
    model: sentence-transformers/all-mpnet-base-v2
```

### `neighbors`

Computes output-to-input compatibility scores between tool pairs using the embedding index. Stores a weighted graph in `registry.artifacts["neighbors"]` keyed by qualified tool name. Requires the `"embeddings"` artifact to already be present.

## Caching

Enrichments are expensive: `output_parameters` calls an LLM once per unannotated tool; `embeddings` runs a sentence-transformer over every tool; `dataflow` embeds every parameter in the registry. None of these should re-run on a process restart or when a second task uses the same tool set.

**Cache location:** `{DGT_CACHE_DIR}/enrichments/{enrichment_type}/{fingerprint}.json`, where `DGT_CACHE_DIR` defaults to `.cache` under the run directory and is overridable via the `DGT_CACHE` environment variable.

**Fingerprint inputs:**

| Enrichment | Fingerprint covers |
|---|---|
| `output_parameters` | Qualified names of tools lacking `output_parameters` + LM model ID |
| `embeddings` | Qualified names of all tools + sentence-transformer model name |
| `dataflow` | Qualified names, schema fingerprints, parameter texts across all tools + model name + `max_neighbors` |

Temperature and other sampling parameters are excluded from `output_parameters` fingerprints. The output schema for a tool is a semantic fact about that tool, not a function of generation temperature. Two tasks using the same LM model on the same tools share the same cache entry regardless of temperature.

**Delta-merge:** adding new tools to a registry does not discard existing cache entries. Only new tools are computed and appended. Two tasks that declare the same enrichment type and model over the same tool set automatically share the same cache file.

**Force refresh:** set `force: true` on any enrichment to bypass the cache load and overwrite on completion. Useful when tool descriptions have changed without the qualified names changing.

```yaml
enrichments:
  - type: output_parameters
    force: true
    lm_config:
      type: ollama
      model_id_or_path: granite3.3:8b
```

## Enrichment-sampler dependency

Enrichments run once at task initialization. Samplers run per-scenario at generation time. The link between them is declared explicitly:

- Each enrichment declares `artifact_key` — the key it writes to `registry.artifacts`.
- Each sampler declares `required_artifacts` — the keys it needs from `registry.artifacts`.

If a sampler's required artifact is absent at construction time, the framework raises a clear error rather than failing silently at generation time:

```
SamplingError: tc/chain requires the 'dataflow' artifact;
add DataflowEnrichment to tools.enrichments
```

This makes the cost and dependency of each strategy explicit and debuggable at configuration time.
