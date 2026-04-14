# Samplers

A `ToolSampler` selects a subset of tools for one generated scenario. Selection strategy determines what kind of reasoning the model must learn: breadth across many tools, sequential dataflow between dependent calls, parallel dispatch, aggregation. Different strategies produce qualitatively different training signal.

## Base class

```python
class ToolSampler(ABC):
    required_artifacts: list[str] = []  # artifact keys this sampler needs

    def __init__(self, registry: ToolRegistry, **kwargs): ...

    @abstractmethod
    def sample(self, k: int | None = None, **kwargs) -> list[Tool]: ...
```

`required_artifacts` is checked at construction time against `registry.artifacts`. A missing artifact raises immediately with a message pointing to the enrichment that produces it. Samplers are safe to construct early (at stage initialization) and fail loudly before generation begins rather than silently during it.

`**kwargs` on `sample()` is the per-call override channel. Any constructor parameter can be overridden at call time. Samplers ignore kwargs they do not recognize, so callers can pass a uniform argument set across sampler types.

## Sampling strategies

| Strategy | Training signal | Required enrichment |
|---|---|---|
| `tc/random` | Breadth, coverage | None |
| `tc/chain` | Sequential dataflow: each call depends on the previous | `dataflow` |
| `tc/fan_out` | Parallel dispatch: one seed feeds N independent downstream calls | `dataflow` |
| `tc/fan_in` | Aggregation: N predecessors feed one sink tool | `dataflow` |

### `tc/random`

Baseline. Draws `k` tools uniformly at random. Supports namespace filtering and per-namespace weighting when the registry spans multiple sources.

```yaml
type: tc/random
k: 8
```

```yaml
type: tc/random
k: 4
namespace: weather_api          # restrict to one namespace
```

```yaml
type: tc/random
k: 8
strategy: proportional          # weight namespaces by their tool counts
```

Zero required enrichments. This is the starting point for any recipe. Use it to validate your pipeline before adding topology-aware strategies.

### Topology-aware samplers

The four topology-aware samplers each produce a qualitatively different call structure:

| Sampler | Topology | List ordering |
|---|---|---|
| `tc/chain` | A→B→C→D | A first; each tool depends on the previous |
| `tc/fan_out` | Seed→{B,C,D} | Seed first; remaining tools are independent successors |
| `tc/fan_in` | {A,B,C}→Sink | Sink last; all predecessors before it |
| `tc/dag` | Connected DAG subgraph | Topological order; tools at same depth may run in parallel |

Return order is meaningful. The caller knows which sampler was used and interprets list position accordingly. No separate topology descriptor is returned.

Each sampler exposes a `sampler_name` class attribute set by the `@register_tool_sampler` decorator. A stage that selects a prompt template based on topology reads `sampler.sampler_name` rather than storing that information separately.

All four read from `registry.artifacts`. `tc/chain`, `tc/fan_out`, and `tc/dag` use the forward index; `tc/fan_in` uses the reverse index. Both are produced in a single `DataflowEnrichment` pass.

Additional strategies targeting distractor robustness (`tc/sparse`), tool discrimination (`tc/conflict`), and full dependency-graph sampling (`tc/dag`) are in development.

## `SamplingError` contract

`k` is a hard requirement. If a sampler cannot return exactly `k` tools (chain dead-ends, graph is too sparse, `min_score` filters eliminate all candidates), it raises `SamplingError` rather than returning a short list silently:

```python
class SamplingError(Exception):
    requested: int        # the k that was requested
    tools: list[Tool]     # partial list collected before the dead end; may be empty
```

The caller catches `SamplingError` and decides: retry with a different seed, use the partial list in `e.tools`, or propagate. The partial list is explicitly typed and documented as potentially incomplete.

`k` means exactly `k`. A `min_k` soft lower bound was considered and rejected: it would add a second knob that interacts with `k` in non-obvious ways. The `SamplingError` model is simpler and gives the caller the partial result to make its own recovery decision.

## Concurrency rules

Stage instances are shared across all inner thread pool workers. A sampler stored on a stage is called from multiple threads simultaneously. Two rules follow:

1. **All derived state must be built in `__init__`, never lazily in `sample()`.** Lazy initialization creates a write-after-check race when two threads simultaneously find the attribute unset.
2. **`sample()` must be stateless across calls.** No instance variable may be written during `sample()`. All call-specific state lives in local variables.

These rules apply to all sampler implementations.

## YAML shape

Samplers are constructed via `get_tool_sampler(type, registry=..., **kwargs)`. The config block follows the same shape whether it lives in a task YAML, a stage config, or is constructed programmatically:

```yaml
type: tc/chain
k: 4
min_score: 0.7
```

```yaml
type: tc/fan_out
k: 4
```

```yaml
type: tc/random
k: 8
namespace: weather_api
```

All constructor parameters accept call-site overrides via `sample(**kwargs)`. A stage that wants to vary `k` per scenario does not need to rebuild the sampler.
