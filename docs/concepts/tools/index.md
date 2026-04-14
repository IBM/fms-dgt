# Tools

Generating tool-calling training data requires answering three questions for every scenario:

- **What tools exist?** Tool definitions must come from somewhere, be organized, and be available at generation time.
- **Which tools should appear in this scenario?** Random selection produces coverage, but targeted selection produces richer training signal.
- **What happens when the model calls a tool?** The framework needs to simulate or execute the call and return a result the assistant stage can use.

These three concerns map to three component families with a clear dependency structure:

```
ToolLoader ──► ToolRegistry ◄── ToolEnrichment
                    │
              ┌─────┴──────┐
              ▼            ▼
         ToolSampler   ToolEngine
```

`ToolRegistry` is the central store. Loaders populate it; enrichments augment it; samplers and engines consume it. Neither sampler nor engine modifies the registry after construction.

## Components at a glance

| Component | Role | When you need it |
|---|---|---|
| [ToolLoader](registry.md#loaders) | Reads tool definitions from a source (file, MCP, REST) | Always |
| [ToolRegistry](registry.md) | Stores, validates, and exposes tool definitions | Always |
| [ToolEnrichment](enrichments.md) | Augments tools with output schemas, embeddings, or dataflow edges | When using topology-aware or embedding-based samplers |
| [ToolSampler](samplers.md) | Selects a subset of tools per scenario | In every generation stage that builds tool-calling scenarios |
| [ToolEngine](engines.md) | Executes or simulates tool calls at runtime | In every generation stage that produces tool call/result pairs |

## YAML integration

The `tools:` block is a first-class field on any task, at the same level as `datastore:` and `formatter:`. Its three sub-keys map directly to the components above:

```yaml
tools:
  registry:                         # required — one or more loader entries
    - type: file
      path: ${DGT_DATA_PATH}/weather_tools.yaml
      namespace: weather_api
  enrichments:                      # optional — omit if not needed
    - type: output_parameters
      lm_config:
        type: ollama
        model_id_or_path: granite3.3:8b
    - type: dataflow
      model: sentence-transformers/all-mpnet-base-v2
  engines:                          # optional — omit for registry-only tasks
    lm_sim:
      type: lm
      lm_config:
        type: ollama
        model_id_or_path: granite3.3:8b
        temperature: 0.0
        max_new_tokens: 512
```

At runtime, `Task.__init__` builds the registry from the loader entries, runs enrichments in dependency order, and constructs the engine. The resulting `task.tool_registry` and `task.tool_engine` are ready for stages to consume.

## Reading path

| I want to... | Go to |
|---|---|
| Define tools and load them from files or external servers | [Registry and Loaders](registry.md) |
| Understand enrichments and when to enable them | [Enrichments](enrichments.md) |
| Choose a sampling strategy for my recipe | [Samplers](samplers.md) |
| Understand how tool calls are executed or simulated | [Engines](engines.md) |
