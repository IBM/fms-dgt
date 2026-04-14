# Engines

A `ToolEngine` answers: given a tool call issued by the assistant, what is the result? It handles execution, simulation, session state, and routing. It is separate from `ToolRegistry` because tool definition lookup and tool call execution have different consumers, different state lifetimes, and different deployment requirements.

## Session interface

```python
class ToolEngine(ABC):
    def setup(self, session_id: str, *args, **kwargs) -> None
    def execute(self, session_id: str, tool_calls: list[ToolCall]) -> list[ToolResult]
    def simulate(self, session_id: str, tool_calls: list[ToolCall]) -> list[ToolResult]
    def teardown(self, session_id: str) -> None
    def get_session_state(self, session_id: str) -> dict | None
```

`session_id` is an opaque string. The engine has no knowledge of conversations, databuilders, or pipelines. The caller decides what to use as a session ID — typically a conversation ID for generation stages.

`setup` on an already-registered `session_id` raises `ValueError`. The caller that creates a session owns teardown. The pattern is to call `setup` before stages run and `teardown` in a `finally` block when the conversation completes.

## `execute` vs. `simulate`

These two methods differ in how they handle session state:

- **`execute`** — processes tool calls and permanently appends each `(tool_call, tool_output)` pair to session history. Within the batch, earlier results are visible to later calls.
- **`simulate`** — processes tool calls but rolls back all session state mutations before returning. The session state after `simulate` returns is identical to before the call.

`simulate` is useful when a stage wants to probe multiple candidate tool calls against the current session state and pick the most informative one before committing via `execute`. The base class `simulate` delegates to `execute` as a convenience for stateless engines. Stateful engines (e.g. `LMToolEngine`) override `simulate` to use a transaction with rollback.

## Engine types

| Registered name | Behavior |
|---|---|
| `lm` | LLM simulates a plausible result from the tool schema and call |
| `mcp` | Real execution via MCP tool server (SSE transport) |
| `rest` | Real HTTP call; path/query/body routing; auth and TLS |
| `multi` | Routes by namespace across multiple engine instances |

### `lm` engine

`LMToolEngine` simulates tool results using an LLM. It is the default engine for recipe development and for tool sets where real infrastructure is not available.

The engine builds a dynamic conversation where prior tool calls from session history serve as few-shot context. There are no hardcoded ICL examples.

```
[system]    static instruction: simulate a realistic, successful result conforming to output_parameters
[user]      tool spec + historical call 1      ─┐ repeated for each entry
[assistant] result of historical call 1        ─┘ in session history
...
[user]      tool spec + current call to simulate
```

If `tool.output_parameters` is present, the LLM is called with `response_format: json_schema` and the output is validated against the declared schema. If absent, the LLM is called with `response_format: json_object` and the raw parsed dict is returned.

Failed results (parse failure, schema violation) are still appended to session history on `execute`. The assistant must learn to handle tool errors, not just successes.

**Error simulation:** `LMToolEngine` supports probabilistic error injection to produce training data diversity. Error categories are declared at the engine level:

```yaml
engines:
  lm_sim:
    type: lm
    lm_config:
      type: ollama
      model_id_or_path: granite3.3:8b
    error_categories:
      - type: network_error
        probability: 0.05
        message: "Connection timed out"
      - type: unparseable_result
        probability: 0.05
      - type: schema_violation
        probability: 0.05
```

| Error type | Result |
|---|---|
| `network_error` | `ToolResult(error=message)` immediately; no LLM call |
| `unparseable_result` | `ToolResult(result="<garbled: ...>")` — malformed but non-empty |
| `schema_violation` | `ToolResult(error=message)` — treated as a generic error |

All categories are sampled independently per call. If multiple fire, one is chosen at random.

### `multi` engine

`MultiToolEngine` holds a `dict[namespace -> ToolEngine]` and routes by parsing the namespace from `ToolCall.name`. Since names are always qualified with `::`, routing splits on `::` and dispatches to the corresponding engine. No conditional logic.

**Per-namespace routing** is configured by adding an `engine:` key to individual registry entries and defining matching engine names in `tools.engines:`:

```yaml
tools:
  registry:
    - type: file
      path: weather_tools.yaml
      namespace: weather_api
      engine: lm_sim          # simulated execution for file-loaded tools
    - type: mcp
      url: http://localhost:8080
      namespace: hr_api
      engine: mcp_live        # real MCP execution for live tools
  engines:
    lm_sim:
      type: lm
      lm_config: {type: ollama, model_id_or_path: granite3.3:8b}
    mcp_live:
      type: mcp
      url: http://localhost:8080
```

**Concurrency note:** conversations run concurrently inside the thread pool. For stateless backends (`lm`, read-only REST), this is safe. For stateful backends (REST APIs that write records, MCP servers with session state), concurrent calls from different conversations can produce race conditions. Recipe authors are responsible for handling this — use a stateless API, add conversation-scoped isolation at the API layer, or set `max_concurrent_conversations: 1` on the databuilder.

## Runtime usage

Once a `Task` is initialized, `task.tool_engine` is ready. The typical pattern in a generation stage:

```python
session_id = str(uuid.uuid4())
task.tool_engine.setup(session_id)

try:
    results = task.tool_engine.execute(session_id, tool_calls)  # list[ToolResult]
finally:
    task.tool_engine.teardown(session_id)
```

`teardown` is safe to call on an already-removed session, so the `finally` block is always safe to include.
