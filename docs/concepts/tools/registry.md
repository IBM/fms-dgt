# Registry and Loaders

`ToolRegistry` is the central store for tool definitions. It is the point of ingestion for all tool sources and the shared dependency of samplers and engines.

## Tool data model

Three dataclasses form the wire format for everything that flows through the subsystem.

**`Tool`** — a definition loaded from any source:

```python
@dataclass
class Tool:
    name: str                 # unqualified name within the namespace
    namespace: str            # required; set exclusively by the loader
    description: str = ""
    parameters: dict = ...    # JSON Schema object for inputs
    output_parameters: dict = ...  # JSON Schema object for outputs; {} if absent
    metadata: dict = ...
```

**`ToolCall`** — a call issued by the assistant:

```python
@dataclass
class ToolCall:
    name: str          # always a qualified name: namespace::tool_name
    arguments: dict = ...
    call_id: str | None = None   # correlation ID, echoed in ToolResult
```

**`ToolResult`** — the result returned by an engine:

```python
@dataclass
class ToolResult:
    call_id: str | None   # matches ToolCall.call_id; matched by position if None
    name: str             # qualified tool name, copied from the originating ToolCall
    result: Any = None
    error: str | None = None
    metadata: dict = ...  # engine-specific side-channel data
```

### Qualified names

Every tool carries a fully qualified name of the form `namespace::tool_name`. This is a uniform contract with no special cases.

- Single-namespace setup: `weather_api::get_weather`
- Multi-server setup: `server_a::search`, `server_b::search`
- The sampler always emits qualified names. Prompts always show qualified names. `ToolCall.name` is always `namespace::tool_name`.
- The engine always routes by splitting on `::`. No conditional logic.

The separator `::` follows the convention established in C++, Rust, and Go module paths. Single `:` appears in URLs; `/` appears in file paths and tool names. The constant is `TOOL_NAMESPACE_SEP = "::"` in `fms_dgt/core/tools/constants.py`.

Within a namespace, two tools may share the same `tool_name` if their input schemas differ (overloading). The engine resolves by schema matching at dispatch time. Duplicate tools with identical schemas are a hard error caught at registry construction time.

## ToolRegistry

`ToolRegistry` enforces invariants immediately so no downstream component ever sees an inconsistent state. If construction succeeds, the run can proceed.

**Invariants:**
- `(namespace, tool_name, input_schema_fingerprint)` must be globally unique. Identical schema, same name, same namespace raises: `"duplicate tool: weather_api::get_weather with identical input schema registered twice"`.
- Same name, different input schema, same namespace: valid overload.
- Same name across different namespaces: always allowed.

**Construction paths:**

```python
# Single file — loader assigns namespace
registry = ToolRegistry.from_file("tools.yaml", namespace="weather_api")

# Multiple sources
registry = ToolRegistry.from_loaders([
    FileToolLoader("weather.yaml", namespace="weather_api"),
    FileToolLoader("hr.yaml", namespace="hr_api"),
])

# Direct construction — caller owns namespace on each Tool
registry = ToolRegistry([
    Tool(name="search", namespace="weather_api", ...),
])
```

**`refresh()`** re-runs all loaders then re-runs all enrichments in dependency order. Useful when tool definitions change mid-run. Stateful loaders (e.g. `MCPToolLoader`) keep their connection open between `load()` calls.

**`artifacts`** is a side-channel dict populated by enrichments and consumed by samplers:

```python
registry.artifacts["embeddings"]  # set by EmbeddingsEnrichment
registry.artifacts["dataflow"]    # set by DataflowEnrichment
```

New enrichment types add new keys without any changes to `ToolRegistry`.

## Loaders

Loaders handle I/O. `ToolRegistry` imports nothing from `mcp`, `grpc`, or `openapi` — those dependencies live entirely in loader implementations.

| Registered name | Source |
|---|---|
| `file` | YAML or JSON file |
| `mcp` | MCP server via `tools/list` (SSE transport) |
| `rest` | OpenAPI 3.x / Swagger 2.0 spec (local file or URL) |

Loaders are registered with `@register_tool_loader("name")`, following the same decorator pattern as blocks, datastores, and stages.

### File format

`FileToolLoader` accepts three shapes.

**Shape 1** — single-key dict mapping namespace to a list of tool dicts. The key becomes the namespace:

```yaml
weather_api:
  - name: get_weather
    description: Get current weather for a location.
    parameters:
      type: object
      properties:
        location: {type: string}
      required: [location]
```

**Shape 2** — bare list; namespace comes from the loader constructor argument or defaults to `"default"`:

```yaml
- name: get_weather
  description: Get current weather for a location.
  parameters:
    type: object
    properties:
      location: {type: string}
    required: [location]
```

**Shape 3** — dict mapping tool name to tool def (common format used in public benchmark datasets):

```yaml
get_weather:
  name: get_weather
  description: Get current weather for a location.
  parameters:
    type: object
    properties:
      location: {type: string}
    required: [location]
```

**Namespace precedence:** a `namespace` key on an individual tool dict takes highest precedence, followed by the `namespace` constructor argument, followed by any file-level namespace key.

### YAML configuration

Each entry in `tools.registry:` specifies a loader type and its arguments:

```yaml
tools:
  registry:
    - type: file
      path: ${DGT_DATA_DIR}/weather_tools.yaml
      namespace: weather_api
    - type: mcp
      url: http://localhost:8080
      namespace: hr_api
    - type: rest
      spec: https://petstore3.swagger.io/api/v3/openapi.json
      namespace: petstore
```

Each entry may carry an optional `engine:` key referencing a name in `tools.engines:`. This enables per-namespace engine routing when different tool sources require different execution backends. See [Engines](engines.md#multi-engine) for details.
