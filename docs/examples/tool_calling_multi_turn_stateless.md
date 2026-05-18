# Multi-Turn Stateless Tool Calling

This example demonstrates how to generate synthetic multi-turn tool calling conversations using the `tool_calling/multi_turn/stateless` databuilder. This builder creates realistic interactions between users and AI assistants that involve planning and executing tool calls across multiple conversation turns.

The task configuration is at [`tasks/public/tool_calling/multi_turn/stateless/toolmind/conversation.yaml`](https://github.com/IBM/fms-dgt/tree/main/tasks/public/tool_calling/multi_turn/stateless/toolmind/).

## What is the Multi-Turn Stateless Tool Calling Databuilder?

The `tool_calling/multi_turn/stateless` databuilder generates synthetic tool calling conversations using a stage-based pipeline. Key features include:

- **Persona-driven Generation**: Uses diverse user personas to create realistic conversation patterns
- **Multi-turn Planning**: Generates complex multi-step tool calling plans with dependencies
- **Nested Tool Calls**: Supports tool calls where outputs feed into subsequent calls
- **Tool Namespace Management**: Organizes tools from multiple sources into namespaces
- **LM-based Tool Simulation**: Simulates tool execution using language models
- **Quality Filtering**: Automatically filters low-quality or invalid conversations

The databuilder is registered as `tool_calling/multi_turn/stateless`.

## Prerequisites

Set up a language model backend. This example uses OpenAI-compatible API endpoints:

```bash
# If using vLLM or similar OpenAI-compatible server
# Start your model server on http://localhost:8000
```

## Run it

Generate 10 multi-turn tool calling conversations:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/tool_calling/multi_turn/stateless/toolmind/conversation.yaml \
  --num-outputs-to-generate 10 \
  --restart
```

Output is written under `output/tool_calling/multi_turn/stateless/toolmind/`:

- `final_data.jsonl`: full internal representation with all pipeline steps, tool plans, and execution results
- `formatted_output.jsonl`: clean `messages` format with tools attached, ready for SFT fine-tuning

## The stage pipeline

The `tool_calling/multi_turn/stateless` databuilder runs each conversation through two phases.

### Initialization

Initialization stages run once per conversation before any turns are generated.

**`sample/persona`** assigns a user persona sampled from a persona store. The persona shapes the simulated user's voice, goals, expertise level, and personality throughout the conversation. Personalities are modeled using the Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism).

**`tool_calling/multi_turn/stages/scenario_generator`** samples tools from the tool registry using configurable samplers (e.g., neighbor-based sampling using embeddings) and generates an initial scenario that describes a realistic task requiring those tools.

### Iteration

Iteration stages run in a loop, one turn at a time.

**`tool_calling/multi_turn/stages/planner`** generates a plan consisting of one or more tool calls. The plan length is constrained by `min_plan_length` and `max_plan_length` parameters. When `has_nested: true`, the planner can create tool calls that reference outputs from previous tools using variable syntax (e.g., `$1.created_at`).

**`tool_calling/multi_turn/stages/user`** generates the user's next message based on the persona, scenario, and planned tool calls. The user message implicitly requires the tools in the plan without explicitly naming them, creating a realistic information need.

**`tool_calling/multi_turn/stages/verifier`** validates that the planned tool calls are valid according to the tool definitions and that nested references resolve correctly.

**`tool_calling/multi_turn/stages/execute`** executes the planned tools using the configured tool engine. The `lm_simulator` engine uses a language model to generate synthetic but realistic tool outputs based on the tool definitions and input parameters.

**`tool_calling/multi_turn/stages/summarize`** generates the assistant's response that summarizes the tool execution results in natural language.

The pipeline continues iterating until either:
- The conversation reaches `max_turns`
- A natural stopping condition is met (based on the scenario completion)
- The conversation satisfies `min_turns` and reaches a logical conclusion

### Termination

**`tool_calling/multi_turn/stages/filter`** filters out conversations that don't meet quality criteria (e.g., failed tool calls, invalid plans, incomplete scenarios).

**`tool_calling/multi_turn/stages/error_hint`** adds error hints to conversations where tool calls failed, providing learning signals for error recovery.

## Turn bounds

```yaml
max_turns: 3
min_turns: 1
```

Conversations shorter than `min_turns` are discarded. Conversations that reach `max_turns` are ended and kept if they satisfy `min_turns`.

## Tool configuration

The task YAML specifies tools through a registry system:

```yaml
tools:
  registry:
    - type: file
      path: ${DGT_DATA_DIR}/public/tool_calling/toolmind/APIGen-MT-5k-query_tools.yaml
      namespace: toolmind_apigen
      engine: lm_simulator
  engines:
    lm_simulator:
      type: lm
      lm_config:
        type: openai
        model_id_or_path: openai/gpt-oss-20b
        base_url: http://localhost:8000/v1/
        n: 8
        temperature: 1.0
        max_new_tokens: 1024
```

Multiple tool registries can be specified, each with its own namespace. This allows organizing tools from different sources (e.g., `toolmind_apigen`, `toolmind_glaive`, `toolmind_button`) while avoiding name collisions.

## Tool enrichments

The pipeline supports optional tool enrichments:

```yaml
enrichments:
  - type: output_parameters
    lm_config:
      type: openai
      model_id_or_path: openai/gpt-oss-20b
  - type: embeddings
    model: sentence-transformers/all-mpnet-base-v2
  - type: neighbors
    model: sentence-transformers/all-mpnet-base-v2
```

- **output_parameters**: Automatically generates output parameter schemas for tools using an LM
- **embeddings**: Computes embeddings for tools to enable semantic search
- **neighbors**: Enables neighbor-based tool sampling using embedding similarity

## Nested tool calls

When `has_nested: true`, the planner can create tool calls that depend on outputs from previous tools:

```json
{
  "plan": [
    {
      "name": "get_order_details",
      "arguments": {"order_id": "#W1234567"},
      "call_id": "$1"
    },
    {
      "name": "mass_messages",
      "arguments": {
        "timezone": "America/Sao_Paulo",
        "signstart": "$1.created_at",
        "signend": "1993-11-21T23:59:59Z"
      },
      "call_id": "$2"
    }
  ]
}
```

Here, the second tool call references `$1.created_at` from the first tool's output.

## Context separation

When `separate_context: true` (default), implicit tool calls are treated as hidden tool calls. This means a subset of tool calls will be instructed to be hidden within the user request.

## Output format

The formatter writes each conversation as a `messages` array with tools attached:

```json
{
  "task_name": "tool_calling/multi_turn/stateless/toolmind",
  "conversation_id": "ac368f35-9245-4f7c-b5fc-366de54cf327",
  "messages": [
    {
      "role": "user",
      "content": "Give me the last 100 mass messages. Use the timezone America/Sao_Paulo..."
    },
    {
      "role": "tool_call",
      "content": {
        "name": "get_order_details",
        "arguments": {"order_id": "#W1234567"},
        "id": "$1"
      }
    },
    {
      "role": "tool_result",
      "content": {
        "id": "$1",
        "name": "get_order_details",
        "result": {
          "order_id": "#W1234567",
          "status": "shipped",
          "created_at": "2024-03-15T10:23:45Z"
        }
      }
    },
    {
      "role": "tool_call",
      "content": {
        "name": "mass_messages",
        "arguments": {
          "timezone": "America/Sao_Paulo",
          "signstart": "2024-03-15T10:23:45Z"
        },
        "id": "$2"
      }
    },
    {
      "role": "tool_result",
      "content": {
        "id": "$2",
        "name": "mass_messages",
        "result": {"messages": [...], "total": 5}
      }
    },
    {
      "role": "assistant",
      "content": "I retrieved the details of order #W1234567, which was shipped on 2024-03-15..."
    }
  ],
  "tools": [
    {
      "name": "get_order_details",
      "namespace": "toolmind_apigen",
      "description": "Get the status and details of an order.",
      "parameters": {...}
    }
  ]
}
```

The `conversation_id` matches the corresponding record in `final_data.jsonl`. The `tools` field contains all tool definitions available in the conversation.

## Formatters

Two formatters are available:

### Multi-turn formatter (`tool_calling/formatters/multi_turn`)

Formats complete conversation exchanges including user messages, tool calls, tool results, and assistant responses. Suitable for training models on full tool calling conversations.

### Single-turn formatter (`tool_calling/formatters/single_turn`)

Formats planning-focused single turns that include user messages and corresponding tool call plans (without execution). Each turn is an independent user→plan pair. Suitable for training models specifically on tool planning tasks.

## Data files

- **Tool registries**: Tool definitions are in YAML files under `data/public/tool_calling/toolmind/`
- **Persona store**: `data/public/tool_calling/personas.jsonl` contains user personas with roles, expertise, and personality traits
- **Seed examples**: Some stages may use seed examples for in-context learning

See [`fms_dgt/public/databuilders/tool_calling/multi_turn/stateless/README.md`](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/tool_calling/multi_turn/stateless/README.md) for complete documentation.

## Advanced configuration

### Plan length control

Control the complexity of tool calling plans:

```yaml
min_plan_length: 1
max_plan_length: 8
```

Longer plans create more complex multi-step scenarios but may have higher failure rates.

### Sampling strategy

Control how tools are sampled for each scenario:

```yaml
sampler_mix:
  - sampler: { type: tc/neighbor, k: 15, min_score: 0.0 }
    weight: 1.0
```

The `tc/neighbor` sampler selects tools based on embedding similarity to encourage semantically related tool sets.

### Generation parameters

The default generator configuration is in `fms_dgt/public/databuilders/tool_calling/multi_turn/stateless/multi_turn_tool_calling.yaml`:

```yaml
blocks:
  - name: generator
    type: openai
    model_id_or_path: openai/gpt-oss-20b
    n: 8
    temperature: 1.0
    max_tokens: 4096
```

The `n: 8` parameter generates 8 candidates per stage and selects the best one, significantly improving quality at the cost of more API calls.

## Next steps

- Swap the LM engine: see [Changing the Language Model Engine](../tutorials/changing_lm_engine.md).
- Add custom tools: create a new tool registry YAML file and reference it in your task configuration.
- Customize personas: edit `data/public/tool_calling/personas.jsonl` to add domain-specific user roles and expertise levels.
- Explore the implementation: see [generate.py](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/tool_calling/multi_turn/stateless/generate.py) for the databuilder entry point and [task.py](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/tool_calling/multi_turn/stateless/task.py) for task configuration details.
