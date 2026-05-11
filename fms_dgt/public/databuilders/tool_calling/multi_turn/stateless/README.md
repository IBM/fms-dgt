# Multi-Turn Stateless Tool Calling Data Generation

**[Task Specification](#task-specification) | [Generators and post-processors](#generators-and-post-processors) | [Usage](#usage) | [Output Format](#output-format) | [Contributors](#contributors)**

Data builder for generating synthetic multi-turn tool calling conversations using a multi-stage approach. This builder creates realistic interactions between users and AI assistants that involve planning and executing tool calls across multiple conversation turns.

## Task specification

This data builder supports generation with the following parameters:

### Required

- `task_name`: Unique identifier for the task
- `data_builder`: Must be set to `tool_calling/multi_turn/stateless`
- `formatter`: Formatter specification for output format
  - `type: tool_calling/formatters/multi_turn` - Formats complete conversation exchanges
  - `type: tool_calling/formatters/single_turn` - Formats planning-focused single turns

### Tool Settings

- `tools.registry`: List of tool registries to load
  - `type`: Registry type (e.g., `file`)
  - `path`: Path to tool definitions YAML file
  - `namespace`: Namespace for organizing tools
  - `engine`: Tool execution engine (e.g., `lm_simulator`)
- `tools.engines`: Configuration for tool execution engines
- `tools.enrichments`: Optional tool enrichment configurations (e.g., output parameters, embeddings)

### Multi-Turn Specific Parameters

- `max_turns`: Maximum number of conversation turns (default: varies by task)
- `min_turns`: Minimum number of conversation turns (default: 1)
- `has_nested`: Whether to allow nested tool calls (default: false)
- `separate_context`: Whether to keep tool contexts separate (default: true)
- `min_plan_length`: Minimum number of tools in a plan (default: 1)
- `max_plan_length`: Maximum number of tools in a plan (default: 8)

### Stage Configuration

- `initialization_stages`: Stages run at the start of generation
  - `sample/persona`: Generates user personas from a persona store
  - `tool_calling/multi_turn/stages/scenario_generator`: Creates initial scenarios
- `iteration_stages`: Stages run for each conversation turn
  - `tool_calling/multi_turn/stages/planner`: Plans tool calls
  - `tool_calling/multi_turn/stages/user`: Generates user messages
  - `tool_calling/multi_turn/stages/verifier`: Verifies tool call validity
  - `tool_calling/multi_turn/stages/execute`: Executes planned tools
  - `tool_calling/multi_turn/stages/summarize`: Summarizes assistant responses
- `termination_stages`: Stages run after conversation completes
  - `tool_calling/multi_turn/stages/filter`: Filters out low-quality conversations
  - `tool_calling/multi_turn/stages/error_hint`: Adds error handling hints

An example task configuration can be found [here](../../../../../tasks/public/tool_calling/multi_turn/stateless/toolmind/conversation.yaml).

## Generators and post-processors

Default configuration for generators used by the data builder is available [here](./multi_turn_tool_calling.yaml).

### Generators

- Configurable LLM-based generators via `ollama` or other supported backends
- Example: `gpt-oss:20b` via `ollama` with sampling parameters

### Tool Execution Engines

- `lm_simulator`: LLM-based tool simulation engine that generates synthetic tool outputs

### Post-processors

- Tool call verification and validation
- Conversation filtering based on quality criteria
- Error hint generation for failed tool calls

## Usage

To try out the databuilder, run the following command:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/tool_calling/multi_turn/stateless/toolmind/conversation.yaml \
  --num-outputs-to-generate 10 \
  --restart
```

This launches a data generation job by passing task configuration using the YAML file specified via `--task-paths` argument.

By default, the output (\*.jsonl) is generated in sub-directories under `output/tool_calling/multi_turn/stateless/`. Here's a sample output:

## Output Format

```json
{
  "task_name": "tool_calling/multi_turn/stateless/toolmind",
  "is_seed": false,
  "conversation_id": "ac368f35-9245-4f7c-b5fc-366de54cf327",
  "steps": [
    {
      "role": "persona",
      "content": "Role: A competitive shooter from Brazil.\nPersonality: conventional and resistant to novelty, spontaneous and disorganized, skeptical and challenging, calm and confident.",
      "stage_name": "sample/persona",
      "persona_role": "A competitive shooter from Brazil",
      "personality": {
        "openness": -0.9,
        "conscientiousness": -0.4,
        "extraversion": -0.1,
        "agreeableness": -0.8,
        "neuroticism": -0.9
      }
    },
    {
      "role": "tc/sampler",
      "tools": [
        {
          "name": "get_order_details",
          "namespace": "toolmind_apigen",
          "description": "Get the status and details of an order.",
          "parameters": {
            "type": "object",
            "properties": {
              "order_id": {
                "type": "string",
                "description": "The order id, such as '#W0000000'."
              }
            },
            "required": ["order_id"]
          }
        }
      ]
    },
    {
      "role": "tc/plan",
      "plan": [
        {
          "name": "get_order_details",
          "namespace": "toolmind_apigen",
          "qualified_name": "toolmind_apigen::get_order_details",
          "arguments": {
            "order_id": "#W1234567"
          },
          "call_id": "$1"
        }
      ]
    },
    {
      "role": "tc/user",
      "content": "Give me the last 100 mass messages...",
      "required_terms": ["America/Sao_Paulo", "#W1234567"],
      "hidden_set": [
        {
          "name": "get_order_details",
          "arguments": { "order_id": "#W1234567" },
          "id": "$1"
        }
      ]
    },
    {
      "role": "tool_call",
      "content": {
        "name": "get_order_details",
        "arguments": { "order_id": "#W1234567" },
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
      "role": "assistant",
      "content": "I retrieved the details of order #W1234567..."
    }
  ]
}
```

A complete example can be found [here](../../../../../output/tool_calling/multi_turn/stateless/toolmind/example_output.jsonl).

## Key Features

- **Persona-driven Generation**: Uses diverse user personas to create realistic conversation patterns
- **Multi-turn Planning**: Generates complex multi-step tool calling plans
- **Nested Tool Calls**: Supports tool calls that depend on outputs from previous tools
- **Tool Namespace Management**: Organizes tools from multiple sources into namespaces
- **LM-based Tool Simulation**: Simulates tool execution using language models
- **Quality Filtering**: Automatically filters low-quality or invalid conversations
- **Flexible Formatting**: Supports both multi-turn and single-turn output formats

## Architecture

The data builder follows a stage-based architecture:

1. **Initialization**: Sample personas and generate initial scenarios
2. **Iteration**: For each turn, plan tool calls, generate user messages, verify, execute, and summarize
3. **Termination**: Filter results and add error hints where applicable

See the [generate.py](./generate.py) file for the main entry point and [task.py](./task.py) for task configuration details.

## Contributors

**Author and Maintainer**: IBM Research

## Citation

If you make use of this work, please cite it as

```
@article{crouse2026simulating,
  title={Simulating Complex Multi-Turn Tool Calling Interactions in Stateless Execution Environments},
  author={Crouse, Maxwell and Abdelaziz, Ibrahim and Fadnis, Kshitij and Patel, Siva Sankalp and Basu, Kinjal and Gunasekara, Chulaka and Kumaravel, Sadhana and Munawar, Asim and Kapanipathi, Pavan},
  journal={arXiv preprint arXiv:2601.19914},
  year={2026},
  url={https://arxiv.org/abs/2601.19914}
}
```
