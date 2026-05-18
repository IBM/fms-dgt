# Chit-Chat Conversations

This example generates multi-turn chit-chat conversations using the `core/conversation` databuilder. It demonstrates DiGiT's stage pipeline: a sequence of initialization and iteration stages that produce fully synthetic, persona-driven conversations ready for SFT fine-tuning.

The full task configuration is at [`tasks/public/examples/chit_chat/task.yaml`](https://github.com/IBM/fms-dgt/tree/main/tasks/public/examples/chit_chat/task.yaml).

## Run it

```bash
ollama pull granite4:3b

python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/chit_chat/task.yaml \
  --num-outputs-to-generate 10 \
  --restart
```

Two output files are written under `output/public/examples/chit_chat/`:

- `final_data.jsonl`: full internal representation with all pipeline steps
- `formatted_output.jsonl`: clean `messages` format ready for SFT fine-tuning

## The stage pipeline

The `core/conversation` databuilder runs each conversation through two phases.

### Initialization

Initialization stages run once per conversation before any turns are generated. This example uses two:

**`lm/scenario`** generates a scenario that sets the context for the conversation. It uses in-context learning from seed examples to produce a short description like: "A curious student wants to learn about space exploration."

**`sample/persona`** samples a persona from `data/public/examples/chit_chat/personas.jsonl` and assigns it to the user: role, expertise level, domain, goals, and personality traits. The persona is injected into the user stage each turn so the simulated user stays consistent throughout.

To use custom personas, replace or extend `personas.jsonl` with your own records, or point `data_path` at a different file:

```yaml
- name: sample/persona
  persona_store:
    type: default
    data_path: /path/to/your/personas.jsonl
```

Each record must have at minimum a `role` field. See `data/public/examples/chit_chat/personas.jsonl` for the full schema.

### Iteration

Iteration stages run in a loop, one turn at a time, until the conversation ends.

**`lm/flow_controller`** runs first each turn. It reads the conversation history and selects a pattern that determines what the user should do next. Patterns are defined in the task YAML with a name, description, and a hint passed to the user stage:

```yaml
patterns:
  - name: chit_chat/follow_up
    description: "User follows up on the previous assistant turn with a related question or comment."
    hint: "Generate a user message that naturally follows up on what was just discussed."
    weight: 0.35
```

When the flow controller selects a termination pattern, it signals the pipeline to end the conversation. The conversation is only kept if it has reached `min_turns`.

**`lm/user/guided`** generates the user's next message. It receives the scenario, the persona, the conversation history, and the hint from the flow controller.

**`lm/assistant/naive`** generates the assistant's response given the scenario and conversation history.

## Turn bounds

```yaml
max_turns: 4
min_turns: 2
```

`max_turns` caps how long a conversation can run. `min_turns` sets the minimum number of turns a conversation must complete before it is accepted as output. Conversations terminated by the flow controller or dropped by a stage before `min_turns` are silently discarded.

## Patterns

Patterns define the space of conversational moves available to the flow controller. Each pattern has:

- `name`: a unique identifier used in logs and seed data
- `description`: what this move represents
- `hint`: the instruction passed to the user stage
- `weight`: relative sampling probability

The flow controller selects a pattern each turn (weighted by `weight`), then generates the user's instruction. Adding, removing, or reweighting patterns changes the character of the generated conversations without touching any code.

## Output format

The formatter strips internal pipeline steps and writes only user and assistant turns:

```json
{
  "conversation_id": "a3f2c1d8",
  "messages": [
    {"role": "user", "content": "I was reading about the James Webb Space Telescope. What's the most exciting discovery so far?"},
    {"role": "assistant", "content": "One of the most exciting findings is the first clear detection of carbon dioxide in an exoplanet atmosphere..."},
    {"role": "user", "content": "Could Webb actually find signs of life on another planet?"},
    {"role": "assistant", "content": "It's possible. Webb can analyze atmospheric chemistry by measuring which wavelengths of starlight are absorbed..."}
  ]
}
```

The `conversation_id` matches the corresponding record in `final_data.jsonl` for cross-referencing.

## Data files

Two data files drive this example:

- **`data/public/examples/chit_chat/seed_examples.jsonl`**: complete conversations used as in-context learning examples by the scenario stage. Each seed includes a scenario, persona, flow controller decisions, and user and assistant turns.
- **`data/public/examples/chit_chat/personas.jsonl`**: persona definitions sampled by the `sample/persona` stage. Replace or extend this file to change the user personas generated.

## Next steps

- Swap the LM engine: see [Changing the Language Model Engine](../tutorials/changing_lm_engine.md).
- To build a custom stage, see the `core/conversation` databuilder source at `fms_dgt/core/databuilders/conversation/`.
