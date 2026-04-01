# Chit-Chat Conversations

Generates multi-turn chit-chat conversations between a user and an AI assistant. Each conversation is fully synthetic: DiGiT first generates a scenario and assigns the user a persona, then runs an iterative loop where a flow controller directs how the conversation develops turn by turn until it reaches a natural ending.

This example uses the `core/conversation` databuilder, DiGiT's built-in pipeline for multi-turn data generation.

## Prerequisites

Pull the default model if you haven't already:

```bash
ollama pull granite4:3b
```

## Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/chit_chat/task.yaml \
  --num-outputs-to-generate 10 \
  --restart
```

Two output files are written under `output/public/examples/chit_chat/`:

- `final_data.jsonl`: full internal representation with all pipeline steps
- `formatted_output.jsonl`: clean `messages` format ready for SFT fine-tuning

## Sample formatted output

```json
{
  "conversation_id": "a3f2c1d8",
  "messages": [
    {
      "role": "user",
      "content": "I was just reading about the James Webb Space Telescope. What's the most exciting thing it has discovered so far?"
    },
    {
      "role": "assistant",
      "content": "One of the most exciting findings is the first clear detection of carbon dioxide in an exoplanet atmosphere..."
    },
    {
      "role": "user",
      "content": "Could Webb actually find signs of life on another planet someday?"
    },
    {
      "role": "assistant",
      "content": "It's possible. Webb can analyze atmospheric chemistry by measuring which wavelengths of starlight are absorbed..."
    }
  ]
}
```

The `conversation_id` matches the corresponding record in `final_data.jsonl` for cross-referencing.

## How it works

The pipeline runs in two phases for each conversation:

1. **Initialization**: a scenario is generated from seed examples, then a user persona is assigned.
2. **Iteration**: a flow controller selects a conversation pattern (new topic, follow-up, clarification, etc.) each turn, a user turn is generated matching that pattern, and the assistant responds. The loop continues until the flow controller signals termination or `max_turns` is reached.

For a full explanation of the stage pipeline, see the [chit-chat example page](../../../../docs/examples/chit_chat.md) in the docs.

## Customization

- **Personas**: edit `data/public/examples/chit_chat/personas.jsonl` or point `persona_store.data_path` in `task.yaml` at your own file to change the user personas sampled during generation.
- **Patterns**: add, remove, or reweight entries under `iteration_stages[lm/flow_controller].patterns` in `task.yaml` to change how conversations develop.
- **Seed examples**: edit `data/public/examples/chit_chat/seed_examples.jsonl` to change the scenarios the model generates.
