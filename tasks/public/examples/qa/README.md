# Geography QA

Generates geography question-answer pairs using in-context learning. The `SimpleDataBuilder` reads the seed examples in `task.yaml`, prompts the language model to produce new questions and answers in the same style, and deduplicates near-identical outputs using Rouge-L scoring.

## Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

Output: `output/public/examples/geography_qa/final_data.jsonl`

## Sample output

```json
{
  "task_name": "public/examples/geography_qa",
  "is_seed": false,
  "instruction": "What is the longest river in South America?",
  "input": "",
  "output": "The Amazon River is the longest river in South America, stretching approximately 6,400 kilometers."
}
```

## Next steps

- Rate the generated pairs for difficulty: see [`../rate/`](../rate/README.md).
- Swap the LM engine: see [Changing the Language Model Engine](../../../../docs/tutorials/changing_lm_engine.md).
