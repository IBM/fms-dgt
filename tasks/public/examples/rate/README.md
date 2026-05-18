# QA Rater

Scores existing question-answer pairs for difficulty using an LLM judge. This example reads the geography QA data produced by the [`qa`](../qa/README.md) example and adds a numeric `rating` field to each record.

## Prerequisites

Run the QA generation example first to produce the input data:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/qa/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

## Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/rate/task.yaml \
  --restart
```

Output: `output/public/examples/qa_ratings/final_data.jsonl`

## Sample output

```json
{
  "task_name": "public/examples/qa_ratings",
  "is_seed": false,
  "question": "What is the deepest lake in the world?",
  "answer": "Lake Baikal in Siberia, Russia, is the deepest lake in the world.",
  "rating": 2
}
```

## Next steps

- Swap the LM engine: see [Changing the Language Model Engine](../../../../docs/tutorials/changing_lm_engine.md).
- To understand how transformation databuilders work, see [Data Transformation](../../../../docs/tutorials/transform_data.md).
