# InstructLab: Skills and Knowledge

DiGiT ships two databuilders that implement the [LAB method](https://arxiv.org/abs/2403.01081) for generating instruction-tuning data. Both are ready to run with the included example tasks.

- **Skills** generates instruction-response pairs for a capability you want to teach, such as writing, editing, or reasoning.
- **Knowledge** generates question-answer pairs grounded in a reference document, for teaching factual or domain-specific content.

!!! note
    These databuilders are inspired by InstructLab's implementation of the LAB method and are intended for demonstration and experimentation. They may differ from InstructLab's production pipeline.

## Skills: freeform debate generation

This example generates debate-style responses where the model argues multiple perspectives on a topic.

### Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/instructlab/skills/writing/freeform/debate/task.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

### Task specification

The task file is at `tasks/public/instructlab/skills/writing/freeform/debate/task.yaml`. Each seed example provides a `question` (the debate prompt) and an `answer` (a multi-perspective response):

```yaml
task_name: instructlab/skills/writing/freeform/debate
task_description: To teach a language model to formulate debate points
created_by: IBM Research
data_builder: instructlab/skills

seed_examples:
  - question: >
      Debate the merits and drawbacks of implementing a universal basic income
      between an economist, a sociologist, and a policy maker.
    answer: >
      Economist: "Implementing a universal basic income (UBI) could significantly
      reduce poverty rates and provide a financial safety net..."

      Sociologist: "From a sociological perspective, UBI has the potential to
      address income inequality and promote social cohesion..."

      Policy Maker: "As a policy maker, I see the appeal of UBI in its potential
      to alleviate poverty and simplify the welfare system..."
```

### How it works

The skills databuilder runs a multi-stage pipeline:

1. **Generation:** the model produces new question-answer pairs in the style of the seed examples.
2. **Validation:** an LM judge scores each pair for coherence and relevance, filtering out low-quality outputs.
3. **Tagging and deduplication:** each output is tagged for difficulty and quality, and near-duplicates are removed.

### Sample output

Generated data lands in `output/instructlab/skills/writing/freeform/debate/final_data.jsonl`:

```json
{
  "task_name": "instructlab/skills/writing/freeform/debate",
  "is_seed": false,
  "instruction": "Discuss the pros and cons of remote work from the perspective of an employee, a manager, and an HR professional.",
  "input": "",
  "output": "Employee: Remote work offers flexibility and eliminates commuting time, boosting productivity for self-motivated individuals...\n\nManager: Managing remote teams requires strong communication practices and clear goal-setting...\n\nHR Professional: From a talent acquisition standpoint, remote-first policies significantly expand the candidate pool..."
}
```

## Knowledge: photosynthesis QA generation

This example generates question-answer pairs grounded in a biology document on photosynthesis.

### Prerequisites

The example requires the photosynthesis document, which ships with the repository at:

```
data/public/instructlab/knowledge/textbook/science/biology/photosynthesis/photosynthesis.md
```

No download required.

### Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/instructlab/knowledge/textbook/science/biology/photosynthesis/task.yaml \
  --restart
```

### Task specification

The task file is at `tasks/public/instructlab/knowledge/textbook/science/biology/photosynthesis/task.yaml`. Key fields:

```yaml
task_name: instructlab/knowledge/textbook/science/biology/photosynthesis
task_description: To teach a language model about photosynthesis
created_by: IBM Research
data_builder: instructlab/knowledge

seed_examples:
  - question: What is respiration?
    answer: The word respiration is commonly used to describe the process of breathing in oxygen and breathing out carbon dioxide.
  - question: What is an ecosystem?
    answer: An ecosystem is a community of organisms and their physical environment interacting together.
  - question: What is metabolism?
    answer: Metabolism is the chemical reactions in the body's cells that change food into energy.

include:
  documents:
    photosynthesis: ${DGT_DATA_DIR}/public/instructlab/knowledge/textbook/science/biology/photosynthesis/photosynthesis.md

domain: biology
chunk_size: 800
question_style: FRQ
criteria:
  - faithfulness
  - relevancy
  - question_verification
```

The `include.documents` directive loads the reference document. DiGiT chunks it automatically according to `chunk_size` (in tokens) and uses each chunk as grounding context for question generation.

### How it works

The knowledge databuilder runs a multi-stage pipeline:

1. **Generation:** the model generates questions grounded in each document chunk, using the seed examples as style references.
2. **Validation:** an LM judge checks each QA pair against the document for faithfulness and relevancy, and verifies the question is answerable.
3. **Tagging and deduplication:** outputs are tagged and near-duplicates are removed.

### Sample output

Generated data lands in `output/instructlab/knowledge/textbook/science/biology/photosynthesis/final_data.jsonl`:

```json
{
  "task_name": "instructlab/knowledge/textbook/science/biology/photosynthesis",
  "is_seed": false,
  "question": "What role does photosynthesis play in the global carbon cycle?",
  "answer": "Photosynthesis removes carbon dioxide from the atmosphere and converts it into carbohydrates stored in plant tissue. This process counteracts the carbon dioxide released by burning fossil fuels, making photosynthesis a critical regulator of atmospheric carbon and therefore of global climate.",
  "domain": "biology",
  "context": "..."
}
```

## Using a different model or provider

Both databuilders default to `granite4:3b` via Ollama. To switch to a cloud provider, pass a config file:

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/instructlab/skills/writing/freeform/debate/task.yaml \
  --config-path ./configs/public/instructlab/watsonx_skills.yaml \
  --num-outputs-to-generate 20 \
  --restart
```

See [Changing the LM Engine](../tutorials/changing_lm_engine.md) for details on all supported providers.

## Next steps

- Add your own documents to the knowledge databuilder by creating a new task YAML under `tasks/public/instructlab/knowledge/`.
- Add your own skills by creating a new task YAML under `tasks/public/instructlab/skills/`.
- Read the [Skills README](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/instructlab/skills) and [Knowledge README](https://github.com/IBM/fms-dgt/tree/main/fms_dgt/public/databuilders/instructlab/knowledge) for the full list of task parameters.
