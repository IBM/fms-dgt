# Skills Generation

Data builder used for generating instruction-response pairs driven by examples in the compositional skills branch of InstructLab Taxonomy.

It generates data using the specified model as the teacher (generator and validator) and prompt templates.

> [!WARNING]  
> **Issue**
>
> Segmentation faults and similar errors on macOS
>
> **Solution**
>
> Set the following environment parameters
>
> 1. Disable OpenMP threading via `export OMP_NUM_THREADS=1`
> 2. If error still persists, disable PyTorch MPS device via `export PYTORCH_MPS_DISABLE=1`
> 3. If error still persists, disable llama.cpp metal via `export LLAMA_NO_METAL=1`
> 4. Final attempt can be made as a dangerous workaround via `export KMP_DUPLICATE_LIB_OK=TRUE`
>
> Reference: https://github.com/neuml/txtai/issues/813#issuecomment-2485349327

## Task specification

This data builder supports [tasks](./task.py) defining the following parameters:

### Parameters

- `task_name`: (str) Name of the task
- `created_by`: (str) Creator of the task
- `task_description`: (str) Description of the task
- `data_builder`: (str) Must be `instructlab/skills`

## Data specification through reserved keywords

Tasks executed by this data builder require seed examples that use the following parameters

#### Seed examples

Seed examples can be provided through the `seed_examples` field with the following parameters:

- `question`: (str) task for model to follow
- `answer`: (str) result that model should produce
- `context`: (str) optional context for the question and answer

### Additional Task Parameters

In addition, we can also pass the following parameters:

- `num_icl_examples_per_prompt`: (int) No. of ICL examples to use per prompt (defaults to `3`)
- `num_questions_to_generate_per_prompt`: (int) No. of questions to generate per prompt (defaults to `5`)

An example can be found [here](../../../../../tasks/public/instructlab/skills/writing/freeform/debate/task.yaml).

## Databuilder specification

- `generator`: `mistral-small3.2` via `ollama`
- `validator`: `mistral-small3.2` via `lm_judge` and `ollama`
- `tagger`: `mistral-small3.2` via `magpie_tag` and `ollama` (see [Magpie Tagger](../../../blocks/magpie/tag/README.md) block)

#### Postprocessors:

- `dedup`: via `magpie_distance` (see [Magpie Distance](../../../blocks/magpie/distance/))
- `filter` via `magpie_filter` (see [Magpie Filter](../../../blocks/magpie/filter/README.md))

Default configuration for generator and validator used by the data builder is available [here](./skills.yaml).

## Usage

To try out the databuilder, run the following command:

```
python -m fms_dgt.public --task-paths ./tasks/public/instructlab/skills/writing/freeform/debate/task.yaml
```

This launches a data generation job by passing seed examples data using the `--task-paths` argument.

## Explanation

As you can see there's a `data_builder` field in the [task.yaml](../../../../../tasks/public/instructlab/skills/writing/freeform/debate/task.yaml) file that points to the databuilder to be used for this task.

```yaml
created_by: IBM Research
data_builder: instructlab/skills
seed_examples:
  - answer: ...
    question: ...
  - answer: ...
    question: ...
```

This particular task does freeform generation of QA pairs using seed examples. More specifically, the seed examples are passed to the `__call__` method in [`generate.py`](generate.py). Based on whether the particular task has `context` or not the data generation flow is determined.

#### Without context:

1. Generate freeform questions using the prompt in [freeform_question_generation.txt](./prompt_templates/freeform_question_generation.txt)
2. Validate generated freeform questions using the prompt in [freeform_question_validation.txt](./prompt_templates/freeform_question_validation.txt)
3. Generate answers for the validated freeform questions using the prompt in [answer_generation.txt](./prompt_templates/answer_generation.txt)
4. Validate the final QA pairs using the prompt in [answer_validation.txt](./prompt_templates/answer_validation.txt)

#### With context:

1. Generate freeform context using the prompt in [context_generation](./prompt_templates/context_generation.txt)
2. Generate question grounded on the generated freeform context using the prompt in[context_based_question_generation.txt](./prompt_templates/context_based_question_generation.txt)
3. Validated the generated grounded questions with context using the prompt in [context_based_question_validation.txt](./prompt_templates/context_based_question_validation.txt)
4. Generate answers for the validated grounded questions with context using the prompt in[context_based_answer_generation.txt](./prompt_templates/context_based_answer_generation.txt)
5. Validate the final grounded QA pairs using the prompt in [context_based_answer_validation](./prompt_templates/context_based_answer_validation.txt)

For an example of grounded QA generation, run the following command:

```
python -m fms_dgt.public --task-paths ./tasks/public/instructlab/skills/writing/grounded/editing/grammar/task.yaml
```

## Contributors

**Authors**: Siva Sankalp Patel, Maxwell Crouse, Kshitij Fadnis
