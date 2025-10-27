# Skills Generation



Data builder used for generating instruction-response pairs driven by examples in the compositional skills branch of InstructLab Taxonomy.



It generates data using the open-source Mixtral-7x8B model as the teacher (generator and evaluator) and prompt templates.



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



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be `skills_sdg`

- `taxonomy_path`: (str, optional) used to indicate part of instruct-lab taxonomy data is produced from (defaults to data file path)



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/instructlab/skills/writing/freeform/debate/task.yaml).



## Data specification



Tasks executed by this data builder require seed examples that use the following parameters



### Parameters



- `question`: (str) task for model to follow

- `answer`: (str) result that model should produce

- `context`: (str) optional context for the question and answer



An example can be found in the `seed_examples` field [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/instructlab/skills/writing/freeform/debate/task.yaml).



## Databuilder specification



Default configuration for generator and validator used by the data builder is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/skills_sdg.yaml).



- `generator`: `mistralai/mixtral-8x7B-instruct-v0.1` via `watsonx`

- `validator`: `mistralai/mixtral-8x7B-instruct-v0.1` via `lm_judge` and `watsonx`



In addition, we also pass the following parameters:



- `teacher_config`: (str) Path to prompt templates for various stages of the data generation process. (defaults to `templates/teacher_config.yaml`)

- `num_prompt_instructions`: (int) No. of ICL examples to use per prompt (defaults to `3`)

- `request_batch_size`: (int) No. of samples to generate per prompt (defaults to `5`)



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.watsonx --task-paths ./tasks/watsonx/instructlab/skills/writing/freeform/debate/task.yaml

```



This launches a data generation job by passing seed examples data using the `--task-paths` argument.



By default, the generation engine used is WatsonX. To use RITS, we override the databuilder config:



```

python -m fms_dgt.research --task-paths ./tasks/watsonx/instructlab/skills/writing/freeform/debate/task.yaml --config-path ./configs/skills_sdg_rits.yaml

```



## Explanation



As you can see there's a `data_builder` field in the [task.yaml](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/instructlab/skills/writing/freeform/debate/task.yaml) file that points to the databuilder to be used for this task.



```yaml

created_by: IBM Research

data_builder: skills_sdg

seed_examples:

  - answer: ...

    question: ...

  - answer: ...

    question: ...

```



This particular task does freeform generation of QA pairs using seed examples. More specifically, the seed examples are passed to the `__call__` method in [`generate.py`](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/generate.py). Based on whether the particular task has `context` or not the data generation flow is determined.



#### Without context:



1. Generate freeform questions using [question_template_freeform](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/question_template_freeform.yaml)

2. Validate generated freeform questions using [filter_questions_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/filter_questions_template.yaml)

3. Generate answers for the validated freeform questions using [answer_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/answer_template.yaml)

4. Validate the final QA pairs using [filter_qa_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/filter_qa_template.yaml)



#### With context:



1. Generate freeform context using [context_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/context_template.yaml)

2. Generate question grounded on the generated freeform context using [question_template_grounded](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/question_template_grounded.yaml)

3. Validated the generated grounded questions with context using [filter_questions_context_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/filter_questions_context_template.yaml)

4. Generate answers for the validated grounded questions with context using [answer_context_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/answer_context_template.yaml)

5. Validate the final grounded QA pairs using [filter_qa_context_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/instructlab/skills_sdg/templates/filter_qa_context_template.yaml)



For an example of grounded QA generation, run the following command:



```

python -m fms_dgt.watsonx --task-paths ./tasks/watsonx/instructlab/skills/writing/grounded/editing/grammar/task.yaml

```



## Contributors



**Authors**: Siva Sankalp Patel, Maxwell Crouse
