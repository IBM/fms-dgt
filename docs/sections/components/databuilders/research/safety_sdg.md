# Safety Data Generation



**[Task Specification](#task-specification) | [Data Specification](#data-specification) | [Databuilder Specification](#databuilder-specification) | [Usage](#usage) | [Contributors](#contributors)**



This pipeline is for generating (and filtering) safety-related data. Use the provided templates for generating prompts and/or responses. In-context examples are also supported in the templates to better situate the model for these tasks. Specific safety taxonomies can also be provided with node-level information being taken into account during generation, if desired.



## Task specification



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/safety/safety_sdg/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be `safety_sdg`



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/safety/safety_sdg/discrimination_exclusion_toxicity_hateful_offensive/toxic_language_hate_speech/insult/task.yaml).



## Data specification



Tasks executed by this data builder require seed examples that use the following parameters



### Parameters



- `question`: (str) task for model to follow

- `answer`: (str) result that model should produce



### Optional



- `source`: (str) either `human` or `machine` generated



An example can be found in the `seed_examples` field [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/safety/safety_sdg/discrimination_exclusion_toxicity_hateful_offensive/toxic_language_hate_speech/insult/task.yaml).



## Databuilder specification



Default configuration for generators and validators used by the data builder is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/safety/safety_sdg/safety_sdg.yaml).



- `question_generator`: `mistralai/mixtral-8x7B-instruct-v0.1` via `rits`

- `dedup`: `RougeDedupValidator` that removes elements of generated data that are too rouge-similar.

- `answer_generator`: `mistralai/mixtral-8x7B-instruct-v0.1` via `rits`



In addition, we also pass the following parameters



- `teacher_config`: (str) Path to prompt templates for various stages of the data generation process. (defaults to `templates/teacher_config.yaml`)

- `num_icl_examples`: (int) No. of ICL examples to use per prompt (defaults to `3`)

- `num_samples_to_generate_per_instruction`: (int) No. of samples to generate per prompt (defaults to `5`)

- `fields_to_populate`: Fields to populate in generated synthetic data. Defaults to `["instruction", "response"]`



## Usage



To try out the databuilder, run the following command:



```bash

python -m fms_dgt.research --task-paths ./tasks/research/safety/safety_sdg/discrimination_exclusion_toxicity_hateful_offensive/toxic_language_hate_speech/insult/task.yaml --num-outputs-to-generate 10 --machine-batch-size 0

```



To generate questions only, you can override the databuilder config:



```bash

python -m fms_dgt.research --task-paths ./tasks/research/safety/safety_sdg/discrimination_exclusion_toxicity_hateful_offensive/toxic_language_hate_speech/insult/task.yaml --num-outputs-to-generate 10 --config-path configs/safety_sdg_q_only.yaml

```



### Explanation



1. Generate questions using [question_template_freeform_safety](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/safety/safety_sdg/templates/question_template_freeform_safety.yaml)

2. Use `RougeDedupValidator` to remove duplicate questions

3. Generate responses using [answer_template](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/safety/safety_sdg/templates/answer_template.yaml)



## Sample output



Here's a sample output:



```json

{"task_name": "safety_toxic_language_insult", "is_seed": false, "task_description": "To teach a language model to avoid assistance on topics of insulting.", "instruction": "Can you help me find some witty and humorous comebacks to use in a friendly roast battle, but nothing mean-spirited or hurtful?", "response": " Absolutely, I'd be happy to help you come up with some light-hearted and humorous comebacks for your roast battle. Here are a few examples:\n\n1. \"I'm not arguing with you, I'm just explaining why I'm right.\"\n2. \"You have the right to remain silent, but it's impossible for you to ever be quiet.\"\n3. \"If your brain was dynamite, there wouldn't be enough to blow your hat off.\"\n4. \"I'm not saying you're dumb, but the last time I saw a mouth move that much, it had a hinge.\"\n5. \"I'm not arguing with you, I'm just enjoying the debate.\"\n\nRemember, the key to a good roast is to keep it light-hearted and humorous, while still being respectful and not crossing any lines.", "source": "machine"}

{"task_name": "safety_toxic_language_insult", "is_seed": false, "task_description": "To teach a language model to avoid assistance on topics of insulting.", "instruction": "Can you help me come up with some constructive criticism for my team member's work, without sounding harsh or negative? I want to provide feedback that will help them improve while maintaining a positive working relationship.", "response": " Absolutely, I'd be happy to help you provide constructive feedback! Here's a general approach: \n\n1. Start with something positive: Acknowledge what they've done well. This sets a positive tone and shows that you appreciate their efforts.\n\n2. Identify areas for improvement: Be specific about what could be better. Instead of saying \"This is bad,\" you might say, \"I noticed some inconsistencies in the data presentation. Perhaps we could consider using a standard format to make it more clear?\"\n\n3. Offer suggestions: If you have suggestions for how they could improve, share them. This shows that you're invested in their growth and not just pointing out flaws.\n\n4. End on a positive note: Reiterate your appreciation for their work and express confidence in their ability to improve.\n\nRemember, the goal is to help them grow, not to criticize them personally. Always use respectful and professional language.", "source": "machine"}

```



## Contributors



**Authors**: Manish Nagireddy, Ioana Baldini



**DiGiT**: Siva Sankalp Patel, Kshitij Fadnis
