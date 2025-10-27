# Natural Conversation Variation (NCV)



**[Installation](#installation) | [Task Specification](#task-specification) | [Databuilder specification](#databuilder-specification) | [Usage](#usage)| [Output](#output) | [Contributors](#contributors)**



This data builder creates synthetic data for training the Granite Guardian models to perform correction alongside existing detection capabilities. It generates questions as well as 'bad' and 'good' responses to those questions and uses two LLMs in the process - one acting as the 'generation' model and one acting as the 'critic' model.



- Stage 1: Topic generation

  Generation starts with a 'topic brainstorming' step which maximizes topic diversity. The topics are generated based on seed data that contains topics and question types. The topics are deduplicated and the unique topics are then used as seeds in the subsequent stages.



- Stage 2: Question generation

  One question per topic is generated using seeds from the prior stage. Additional filtering is applied to generated questions to remove non-english, grammatically incorrect questions.



- Stage 3: Principles generation

  Principles are generated for each question using seeds from the prior stages. Principles are critiqued and refined to produce well tailored principles.



- Stage 4: Response generation

  Initially, a 'bad' response is generated for each question-principle pair. A 'bad' response answers the question in a manner that contradicts the principles. A critic model - an LLM that is different from the generation model is used to identify those contradictions in the 'bad' response in the form of score and feedback. The generation model is provided with the score and feedback from the critic model, alongside the question and 'bad' response, and is asked to rephrase the response. This procedure iteratively improves the response until it scores >= 4 (/5), at which point the response is considered 'good' and is saved as the final 'aligned' response together with the initial 'bad' response. For instance, a principle could be: "When responding to the question, avoid discrimination based on gender, age, or socioeconomic status". A 'good' response would be one that adheres to this principle while a 'bad' response violates the principle.



  Additional filtering is applied to generated responses to remove non-english, grammatically incorrect responses.



## Installation



This databuilder uses `spacy` to perform syntactical quality checks on generated questions, bad and aligned responses. Please make sure you install additional depedencies as follows



```bash

pip install -e ".[mitigator]"

python -m spacy download en_core_web_sm

```



## Task specification



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/mitigator/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be `mitigator`

- `topic_icls_path`: (str) path to JSONL file containing `{"topic": "...", "question_type": "...", "question": "..."}` per line. Please always use path w.r.t to `${DGT_DATA_DIR}`.

- `principle_and_refined_response_icls_path`: (str) path to JSONL file containing `{"topic": "...", "question_type": "...", "question": "...", "principles": "...", "bad_response": "...", "aligned_response": "..."}` per line. Please always use path w.r.t to `${DGT_DATA_DIR}`.



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/mitigator/task.yaml).



## Databuilder specification



The default configuration for all stages of this pipeline is available [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/mitigator/mitigator.yaml).



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.research --task-paths ./tasks/research/mitigator/task.yaml --num-outputs-to-generate 100

```



## Output



The output `.jsonl` file includes the following fields:



- `task_name`: Name of the task

- `is_seed`: False

- `topic`: Generated topic

- `question_type`: Related question type

- `question`: Generated question

- `principles`: Generated principles

- `bad_response`: Generated bad response

- `aligned_response`: Generated aligned response

- `critque`: Final critique for aligned response



## Contributors



**Author and Maintainer**: Rebecka Nordenlow, Kshitij Fadnis
