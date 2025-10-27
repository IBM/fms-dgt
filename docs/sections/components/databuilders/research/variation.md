# Natural Conversation Variation (NCV)



**[Task Specification](#task-specification) | [Operation List](#operation-list) | [Usage](#usage) | [Contributors](#contributors)**



Extends `TransformationDataBuilder` for generating natural variations of existing conversations.



## Task specification



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/research/databuilders/natural_conversation/variation/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be `ncv`

- `operations`: (List[str]) variation operations to apply. Each operation must be from the [operation list](#operation-list).

- `seed_datastore`: (Dict) seed data containing:

  - `type`: (str) Set to `default`.

  - `data_path`: (str) path to JSONL containing data to transform. Please always use path w.r.t to `${DGT_DATA_DIR}`.



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/research/natural_conversation/ncv/task.yaml).



## Operation List



Each operation represents a conversational pattern created by rearranging, adding, or removing dialogue turns. Below is a summary of the available operations:



- `preliminary_screening`: ask if agent can talk about a topic or provide a kind of help before stating long inquiry.

- `preliminary_detail`: give a detail in advance as context for understanding the inquiry.

- `detail_request`: ask for a detail in order to answer the prior inquiry.

- `extended_answer`: multi-part answer with user turns in between them

- `inquiry_context`: give context for the inquiry.

- `inquiry_reverse`: generate inquiry from statement.

- `repair_answer`: ask for paraphrase, definition, example or repeat of the prior inquiry.

- `understanding_check_inquiry`: ask for confirmation of a paraphrase of the prior inquiry to check understanding.

- `understanding_check_answer`: ask for confirmation of a paraphrase of the prior answer to check understanding.

- `sequence_closing`: The user ends the sequence with appreciation, assessment, agreement, or acknowledgment.

- `preclosing_last_topic_check`: ask if there are more topics to discuss.

- `preclosing_no_more_topics`: state that there are no more topics to discuss.

- `example_request`: ask for example of the prior answer.

- `incremental_request`: ask a question that build on previous responses, typically with elliptical phrases such as what about, how about, etc.

- `expand_topic`: A two-part response where the assistant answers the user's inquiry and then encourages further exploration of a related topic.



## Recommended Combination of Operations



For each variation, we recommend combining 2–3 operations from the sets below:



- Set 1: `extended_answer`, `repair_answer`, `inquiry_reverse`, `inquiry_context`

- Set 2: `incremental_request`, `detail_request`

- Set 3: `preliminary_detail`, `expand_topic`, `sequence_closing`

- Set 4: `preliminary_screening`, `understanding_check_answer`, `example_request`, `preclosing_last_topic_check`



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.research --task-paths ./tasks/research/natural_conversation/ncv/task.yaml

```

## Output



The output `.jsonl` file includes the following fields:



- `task_name`: Name of the task

- `is_seed`: False

- `input`: Original input conversation

- `debug`: Details of any replacements made for each operation, including:

  - `replaced`: Turns that were changed

  - `inserted`: Turns that were inserted

  - `replacement`: The replacements used

  - `model_response`: Raw model responses

- `validation`: Conversation quality checks:

  - `is_valid`: Validity flag based on the following

 	 - `format validator`: (boolean) the syntatic check for verifying all mandatory fields are filled

  	- `sequence validator`: (boolean) the syntatic check for verify valid sequence of operations

  	- `label_utterance_alignment`: (high/low) A language model-based evaluator that rates how well an utterance aligns with its assigned conversational label.

- `output`: Final generated conversation





## Contributors



**Author and Maintainer**: Sungeun An, Robert Moore, Kshitij Fadnis
