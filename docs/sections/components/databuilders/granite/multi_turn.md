# Multi-turn Conversation Generation for Tool-calling



Extends `MultiTurnDataBuilder` for the generating conversations with tool-calling.



Refer to the documentation of base databuilder [here](../../../../core/databuilders/multiturn/README.md).



## Task specification



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/granite/databuilders/tool_calling/multi_turn/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be `multiturn_tool_calling`

- `config`: (Dict) configuration for each of the four stages.

  - `max_turns`: (int) Max. number of turns for conversations

  - `scenario_generator`: (Dict) Scenario generator configuration (more details [here](./scenario_generators/README.md))

  - `flow_controller`: (Dict) Flow controller configuration (more details [here](../../../../core/databuilders/multiturn/flow_controllers/README.md))

  - `user`: (Dict) User actor configuration (more details [here](./actors/README.md))

  - `assistant`: (Dict) User actor configuration (more details [here](./actors/README.md))



An example can be found [here](../../../../../tasks/granite/tool_calling/multiturn/selection/glaive/task.yaml).



## Data specification



Tasks executed by this data builder require seed examples. Each stage of the generation process uses seeds for generating data.



- Examples for the tool-calling use-case can be found [here](../../../../../data/granite/tool_calling/multiturn/selection/glaive/)



<!-- ### Parameters -->



## Databuilder specification



We currently have one variant of the multi-turn databuilder for tool-calling:



- `multiturn_tool_calling`: Uses a language model in the assistant stage for tool-calling. Default config is [here](./multiturn_tool_calling.yaml)



Here are the blocks that need to be defined:



- `scenario_generator_lm`: LM defined via `rits`, `watsonx`, or `vllm`

- `flow_controller_lm`: LM defined via `rits`, `watsonx`, or `vllm`

- `user_lm`: LM defined via `rits`, `watsonx`, or `vllm`

- `assistant_lm`: LM defined via `rits`, `watsonx`, or `vllm`



In addition, the following may or may not be needed depending on the task and databuilder variant being used.



- `assistant_val`: LM judge (`lm_judge`) defined via `rits`, `watsonx`, or `vllm` (Compatible with `multiturn_tool_calling`)

- `functions_lm`: LM defined via `rits`, `watsonx`, or `vllm` (Compatible only with `multiturn_tool_calling`)



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.research --task-paths ./tasks/granite/tool_calling/multiturn/selection/glaive/task.yaml

```



## Tools



Framework overview - ![image info](https://github.ibm.com/DGT/fms-dgt/tree/main/docs/imgs/tool_calling_img.png)



### Sources



```

- general_stack

- stack

- glaive

- multiwoz

- sgd

- topv2

- linux

- mathqa

```



### Example of tool specification



```

find_interval_distance:

  name: find_interval_distance

  description: Calculates the total distance between adjacent intervals in a list

    of intervals.

  parameters:

    type: object

    properties:

      intervals:

        type: array

        description: A list of intervals, where each interval is represented as a

          tuple (start, end).

    required:

    - intervals

  output_parameters:

    type: object

    properties:

      result:

        type: number

        description: A value of type <class 'float'>

    required:

    - result

```



### Tool Graph



Automatically constructed by embedding similarity between outgoing edges of tool (i.e., output parameters) and incoming edges of tool (i.e., input arguments)



```

convert_decibel_to_linear:

  dB_to_linear: 0.7171

  power_of_scalar: 0.6253

  scale_linearly: 0.5999

  piecewise_linear: 0.5999

  multiply_list: 0.5986

get_main_diagonal:

  extract_matrix_elements: 0.6986

  generate_symmetric_matrix: 0.6655

  select_first_and_last: 0.6419

  create_identity_matrix: 0.6292

  matrix_multiply_2: 0.5808

```



### Execution Engines



```

- Python

- LLM

- Simulated Linux shell + filesystem

```



## Filters



All function calls generated must



```

- Use substrings of conversation as arguments (reduces hallucinations)

- Must be well-formed w.r.t. provided tool specification

```



## Example of Scenario



```

Bob has recently joined the company and was assigned a new project.

The project needs an isolated environment, and his colleague recommended creating a new user in the linux system for this purpose.

He has decided to create a user called 'bob_project'.

He needs to change the ownership of a file called 'notes.txt' to this new user.

He also wants to update the 'bob_project' user password to strengthen security.

```



## Example of Interaction



```

[

    {

        "role": "user",

        "content": "Can you please help me add a new user called 'bob_project' in the linux system? Once done, I am looking to update the ownership of 'notes.txt' file to this new user and then update the password for this 'bob_project' user. Thanks."

    },

    {

        "role": "assistant",

        "tool_call": {

            "id": "$1",

            "type": "function",

            "function": {

                "name": "useradd",

                "arguments": {

                    "user": "bob_project"

                }

            }

        }

    },

    {

        "role": "tool",

        "content": {

            "status": "created"

        },

        "tool_call_id": "$1",

        "type": "json"

    },

    {

        "role": "assistant",

        "content": "I see that the 'bob_project' user has been created successfully. To proceed with updating the ownership of the 'notes.txt' file to 'bob_project', please provide the complete path to the file.\n\nOnce I have the file path, I will help you update the ownership using the 'chown' tool. After that, I can assist you in updating the password for the 'bob_project' user using the 'passwd' tool.\n\nPlease provide the file path for 'notes.txt' so that I can assist you further."

    },

    {

        "role": "user",

        "content": "Thank you for helping me create the new user. 'notes.txt' can be found in the '/home/bob/projects/notes.txt' directory."

    },

    {

        "role": "assistant",

        "tool_call": {

            "id": "$2",

            "type": "function",

            "function": {

                "name": "chown",

                "arguments": {

                    "file": "/home/bob/projects/notes.txt",

                    "group": "bob_project",

                    "user": "bob_project"

                }

            }

        }

    },

    {

        "role": "tool",

        "content": {

            "success": true

        },

        "tool_call_id": "$2",

        "type": "json"

    },

    {

        "role": "assistant",

        "tool_call": {

            "id": "$3",

            "type": "function",

            "function": {

                "name": "passwd",

                "arguments": {

                    "user": "bob_project"

                }

            }

        }

    },

    {

        "role": "tool",

        "content": {

            "success": true

        },

        "tool_call_id": "$3",

        "type": "json"

    },

    {

        "role": "assistant",

        "content": "Great news! We have created the new user 'bob_project', updated the ownership of '/home/bob/projects/notes.txt' to this user, and successfully set a new password for 'bob_project'. If you have any other tasks, please let me know."

    },

    {

        "role": "user",

        "content": "Thank you so much! I am all set now. Have a good day!"

    },

    {

        "role": "assistant",

        "content": "You're welcome! I'm glad I could help you create the new user, update file ownership, and set a new password for 'bob_project'. Have a great day ahead!"

    },

]

```



## Results



Results with granite-3.1-8b-instruct for BFCL v3:



- Using 1k DGT multi-turn FC data points

- Concatenating conversations to generate longer scenarios



Results for baseline / after training (accuracy):



- multi_turn_base. 0.075 -> 0.315

- multi_turn_miss_func. 0.03 -> 0.155

- multi_turn_long_context. 0.065 -> 0.245

- multi_turn_miss_param. 0.05 -> 0.15
