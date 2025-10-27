# Multi-turn Conversation Generation



Data builder used for generating multi-turn conversation data for the following tasks:



- Freeform

<!-- - RAG

- Tool-calling -->



The `MultiTurnDataBuilder` orchestrates the conversation generation process through 4 main stages:



- **Scenario generator**: This stage sets the problem scenario on which to generate a conversation.

- **Flow controller**: This stage looks at the problem scenario and conversation history to decide the next user interaction pattern.

- **User Actor**: The user is responsible for generating the next user turn in the conversation

- **Assistant Actor**: The assistant is responsible for generating the next assistant turn in the conversation.



Once the scenario generator produces a scenario, the conversation generation process flows through the flow controller, user, and assistant until the flow controller decides to end the conversation or `max_turns` (defined in task.yaml below) have been reached (fail-safe to avoid infinite loops).



![multiturn-workflow](https://github.ibm.com/DGT/fms-dgt/tree/main/docs/imgs/multiturn-workflow.png)



## Task specification



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/core/databuilders/multi_turn/task.py) defining the following parameters:



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be either `multiturn` or `multiturn_v2`

- `config`: (Dict) configuration for each of the four stages.

  - `max_turns`: (int) Max. number of turns for conversations

  - `scenario_generator`: (Dict) Scenario generator configuration (more details [here](./scenario_generators/README.md))

  - `flow_controller`: (Dict) Flow controller configuration (more details [here](./flow_controllers/README.md))

  - `user`: (Dict) User actor configuration (more details [here](./actors/README.md))

  - `assistant`: (Dict) User actor configuration (more details [here](./actors/README.md))



An example can be found [here](../../../../tasks/core/multiturn/).



## Data specification



Tasks executed by this data builder require seed examples. Each stage of the generation process uses seeds for generating data.



- Seed examples for freeform conversation generation can be found [here](../../../../data/core/multiturn/freeform/email/seeds/)

<!-- - Examples for the RAG use-case can be found [here](../../../../data/core/multiturn/documents/)

- Examples for the tool-calling use-case can be found [here](../../../../data/core/multiturn/tool_calling/selection/glaive/) -->



<!-- ### Parameters -->



## Databuilder specification



There are few variants of the multi-turn databuilder.



- `multiturn`: Simple variant that uses for separate language models for each of the stages. Default config is [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/core/databuilders/multi_turn/multiturn.yaml).

- `multiturn_v2`: Uses an additional language model in the assistant stage for validation. Default config is [here](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/core/databuilders/multi_turn/multiturn_v2.yaml)



Here are the blocks that need to be defined:



- `scenario_generator_lm`: LM defined via `rits`, `watsonx`, or `vllm`

- `flow_controller_lm`: LM defined via `rits`, `watsonx`, or `vllm`

- `user_lm`: LM defined via `rits`, `watsonx`, or `vllm`

- `assistant_lm`: LM defined via `rits`, `watsonx`, or `vllm`



In addition, the following may or may not be needed depending on the task and databuilder variant being used.



- `assistant_val`: LM judge (`lm_judge`) defined via `rits`, `watsonx`, or `vllm` (Compatible with `multiturn_v2`)



<!-- - `assistant_val`: LM judge (`lm_judge`) defined via `rits`, `watsonx`, or `vllm` (Compatible with `multiturn_v2` and `multiturn_tool_calling`) -->

<!-- - `functions_lm`: LM defined via `rits`, `watsonx`, or `vllm` (Compatible only with `multiturn_tool_calling`) -->



## Usage



To try out the databuilder, run the following command:



```

python -m fms_dgt.research --task-paths ./tasks/core/multiturn/freeform/email/task.yaml

```



## Contributors
