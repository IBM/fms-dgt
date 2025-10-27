# Function Calling Data Generator



Data builder used to generate synthetic function calling data for the following tasks



- Sequencing: Determine the full function calls (i.e., function names and their arguments) needed to answer a given question (see [here](../../../../../tasks/watsonx/tool_calling/single_turn/basic/))

  - Parallel Single Function: Produce a query and multiple calls to the same function (see [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/tool_calling/single_turn/basic/parallel_single/task.yaml))

  - Parallel Multiple Function: Produce a query and multiple calls to the different function (see [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/tool_calling/single_turn/basic/parallel_multiple/task.yaml))



## Task specification



This data builder supports [tasks](https://github.ibm.com/DGT/fms-dgt/tree/main/fms_dgt/watsonx/databuilders/tool_calling/single_turn/task.py) defining the following parameters



### Parameters



- `created_by`: (str) creator of the task.

- `task_description`: (str) description of the task.

- `data_builder`: (str) must be `function_calling`

- `task_instruction`: (str) general description of function-calling task that will be fed to model for each example

- `fc_specifications`: (Dict) a dictionary with keys being function calling namespaces and values being dictionaries containing all functions in a group

- `exclude_fc_namespaces`: (List[str] = None) a list of any function calling namespaces (if any) in `fc_specifications` to exclude

- `min_func_count`: (int = 1) minimum number of functions a function call should be constructed from

- `max_func_count`: (int = 1) maximum number of functions a function call should be constructed from

- `check_arg_question_overlap`: (bool = False) enforce that all arguments to parameters in function call should be drawn from substring of question

- `single_function`: (bool = False) task requires every function call to use the same function

- `require_nested`: (bool = False) require that generated examples contain at least one nested function call (i.e., a function call where an argument uses the result of a previous function call)

- `allow_subset`: (bool = False) allow generated function calls to use a subset of the randomly selected functions from the specification



An example can be found [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/tool_calling/single_turn/basic/parallel_multiple/task.yaml).



## Data specification



Tasks executed by this data builder require seed examples that use the following parameters



### Parameters



- `input`: (str) Question / request that a function call is required to answer

- `output`: (str) The function call used to solve the `input`

- `namespace`: (str = None) The namespace of the fc specification used to generate `output` (this is not required if there is only one namespace)



An example can be found in the `seed_examples` field [here](https://github.ibm.com/DGT/fms-dgt/tree/main/tasks/watsonx/tool_calling/single_turn/basic/parallel_multiple/task.yaml).



## Function Calling specification



This databuilder accepts specifications written as a dictionary of OpenAPI schema objects. To include your own, create a yaml file (e.g., see [here](../../../../../data/watsonx/tool_calling/single_turn/instana/analytics/instana_api_base.yaml)) with your specification. It should be a straightforward conversion of the JSON format into yaml as follows



```yaml

<name of your FC namespace here, e.g., instana / apptio / gmail / etc.>:

  <function name here>:

    <function 1 specification here>: ...

    <function 2 specification here>: ...

```



This might look something like



```yaml

AccessLogEntry: # <-- function name is the key and the OpenAPI data schema is the value

  name: AccessLogEntry

  description: ""

  parameters:

    properties:

      action:

        enum:

          - GRANT_TEMP_ACCESS

          - FIRST_LOGIN

          - LOGIN

          - ACCESS

          - FAILED_LOGIN

          - LOGOUT

        type: string

      email:

        type: string

      fullName:

        type: string

      tenantId:

        type: string

      tenantUnitId:

        type: string

      timestamp:

        format: int64

        type: integer

    type: object

AccessRule:

  name: AccessRule

  description: ""

  parameters:

    properties:

      accessType:

        enum:

          - READ

          - READ_WRITE

        type: string

      relatedId:

        maxLength: 64

        minLength: 0

        type: string

      relationType:

        enum:

          - USER

          - API_TOKEN

          - ROLE

          - TEAM

          - GLOBAL

        type: string

    type: object

```



## Examples



- Sequencing: Determine the full function calls (i.e., function names and their arguments) needed to answer a given question



  - Parallel Single Function: Produce a query and multiple calls to the same function



    ```bash

      - input: "Hi, I need to convert 500 US dollars to Euros and another 100 Sterling to Euros. Can you help me with that?"

        output: '[ { "name": "convert_currency", "arguments": { "from_currency": "USD", "to_currency": "Euros", "amount": 500} }, { "name": "convert_currency", "arguments": { "from_currency": "Sterling", "to_currency":"Euros", "amount": 100} } ]'

        namespace: glaive

    ```



  - Parallel Multiple Function: Produce a query and multiple calls to the different function



    ```bash

      - input: "find the train that leaves at 12:30 on friday to stevenage. find the attraction in the west area that is called churchill college and is of the dontcare type."

        output: '[ { "name": "find_train", "arguments": { "train-day": "friday", "train-destination": "stevenage", "train-leaveat": "12:30" } }, { "name": "find_attraction", "arguments": { "attraction-area": "west", "attraction-name": "churchill college", "attraction-type": "dontcare" } } ]'

        namespace: multiwoz

    ```



## Evaluation



TBD



## How to run



- Sequencing:



  ```bash

  python -m fms_dgt.watsonx --task-paths ./tasks/watsonx/tool_calling/single_turn/basic --num-outputs 1000 --restart

  ```



  - Parallel Single Function:



    ```bash

    python -m fms_dgt.watsonx --task-paths ./tasks/watsonx/tool_calling/single_turn/basic/parallel_single --num-outputs 1000 --restart

    ```



  - Parallel Multiple Function:



    ```bash

    python -m fms_dgt.watsonx --task-paths ./tasks/watsonx/tool_calling/single_turn/basic/parallel_multiple --num-outputs 1000 --restart

    ```



## Citation



If you use this pipeline in this work, please cite the following paper



```

@article{abdelaziz2024granite,

  title={Granite-function calling model: Introducing function calling abilities via multi-task learning of granular tasks},

  author={Abdelaziz, Ibrahim and Basu, Kinjal and Agarwal, Mayank and Kumaravel, Sadhana and Stallone, Matthew and Panda, Rameswar and Rizk, Yara and Bhargav, GP and Crouse, Maxwell and Gunasekara, Chulaka and others},

  journal={arXiv preprint arXiv:2407.00121},

  year={2024}

}

```
