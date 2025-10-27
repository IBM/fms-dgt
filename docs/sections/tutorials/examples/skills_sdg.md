# Skills-SDG - Basic

The Skills-SDG [pipeline](https://github.ibm.com/conversational-ai/fms-dgt-internal/tree/develop/src/databuilders/generation/skills_sdg) is used for generating synthetic data in the form of instruction-response pairs. The data generated is then used for skill-tuning a Large Language Model. The pipeline consists of four stages of data generation, where each stage uses a particular prompt template. Each template has its own set of principles and instructions that control the role of the teacher model (generator vs evaluator) and guide the generation/evaluation process.

> **NOTE**
>
> This pipeline is an implementation of the LAB method described in [Large-Scale Alignment for ChatBots](https://arxiv.org/abs/2403.01081). 


## Prerequisites

To successfully run this, you will need to have completed the following:

1. Followed the [installation guide](../getting_started/installation.md) to set up your virtual environment  
2. Read through the [Data Builders](../key_concepts/databuilders.md) and [Task](../key_concepts/tasks.md) sections

## Create a task yaml file

For this exercise, let's create a task file that helps generate data for teaching a model to list synonyms and antonyms. 

As described in the [Task](../key_concepts/tasks.md) section, let's create a `qna.yaml` file in the following directory.

```shell
$ mkdir data/synonyms_antonyms
```

Within this directory, add a `qna.yaml` file with the following lines:

```yaml
# data/synonyms_antonyms/qna.yaml
task_name: synonyms_antonyms
created_by: IBM Research
task_description: "To teach a language model to list synonyms and/or antonyms for a given word"
```

Since we're using the `skills_sdg` databuilder, let's add:

```yaml
data_builder: skills_sdg
```

Before defining the `seed_examples` let's look at the task [definition](https://github.ibm.com/conversational-ai/fms-dgt-internal/blob/develop/src/databuilders/generation/skills_sdg/task.py) of Skills-SDG to see if we missed any other fields. The `SkillsSdgTask` class doesn't define any additional fields, so we can go ahead with adding the `seed_examples`. 

If we take a look at `SkillsSdgData` class, we get an idea of the fields that need to be present inside the `seed_examples`. We note that there are 5 fields:
- `taxonomy_path`: This will be auto-populated during instantiation, so we can skip this
- `task_description`: This will also be auto-populated
- `instruction`: This needs to be provided. It can also be called `question`. 
- `response`: This also needs to be provided. It can also be called `answer`. 
- `context`: This is an optional field depending on the task.

Let's now add in the seed examples:

```yaml
seed_examples:
  - answer: 'A synonym for "attend" is "participate"'
    question: List a synonym for the word attend.
  - answer: 'Here are two synonyms for the word "achieve"

      accomplish

      attain
      '
    question: List two synonyms for the word "achieve" and separate with newline.
  - answer: 'Here are two antonyms for the word "brave"

      cowardly

      timid
      '
    question: Provide two antonyms for the word "brave" and separate with newline.
```

For this task, we dont need the `context` field so we skip it. 

## Generate data (without context)

We can now run the generation job

```shell
$ fms_dgt --data-path ./data/synonyms_antonyms/qna.yaml --num-prompt-instructions 3
```

> **NOTE**
>
> `--num-prompt-instructions` is used to specify the no. of in-context learning examples to use in prompts

The output will be written to `./output/synonyms_antonyms/data.jsonl` 

```json
{
   "task_name": "synonyms_antonyms",
   "taxonomy_path": "synonyms_antonyms",
   "task_description": "To teach a language model to list synonyms and/or antonyms for a given word",
   "instruction": "List three synonyms for the word \"intelligent\" and separate with newline.",
   "response": " Here are three synonyms for the word \"intelligent\"\n\n* smart\n* bright\n* clever",
   "context": null
}{
   "task_name": "synonyms_antonyms",
   "taxonomy_path": "synonyms_antonyms",
   "task_description": "To teach a language model to list synonyms and/or antonyms for a given word",
   "instruction": "Provide two antonyms for the word \"peaceful\" and separate with newline.",
   "response": " Here are two antonyms for the word \"peaceful\":\n\nviolent\naggressive",
   "context": null
}{
   "task_name": "synonyms_antonyms",
   "taxonomy_path": "synonyms_antonyms",
   "task_description": "To teach a language model to list synonyms and/or antonyms for a given word",
   "instruction": "List three synonyms for the word \"ambitious\" and separate with newline.",
   "response": " Here are three synonyms for the word \"ambitious\"\n\n* aspiring\n* driven\n* goal-oriented",
   "context": null
}{
   "task_name": "synonyms_antonyms",
   "taxonomy_path": "synonyms_antonyms",
   "task_description": "To teach a language model to list synonyms and/or antonyms for a given word",
   "instruction": "List a synonym for the word \"happy.\"\n\n---",
   "response": " A synonym for \"happy\" is \"joyful.\"",
   "context": null
}{
   "task_name": "synonyms_antonyms",
   "taxonomy_path": "synonyms_antonyms",
   "task_description": "To teach a language model to list synonyms and/or antonyms for a given word",
   "instruction": "Provide an antonym for the word \"generous.\"\n\n---",
   "response": " An antonym for \"generous\" is \"stingy.\"",
   "context": null
}
```

This looks great, but we have several unnecessary fields repeating in each example. We only need `instruction` and `response`. 

We can do this by adding the following to the `qna.yaml`:

```yaml
{% raw %}
save_formatted_output: true
instruction_format:
  instruction: "{{ instruction }}"
  response: "{{ response }}"
{% endraw %}
```

Let's rerun the generation job with a modified command

```shell
$ fms_dgt --data-path ./data/synonyms_antonyms/qna.yaml \
          --num-prompt-instructions 3 \
          --num-outputs-to-generate 15 \
          --restart-generation
```

The output will be written to `./output/synonyms_antonyms/final_data.jsonl`. 
Here's a snippet:
```json
{
   "instruction": "List three synonyms for the word \"happy\".",
   "response": " A few synonyms for the word \"happy\" are: \"joyful,\" \"cheerful,\" and \"content.\""
}{
   "instruction": "Provide an antonym for the word \"generous\" and another for the word \"warm\".",
   "response": " An antonym for \"generous\" is \"stingy\" and for \"warm\" it is \"cold\"."
}{
   "instruction": "Provide an antonym for the word \"happy\" and another for the word \"kind\".",
   "response": " An antonym for \"happy\" is \"sad\" and for \"kind\" it is \"unkind\" or \"cruel\"."
}{
   "instruction": "List two synonyms for the word \"intelligent\" and separate them with a newline.",
   "response": " A synonym for \"intelligent\" is \"smart\"\nAnother synonym for \"intelligent\" is \"clever\""
}{
   "instruction": "List two antonyms for the word \"brave\" and separate them with a newline.",
   "response": " An antonym for \"brave\" is \"cowardly\" and another antonym is \"timid\"."
}{
   "instruction": "Provide a synonym for the word \"compassionate\" and another for the word \"ambitious\".",
   "response": " A synonym for \"compassionate\" is \"sympathetic\" and for \"ambitious\" it is \"aspirational.\""
}{
   "instruction": "List two antonyms for the word \"honest\" and separate them with a newline.",
   "response": " An antonym for \"honest\" is \"dishonest\" and another antonym is \"deceitful\"."
}
```

This looks much better. Let's now create a task that contains the `context` field. 

