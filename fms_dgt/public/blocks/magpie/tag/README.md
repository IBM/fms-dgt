# Magpie Tagging Block

Modified version of [**Magpie**](https://magpie-align.github.io/) to enable working with opensource models.

It generates scores and tags using the specified model as the teacher (generator) and prompt templates.

### Format of Data

The data should have "input", "output" field or "messages" field which is a list of dictionaries with alternating

```
[{'role': 'user', 'content':'something'}, {'role':'assistant', 'content':'something'}]
```

If there is a "messages" field then it will ignore the "input" and "output" field and tag

### Explanation

Tagging the input & output (in case of single turn) or the conversation (in case of multi turn) in terms of :

### Prompt Design

The prompt templates in this block deviate from the originals in the Magpie paper and repository. They have been revised to improve JSON output reliability across instruction-tuned models. The key principles applied:

- The output contract ("Output only a valid JSON object, no other text.") appears at the top of the prompt, before the input content. Models are more likely to honour a format constraint they see before reading the content they need to reason about.
- Reasoning guidance is framed as instructions for filling specific JSON fields, not as a free-form reasoning step before the JSON. This prevents models from narrating their reasoning in prose and never producing the JSON.
- Placeholder syntax inside JSON template values uses natural-language descriptions (e.g., `"difficulty": "very easy/easy/medium/hard/very hard"`) rather than bracket-style markers (e.g., `[very easy/easy/medium/hard/very hard]`). Bracket markers cause some models to preserve the brackets literally in their output, breaking JSON parsing.

These changes reduce tagging failure rates from ~48% to ~1-2% on typical instruction-tuned models.

### Tags

```
quality (question) : [
"very poor",
"poor",
"average",
"good",
"excellent",
]

sample_quality score(question and response) : ["1", "2", "3", "4", "5"]

difficulty : [
"very easy",
"easy",
"medium",
"hard",
"very hard",
]

classification of task: []

```
