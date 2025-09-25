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
