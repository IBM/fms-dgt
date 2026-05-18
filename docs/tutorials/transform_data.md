# Building a Transformation Databuilder

This tutorial walks through building a transformation databuilder from scratch. Unlike generation databuilders that loop until they produce enough synthetic examples, transformation databuilders make a single pass over an input dataset and produce a transformed version of it.

The domain used here is fact extraction: given short factual paragraphs, the builder extracts structured `(subject, predicate, object)` triples from each one. This is a genuine 1-to-N transformation where a single input paragraph may yield multiple output records.

## What you will build

A databuilder registered as `public/examples/fact_triples` that:

- Accepts input paragraphs of factual text
- Prompts a language model to extract structured triples from each paragraph
- Produces one output record per extracted triple

All files go under `fms_dgt/public/databuilders/examples/fact_triples/` and `tasks/public/examples/fact_triples/`.

## Step 1: Create the directory

```bash
mkdir -p fms_dgt/public/databuilders/examples/fact_triples
mkdir -p tasks/public/examples/fact_triples
mkdir -p data/public/examples/fact_triples
```

## Step 2: Define the data schema

Create `fms_dgt/public/databuilders/examples/fact_triples/data_objects.py`:

```{.python title="fms_dgt/public/databuilders/examples/fact_triples/data_objects.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.data_objects import DataPoint


@dataclass(kw_only=True)
class ParagraphData(DataPoint):
    """A paragraph of factual text to extract triples from."""

    paragraph: str


@dataclass(kw_only=True)
class TripleData(DataPoint):
    """A single subject-predicate-object triple extracted from a paragraph."""

    subject: str
    predicate: str
    obj: str
    source_paragraph: str
```

Notice that `INPUT_DATA_TYPE` and `OUTPUT_DATA_TYPE` are different here. `ParagraphData` comes in; `TripleData` comes out. This is the key difference from generation databuilders where the schema is typically the same on both sides.

The `source_paragraph` field on the output keeps a reference back to the input, which is useful for traceability.

## Step 3: Define the task

Create `fms_dgt/public/databuilders/examples/fact_triples/task.py`:

```{.python title="fms_dgt/public/databuilders/examples/fact_triples/task.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Local
from fms_dgt.base.task import TransformationTask
from fms_dgt.public.databuilders.examples.fact_triples.data_objects import (
    ParagraphData,
    TripleData,
)


class FactTriplesTask(TransformationTask):
    """
    Transformation task for extracting fact triples from paragraphs.

    TransformationTask makes a single pass over all input data.
    It always restarts with a clean slate because the transformation
    cardinality (one paragraph may yield many triples) means partial
    output cannot be safely resumed.
    """

    INPUT_DATA_TYPE = ParagraphData
    OUTPUT_DATA_TYPE = TripleData

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def instantiate_input_example(self, **kwargs: Any) -> ParagraphData:
        return ParagraphData(
            task_name=self.name,
            is_seed=False,
            paragraph=kwargs.get("paragraph"),
        )
```

`TransformationTask` is the base class for all transformation databuilders. It handles loading input data, iterating in batches, and writing output. You do not need to manage looping or stopping criteria.

## Step 4: Implement the transformation logic

Create `fms_dgt/public/databuilders/examples/fact_triples/generate.py`:

````{.python title="fms_dgt/public/databuilders/examples/fact_triples/generate.py"}
# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import json

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.public.databuilders.examples.fact_triples.data_objects import (
    ParagraphData,
    TripleData,
)
from fms_dgt.public.databuilders.examples.fact_triples.task import FactTriplesTask


@register_data_builder("public/examples/fact_triples")
class FactTriplesDataBuilder(TransformationDataBuilder):
    """Extracts subject-predicate-object triples from factual paragraphs."""

    TASK_TYPE: FactTriplesTask = FactTriplesTask

    extractor: LMProvider

    _PROMPT = JinjaPromptTemplate(
        template=(
            "Extract all factual subject-predicate-object triples from the paragraph below. "
            "Return a JSON array where each element is an object with keys "
            '"subject", "predicate", and "object". '
            "Only include triples that are directly stated in the paragraph. "
            "Do not infer or add information.\n\n"
            "Paragraph: {{ paragraph }}\n\n"
            "Triples (JSON array):"
        )
    )

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __call__(self, data_points: List[ParagraphData]) -> List[TripleData]:
        # Build one LM request per paragraph.
        extractor_inputs: List[Dict] = []
        for data_point in data_points:
            extractor_inputs.append(
                {
                    "input": [
                        {
                            "role": "user",
                            "content": self._PROMPT.encode(
                                render_dict={"paragraph": data_point.paragraph}
                            ),
                        }
                    ],
                    # Pass the full data point through so we can reference it in outputs.
                    "reference": data_point,
                    # task_name: not used by the block; included so DiGiT can attribute
                    #   token usage to this task in traces.jsonl.
                    "task_name": data_point.task_name,
                }
            )

        extractor_outputs = self.extractor(extractor_inputs, method="chat_completion")

        outputs = []
        for extractor_output in extractor_outputs:
            data_point: ParagraphData = extractor_output["reference"]

            result = extractor_output["result"]
            if isinstance(result, dict):
                result = result.get("content") or ""

            triples = self._parse(result)
            for triple in triples:
                outputs.append(
                    TripleData(
                        task_name=data_point.task_name,
                        is_seed=False,
                        subject=triple.get("subject", ""),
                        predicate=triple.get("predicate", ""),
                        obj=triple.get("object", ""),
                        source_paragraph=data_point.paragraph,
                    )
                )

        return outputs

    def _parse(self, response: str) -> List[Dict]:
        """Parse a JSON array of triples from the LM response."""
        # Strip markdown code fences if present.
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return []
````

Three things worth noting:

- `TransformationDataBuilder.__call__` receives a batch of input records and returns a list of output records. The batch size is controlled by `transform_batch_size` in the runner config (defaults to the full dataset).
- Because one paragraph can produce multiple triples, the output list may be longer than the input list. This is the 1-to-N transformation pattern.
- The `task_name` key in each block input dict is a telemetry convention. The block ignores it, but DiGiT reads it to attribute token usage to the correct task in `traces.jsonl`. Always include it as the last key in every block input dict you build.

## Step 5: Write the builder config

Create `fms_dgt/public/databuilders/examples/fact_triples/fact_triples.yaml`:

```{.yaml title="fms_dgt/public/databuilders/examples/fact_triples/fact_triples.yaml"}
######################################################
#                   MANDATORY FIELDS
######################################################
name: public/examples/fact_triples

######################################################
#                   RESERVED FIELDS
######################################################
blocks:
  - name: extractor
    type: ollama
    model_id_or_path: granite4:3b
    temperature: 0.0
    max_tokens: 512
    num_ctx: 4096
metadata:
  version: 1.0
```

Temperature is set to 0.0 because triple extraction is a deterministic structured task: we want the model to extract what is there, not generate creatively.

## Step 6: Prepare the input data

Create `data/public/examples/fact_triples/paragraphs.jsonl` with a few short factual paragraphs:

```{.json title="data/public/examples/fact_triples/paragraphs.jsonl"}
{"paragraph": "The Amazon River is the largest river in the world by discharge volume. It flows through Brazil, Peru, and Colombia before emptying into the Atlantic Ocean. The river basin covers approximately 7 million square kilometers."}
{"paragraph": "The speed of light in a vacuum is approximately 299,792 kilometers per second. Nothing with mass can travel at or beyond this speed. Light takes about 8 minutes and 20 seconds to travel from the Sun to Earth."}
{"paragraph": "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences, Physics in 1903 and Chemistry in 1911."}
{"paragraph": "The Python programming language was created by Guido van Rossum and first released in 1991. Python emphasizes code readability and uses significant indentation. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."}
{"paragraph": "Mount Everest is the highest mountain on Earth above sea level, with a peak elevation of 8,848.86 meters. It is located in the Himalayas on the border between Nepal and the Tibet Autonomous Region of China."}
```

## Step 7: Create the task file

Create `tasks/public/examples/fact_triples/task.yaml`:

```{.yaml title="tasks/public/examples/fact_triples/task.yaml"}
######################################################
#                   MANDATORY FIELDS
######################################################
task_name: public/examples/fact_triples
task_description: Extract subject-predicate-object triples from factual paragraphs.
created_by: IBM

data_builder: public/examples/fact_triples

######################################################
#                   RESERVED FIELDS
######################################################
data:
  type: default
  data_path: ${DGT_DATA_DIR}/public/examples/fact_triples/paragraphs.jsonl
```

The `data` field points to the input dataset. `${DGT_DATA_DIR}` resolves to the `data/` directory at the root of the repository by default.

## Step 8: Run it

```bash
python -m fms_dgt.public \
  --task-paths ./tasks/public/examples/fact_triples/task.yaml \
  --restart
```

Generated data is written to `output/public/examples/fact_triples/final_data.jsonl`. Each record is one extracted triple:

```json
{"task_name": "public/examples/fact_triples", "is_seed": false, "subject": "Amazon River", "predicate": "is the largest river in the world by", "obj": "discharge volume", "source_paragraph": "The Amazon River is the largest river in the world by discharge volume..."}
{"task_name": "public/examples/fact_triples", "is_seed": false, "subject": "Amazon River", "predicate": "flows through", "obj": "Brazil, Peru, and Colombia", "source_paragraph": "The Amazon River is the largest river in the world by discharge volume..."}
{"task_name": "public/examples/fact_triples", "is_seed": false, "subject": "Amazon River", "predicate": "empties into", "obj": "Atlantic Ocean", "source_paragraph": "The Amazon River is the largest river in the world by discharge volume..."}
```

Notice that the single first paragraph produced three triples. This is the 1-to-N transformation in action.

## Next steps

- To switch to a cloud provider, see [Changing the LM Engine](../tutorials/changing_lm_engine.md).
- To build a generation databuilder, see [Building a Generation Databuilder](generate_data.md).
- To see a built-in transformation example you can run immediately, see [Data Transformation: Rating QA Pairs](../usage.md#rate-the-generated-data).
