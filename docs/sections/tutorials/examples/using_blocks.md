# Testing Out Blocks in Isolation

While blocks would typically be used within the operation of a databuilder, the use of dynamic importing within DGT means that each block can be called independently of the framework (i.e., the dependencies required to run databuilders won't be loaded unless explicitly requested). A consequence of this is that it is easy to test how blocks operate in isolation from the rest of the framework.

## Prerequisites

To successfully run this, you will need to have completed the following:

1. Followed the [installation guide](../../getting_started/installation.md) to set up your virtual environment
2. Read through the [Blocks](../../key_concepts//blocks.md) section

## Example

To see how we can test a specific block, create a file `test.py` in a directory and ensure `fms-dgt/fms_dgt` is accessible within your python path. In `test.py`, write the following:

```python
import json
from fms_dgt.granite.blocks.llm.rits import RITS

model_cfg = {
    "type": "rits",
    "model_id_or_path": "mistralai/mixtral-8x7B-instruct-v0.1",
    "base_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01/v1",
    "decoding_method": "greedy",
    "temperature": 1.0,
    "max_new_tokens": 5,
    "min_new_tokens": 1,
}

lm = RITS(**model_cfg)

inputs = [{"input": "3 + 4 = "}, {"input": "4 + 5 = "}]
results = lm(inputs)
for res in results:
    print(json.dumps(res, indent=4))
```

To execute, call `python test.py` to see the results