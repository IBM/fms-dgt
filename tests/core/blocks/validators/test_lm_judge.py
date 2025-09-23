# Third Party
import pytest

# Local
from fms_dgt.core.blocks.validators.lm_judge import LMJudgeValidator

GREEDY_CFG = {
    "lm_config": {
        "type": "watsonx",
        "decoding_method": "greedy",
        "temperature": 1.0,
        "max_tokens": 25,
        "model_id_or_path": "ibm/granite-8b-code-instruct",
    }
}


@pytest.mark.parametrize("model_backend", ["watsonx"])
def test_completion(model_backend):
    lm_judge = LMJudgeValidator(name=f"test_{model_backend}", **GREEDY_CFG)

    lm_inputs = [
        {
            "input": "Question: 1 + 1 = ?\nAnswer: ",
            "success_func": lambda x: any([num in x for num in ["2"]]),
        }
    ]
    lm_judge(lm_inputs)
    assert lm_inputs[0]["is_valid"], "Result should be true!"

    lm_inputs = [
        {
            "input": "Repeat the following exactly: 'this is true'\n",
            "success_func": lambda x: "false" in x,
        }
    ]
    lm_judge(lm_inputs)
    assert not lm_inputs[0]["is_valid"], "Result should be false!"

    lm_inputs = [
        {
            "input": "Is 'eat' a verb?\nRespond with 'yes' or 'no.\n",
            "success_func": lambda x: "yes" in x.lower(),
        }
    ]
    lm_judge(lm_inputs)
    assert lm_inputs[0]["is_valid"], "Result should be true!"
    assert (
        "yes" in lm_inputs[0]["result"].lower()
    ), "Result should contain the word 'yes'!"
