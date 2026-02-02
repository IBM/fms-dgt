# Third Party
import pytest

# Local
from fms_dgt.public.blocks.validators.data_leakage.data_leakage import (
    DataLeakageValidator,
)

inp1 = "The brown fox jumps over the dog and runs into the dense forest"
inp2 = "A smart little dog watched the fox runs into the forest"
inp3 = "A smart little dog watched the angry and confused fox, who then somehow runs off slowly into a dark and mysterious forest."
inp4 = "The small and fragile package arrived at the warehouse. The delivery truck was not on time."

context1 = "The quick brown fox jumps over the smart little dog and runs into the dense forest without looking back."
context4 = "The delivery truck was late. The small and fragile package was sent yesterday. It arrived at the warehouse this morning."


@pytest.mark.parametrize("filter_flag", [False, True])
def test_local_context(filter_flag):
    inputs = [
        {"input": inp1, "local_context": [context1]},
        {"input": inp2, "local_context": [context1]},
        {"input": inp3, "local_context": [context1]},
        {"input": inp4, "local_context": [context4]},
        {"input": inp4, "local_context": [context4, context1]},
    ]

    data_leakage_val = DataLeakageValidator(
        name="test_data_leakage", threshold=0.55, filter=filter_flag
    )
    outputs = data_leakage_val(inputs)
    if filter_flag:
        assert len(outputs) == 1
        assert outputs[0]["input"] == inp3
    else:
        assert not outputs[0]["is_valid"]
        assert not outputs[1]["is_valid"]
        assert outputs[2]["is_valid"]
        assert not outputs[3]["is_valid"]
        assert not outputs[4]["is_valid"]


@pytest.mark.parametrize("filter_flag", [False, True])
def test_global_context(filter_flag):
    global_context = [context1, context4]
    inputs = [{"input": inp} for inp in [inp1, inp2, inp3, inp4]]

    data_leakage_val = DataLeakageValidator(
        name="test_data_leakage", threshold=0.55, filter=filter_flag
    )
    outputs = data_leakage_val(inputs, context=global_context)
    if filter_flag:
        assert len(outputs) == 1
        assert outputs[0]["input"] == inputs[2]["input"]
    else:
        assert not outputs[0]["is_valid"]
        assert not outputs[1]["is_valid"]
        assert outputs[2]["is_valid"]
        assert not outputs[3]["is_valid"]
