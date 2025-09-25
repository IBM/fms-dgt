# Standard
import json
import logging
import random
import time

# Third Party
import pytest

# Local
from fms_dgt.core.blocks.validators.rouge import RougeDedupValidator


def test_matches():

    all_data = [
        "I went to the store yesterday",
        "blue red yellow green",
    ]

    inputs = [
        {"input": "I went to the store"},
        {"input": "I went to the store yesterday"},
    ]
    validator = RougeDedupValidator(name="test_rouge_validator", fast_rouge=False, threshold=0.91)
    validator(inputs, context=all_data)
    assert inputs[0]["is_valid"] and not inputs[1]["is_valid"]

    inputs = [
        {"input": "I went to the store"},
        {"input": "I went to the store"},
        {"input": "I went to the store yesterday"},
    ]
    validator = RougeDedupValidator(name="test_rouge_validator", fast_rouge=False, threshold=1.0)
    validator(inputs, context=all_data)
    assert inputs[0]["is_valid"] and not inputs[1]["is_valid"] and not inputs[2]["is_valid"]

    validator = RougeDedupValidator(name="test_rouge_validator", fast_rouge=False, threshold=None)
    validator(inputs, context=all_data)
    assert inputs[0]["is_valid"] and inputs[1]["is_valid"] and inputs[2]["is_valid"]

    inputs = [{"input": "one two three"}]
    validator = RougeDedupValidator(name="test_rouge_validator", fast_rouge=False, threshold=0.0)
    validator(inputs, context=all_data)
    assert not inputs[0]["is_valid"]


@pytest.mark.parametrize(
    "max_len,max_shared_len,seq_ct",
    [
        (30, 10, 100),
        # (50, 10, 100),
        # (30, 10, 200),
        # (50, 10, 200),
    ],
)
def test_scaling(max_len, max_shared_len, seq_ct):
    def _gen_seqs():
        shared_len = rand_obj.randint(1, max_shared_len)
        shared_seq, ret_seqs = [str(i) for i in range(shared_len)], []
        for _ in range(seq_ct):
            seq_len = rand_obj.randint(1, max_len)
            inj_idx = rand_obj.randint(0, seq_len - 1)
            out_seq = [str(i) for i in range(shared_len, seq_len + max_len)]
            rand_obj.shuffle(out_seq)
            out_seq = out_seq[:inj_idx] + shared_seq + out_seq[inj_idx:]
            ret_seqs.append({"input": " ".join(out_seq)})
        return ret_seqs

    # To see logging, run with --log-cli-level=INFO
    logger = logging.getLogger("test_scaling")

    rand_obj = random.Random(0)

    to_validate = _gen_seqs()

    for thresh in [0.1, 0.2, 0.5, 0.7, 0.9, 1.0]:

        settings = {
            "threshold": thresh,
            "max length": max_len,
            "max_shared_length": max_shared_len,
            "sequence count": seq_ct,
        }

        logger.info("Running setting %s for standard rouge", json.dumps(settings))
        standard_time = time.time()
        validator = RougeDedupValidator(
            name="test_scaling", fast_rouge=False, threshold=thresh, filter=False
        )
        standard_res = validator(to_validate)
        standard_time = time.time() - standard_time

        logger.info("Running setting %s for fast rouge", json.dumps(settings))
        fast_time = time.time()
        validator = RougeDedupValidator(
            name="test_scaling", fast_rouge=True, threshold=thresh, filter=False
        )
        fast_res = validator(to_validate)
        fast_time = time.time() - fast_time

        filt_ct = len(standard_res) - sum([x["is_valid"] for x in standard_res])

        logger.info(
            "\n\n - Setting:\t\t%s\n - Filtered Outputs:\t%s\n - Standard Rouge:\t%s seconds\n - Fast Rouge:\t\t%s seconds\n",
            json.dumps(settings),
            filt_ct,
            standard_time,
            fast_time,
        )

        # check length
        assert len(standard_res) == len(
            fast_res
        ), f"Standard rouge produced {len(standard_res)} outputs while fast rouge produced {len(fast_res)} outputs"

        # check time
        assert (
            thresh == 1.0 or fast_time <= standard_time
        ), "Fast rouge did not result in a speed up over standard rouge"

        # check for identical results
        for sr, fr in zip(standard_res, fast_res):
            assert (
                sr["is_valid"] == fr["is_valid"]
            ), f"Different results for {sr['input']} and {fr['input']}"
