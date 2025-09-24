# Standard
from dataclasses import dataclass, fields
from importlib.metadata import version
from importlib.util import find_spec
from inspect import signature
from typing import Any, Dict, List, Literal, Tuple

# Third Party
from dotenv import load_dotenv
from packaging.version import parse as parse_version
from tqdm import tqdm
from transformers import AutoTokenizer

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters
from fms_dgt.core.blocks.llm.utils import Grouper, chunks, remap
from fms_dgt.utils import dgt_logger

try:
    # Third Party
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "attempted to use 'vllm' LM type, but package `vllm` not installed. ",
        "please install these via `pip install -r fms_dgt[vllm]`",
    ) from err


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass(kw_only=True)
class vLLMCompletionParameters(Parameters):
    n: int | NotGiven = 1
    presence_penalty: float | NotGiven = 0.0
    frequency_penalty: float | NotGiven = 0.0
    repetition_penalty: float | NotGiven = 1.0
    temperature: float | NotGiven = 1.0
    top_p: float | NotGiven = 1.0
    top_k: int | NotGiven = -1
    seed: int | NotGiven = NOT_GIVEN
    stop: List[str] | NotGiven = NOT_GIVEN
    max_tokens: int | NotGiven = 16
    min_tokens: int | NotGiven = 0
    logprobs: int | NotGiven = NOT_GIVEN

    @classmethod
    def from_dict(cls, params: Dict):
        # map everything to canonical form
        remaped_params = remap(
            dictionary=dict(params),
            mapping={
                "stop": ["stop_sequences"],
            },
        )

        # will filter out unused here
        field_names = [f.name for f in fields(cls)]
        eligible_params = {k: v for k, v in remaped_params.items() if k in field_names}

        return cls(**eligible_params)


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_block("vllm")
class vLLM(LMProvider):
    """vLLM Generator"""

    def __init__(
        self,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: str | NotGiven = NOT_GIVEN,
        trust_remote_code: bool | None = False,
        tokenizer: str | NotGiven = NOT_GIVEN,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: str | NotGiven = NOT_GIVEN,
        tensor_parallel_size: int = 1,
        quantization: str | NotGiven = NOT_GIVEN,
        swap_space: int | NotGiven = NOT_GIVEN,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        lora_local_path: str | NotGiven = NOT_GIVEN,
        **kwargs: Any,
    ):
        # Step 1: Verify necessary resources are available
        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install fms_dgt[vllm]`"
            )

        if device and "cuda" not in device:
            raise ValueError("vLLM only supports CUDA")

        if lora_local_path:
            if parse_version(version("vllm")) < parse_version("0.3.0"):
                raise ValueError("lora adapters only compatible with vllm > v0.3.0.")

        # Step 2: Intialize parent
        super().__init__(**kwargs)

        # Step 3: Load enviroment
        load_dotenv()

        # Step 4: Configure necessary variables
        if not self.batch_size:
            self._batch_size = "auto"

        self.tensor_parallel_size = int(tensor_parallel_size)
        self._lora_request = (
            LoRARequest("finetuned", 1, lora_local_path) if lora_local_path else None
        )

        # Step 4: Initialize model
        model_args = {
            "model": self.model_id_or_path,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "swap_space": int(swap_space) if swap_space is not NOT_GIVEN else NOT_GIVEN,
            "quantization": quantization,
            "seed": (int(self.random_seed) if self.random_seed is not NOT_GIVEN else NOT_GIVEN),
        }
        self.model = LLM(**{k: v for k, v in model_args.items() if v is not NOT_GIVEN})

        # Step 5: Initialize tokenizer
        self._tokenizer = get_tokenizer(
            tokenizer if tokenizer else self.model_id_or_path,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
        )

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def max_length(self):
        if self._parameters.max_length:  # if max length manually set, return it
            return self._parameters.max_length
        elif hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        else:
            return NOT_GIVEN

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_parameters(
        self, **kwargs
    ) -> Tuple[vLLMCompletionParameters, vLLMCompletionParameters]:
        return vLLMCompletionParameters.from_dict(kwargs), vLLMCompletionParameters.from_dict(
            kwargs
        )

    def init_tokenizer(self, model_id_or_path: str = None):
        """Initializes a tokenizer

        Args:
            model_id_or_path (str, optional): Model to be used for initializing tokenizer. Defaults to None.
        """
        try:
            return AutoTokenizer.from_pretrained(model_id_or_path or self.model_id_or_path)
        except (OSError, ValueError) as err:
            dgt_logger.warning(
                'Failed to initialize tokenizer for "%s" due to %s',
                model_id_or_path or self.model_id_or_path,
                err.args[0],
            )
            dgt_logger.warning(
                'Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
            )
            return None

    def _extract_token_log_probabilities(self, output: dict) -> List[Any] | None:
        return (
            [
                {token_prob.decoded_token: token_prob.logprob for token_prob in entry.values()}
                for entry in output.logprobs
            ]
            if output.logprobs
            else None
        )

    def completion(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> None:
        self._execute_requests(requests, disable_tqdm, method=self.COMPLETION)

    def chat_completion(self, requests: List[LMBlockData], disable_tqdm=False) -> None:
        # Verify we have the right structure
        if len(requests) > 0:
            if isinstance(requests[0].input, str):
                raise ValueError("chat() requires List[Dict] as input")

        self._execute_requests(requests, disable_tqdm, method=self.CHAT_COMPLETION)

    def _execute_requests(
        self, requests: List[LMBlockData], disable_tqdm: bool, method: str
    ) -> None:
        # Step 1: Group requests by their generation_kwargs
        grouper = Grouper(requests, lambda x: str(x.gen_kwargs))

        # Step 2: Initialize progress tracker
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc=f"Running {method} requests",
        )

        # Step 3: Iterate over each group
        for _, reqs in grouper.get_grouped().items():
            # Step 3.a: Create request chunks based on maximum allowed batch size
            batches: List[List[LMBlockData]] = chunks(
                reqs,
                n=(
                    1
                    if method == self.CHAT_COMPLETION
                    else int(self.batch_size) if self.batch_size != "auto" else 0
                ),
            )

            # Step 3.b: Iterate over each chunk
            for batch in batches:
                # Step 3.b.i: Fetch generation kwargs from 1st request since generation kwargs within a chunk are identical
                chunk = next(iter(batch))

                # Step 3.b.ii: Extend completion/ chat-completion parameters from gen_kwargs and instantiate SamplingParams
                params = (
                    self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
                ).to_params(chunk.gen_kwargs)

                sampling_params = SamplingParams(
                    **{
                        k: v
                        for k, v in params.items()
                        if k in list(signature(SamplingParams).parameters)
                    }
                )

                # Step 3.b.iii: Trigger vLLM generate function
                if method == self.CHAT_COMPLETION:
                    response = self.model.chat(
                        messages=self._prepare_input(
                            chunk,
                            method=method,
                            max_tokens=params.get("max_tokens", None),
                        ),
                        sampling_params=sampling_params,
                        use_tqdm=False,
                        lora_request=self._lora_request,
                        tools=chunk.tools,
                    )
                else:
                    response = self.model.generate(
                        prompts=[
                            self._prepare_input(
                                instance,
                                method=method,
                                max_tokens=params.get("max_tokens", None),
                            )
                            for instance in batch
                        ],
                        sampling_params=sampling_params,
                        use_tqdm=False,
                        lora_request=self._lora_request,
                    )

                # Step 3.b.iv: If multiple choices requested per input
                # Step 3.b.v.*: Get requested choices count from parameters
                n = params.get("n", 1)

                # Step 3.b.v.**: Verify enough responses are generated per input
                if sum([len(entry.outputs) for entry in response]) != n * len(batch):
                    raise RuntimeError(
                        f"Number of responses does not match number of inputs * n, [{sum([len(entry.outputs) for entry in response])}, {n}, {len(batch)}]"
                    )

                # Step 3.b.v.***: Iterate over each grouped response
                for response_per_input, instance in zip(response, batch):
                    outputs, addtl = [], {"token_logprobs": []}
                    for output in response_per_input.outputs:
                        if method == self.CHAT_COMPLETION:
                            outputs.append({"role": "assistant", "content": output.text})
                        else:
                            outputs.append(output.text)

                        token_logprobs = self._extract_token_log_probabilities(output)
                        if token_logprobs:
                            addtl["token_logprobs"].append(token_logprobs)

                    self.update_instance_with_result(
                        method,
                        outputs if len(outputs) > 1 else outputs[0],
                        instance,
                        stop=params.get("stop", None),
                        additional=addtl,
                    )

                    # Step 3.b.v.*****: Update progress counter
                    pbar.update(1)

        # Step 4: Cleanup
        pbar.close()
