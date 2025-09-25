# Standard
from dataclasses import asdict, fields
from typing import Any, Callable, Dict, List, Set, Tuple
import asyncio
import logging

# Third Party
from ibm_watsonx_ai.foundation_models.schema import TextGenDecodingMethod
from tqdm import tqdm

# Local
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.core.blocks.llm import (
    LMBlockData,
    LMProvider,
    ToolChoice,
)
from fms_dgt.core.resources.watsonx import WatsonXResource
from fms_dgt.utils import dgt_logger
import fms_dgt.core.blocks.llm.utils as generator_utils

try:
    # Third Party
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference as Model
    from ibm_watsonx_ai.foundation_models.schema import (
        TextChatParameters,
        TextGenParameters,
    )
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "attempted to use 'watsonx' LM type, but package `ibm_watsonx_ai` not installed. ",
    ) from err


# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_block("watsonx")
class WatsonXAI(LMProvider):
    """WatsonX AI Generator"""

    def __init__(self, *args: Any, call_limit: int = 10, **kwargs: Any):
        # Adjust call limit, if necessary
        if call_limit is not None and not (call_limit > 0 and call_limit <= 10):
            dgt_logger.warning(
                'Number of simultaneous calls ("call_limit") cannot exceed 10 as per WatsonX.AI terms of conditions. Thus, restricting "call_limit" to 10.',
            )
            call_limit = 10

        # Load WatsonX Resource
        self._watsonx_resource: WatsonXResource = get_resource("watsonx", call_limit=call_limit)

        # Intialize parent
        super().__init__(*args, **kwargs)

        # Set batch size, if not defined
        if not self._batch_size:
            self._batch_size = 10

        # Configure credentials for WatsonX AI service
        if self._watsonx_resource.token:
            self._credentials = Credentials(
                url=self._watsonx_resource.url, token=self._watsonx_resource.token
            )
        else:
            self._credentials = Credentials(
                url=self._watsonx_resource.url, api_key=self._watsonx_resource.key
            )

        # NOTE: WatsonX chat completion only support greedy sampling (temperature=0) with n=1
        if (
            "temperature" in self._chat_parameters and self._chat_parameters["temperature"] == 0.0
        ) and ("n" in self._chat_parameters and self._chat_parameters["n"] > 1):
            dgt_logger.warning(
                'Defaulting "n=1" as per WatsonX.AI\'s chat completion API guidance when using greedy sampling (temperature=0)'
            )
            self._chat_parameters["n"] = 1

        # initialize model
        self._model = Model(
            model_id=self.model_id_or_path,
            credentials=self._credentials,
            project_id=self._watsonx_resource.project_id,
        )

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_parameters(self, **kwargs) -> Tuple[Dict, Dict]:
        # extract parameters from kwargs
        adjusted_kwargs = self._adjust_completion_parameters(kwargs)
        return (
            {
                k: v
                for k, v in adjusted_kwargs.items()
                if k in [field.name for field in fields(TextGenParameters)]
            },
            {
                k: v
                for k, v in adjusted_kwargs.items()
                if k in [field.name for field in fields(TextChatParameters)]
            },
        )

    def init_tokenizer(self, model_id_or_path: str = None):
        """Initializes a tokenizer

        Args:
            model_id_or_path (str, optional): Model to be used for initializing tokenizer. Defaults to None.
        """
        raise NotImplementedError("WatsonX.AI does not support tokenizer initialization")

    def _prepare_input(
        self,
        instance: LMBlockData,
        gen_kwargs: dict,
        method: str,
    ):
        # Step 1: Initialize necessary variables
        prepared_input: str | Dict[str, Any] = None

        # Step 2: Process based in method
        # Step 2.a: "chat_completion" method
        if method == self.CHAT_COMPLETION:
            prepared_input = instance.input

        # Step 2.b: "completion" method
        elif method == self.COMPLETION:
            # Step 2.b.i: string input
            if isinstance(instance.input, str):
                prepared_input = instance.input + (
                    instance.continuation if instance.continuation else ""
                )
            else:
                raise ValueError(
                    f'Unsupported type ({type(instance.input)}) for "LMBlockData.input". Only string is allowed as "LMBlockData.input" when using "{self.COMPLETION}" method.',
                )

        else:
            raise ValueError(
                f"Unsupported method ({method}). Please use one of the folllowing. {self.COMPLETION}, {self.CHAT_COMPLETION}.",
            )

        # Step 3: Return
        return prepared_input

    def _adjust_completion_parameters(self, params: dict) -> dict:
        # Step 1: Remap known parameters
        params = generator_utils.remap(
            dictionary=params,
            mapping={
                "max_new_tokens": ["max_tokens", "max_completion_tokens"],
                "stop_sequences": ["stop"],
                "random_seed": ["seed"],
                "time_limit": ["timeout"],
            },
            override=False,
        )

        # Step 2: Adjust "decoding_method" parameter, if necessary
        params["decoding_method"] = (
            TextGenDecodingMethod.GREEDY
            if params.get("temperature") <= 0.0
            else TextGenDecodingMethod.SAMPLE
        )

        # Step 3: Return relevant parameters
        return {
            k: v
            for k, v in params.items()
            if k in [field.name for field in fields(TextGenParameters)]
        }

    def _adjust_chat_completion_parameters(self, params: dict, raised: Set[str] = set()) -> dict:
        # Step 1: Remap known parameters
        params = generator_utils.remap(
            dictionary=params,
            mapping={
                "max_tokens": ["max_new_tokens", "max_completion_tokens"],
                "stop": ["stop_sequences"],
                "seed": ["random_seed"],
                "time_limit": ["timeout"],
            },
            override=False,
        )

        # Step 2: Adjust parameters
        # Step 2.a: WatsonX chat completion only support greedy sampling (temperature=0) with n=1
        if (
            "temperature" in params
            and params["temperature"] == 0
            and "n" in params
            and params["n"] > 1
        ):
            # Step 2.a.i: Record warning, if necessary
            warning_msg = 'Defaulting "n=1" as per WatsonX.AI\'s chat completion API guidance when using greedy sampling (temperature=0)'
            if warning_msg not in raised:
                dgt_logger.warning(warning_msg)
                raised.add(warning_msg)

            # Step 2.a.ii: Reset 'n' to 1
            params["n"] = 1

        # Step 3: Return relevant parameters
        return {
            k: v
            for k, v in params.items()
            if k in [field.name for field in fields(TextChatParameters)]
        }

    def _extract_token_log_probabilities(self, choice, method: str) -> List[Any] | None:
        if method == self.COMPLETION:
            return (
                [
                    {
                        top_candidate_token["text"]: top_candidate_token["logprob"]
                        for top_candidate_token in generated_token["top_tokens"]
                        if "logprob" in top_candidate_token
                    }
                    for generated_token in choice["generated_tokens"]
                    if generated_token
                ]
                if "generated_tokens" in choice and choice["generated_tokens"]
                else None
            )

        elif method == self.CHAT_COMPLETION:
            return (
                [
                    {
                        top_token_logprobs["token"]: top_token_logprobs["logprob"]
                        for top_token_logprobs in token_logprobs["top_logprobs"]
                        if "logprob" in top_token_logprobs
                    }
                    for token_logprobs in choice["logprobs"]["content"]
                    if token_logprobs
                ]
                if "logprobs" in choice
                and choice["logprobs"]
                and "content" in choice["logprobs"]
                and choice["logprobs"]["content"]
                else None
            )

        else:
            raise ValueError(
                f"Unsupported method ({method}). Please use one of the allowed values: {self.COMPLETION}, {self.CHAT_COMPLETION}."
            )

    async def async_chat_executor(
        self,
        queue: asyncio.Queue,
        update_progress_tracker: Callable,
    ):
        """
        Execute chat completion asynchronously.


        Args:
            queue (asyncio.Queue): instances to complete.
            update_progress_tracker (Callable): progress tracker update function
        """
        while not queue.empty():
            # Step 1: Get a "work item" out of the queue.
            instance = await queue.get()

            # Step 2: Fetch generation kwargs from 1st request since generation kwargs within a chunk are identical
            gen_kwargs = instance.gen_kwargs

            # Step 3: Extract completion parameters from gen_kwargs
            gen_kwargs = self._adjust_chat_completion_parameters(
                params={**self._chat_parameters, **gen_kwargs}
            )

            # adding this here to simplify downstream processing
            if gen_kwargs.get("logprobs") and gen_kwargs.get("top_logprobs") is None:
                gen_kwargs["top_logprobs"] = 1

            # Step 4: Trigger WatsonX.AI chat completion functions
            response = await self._model.achat(
                messages=self._prepare_input(
                    instance, gen_kwargs=gen_kwargs, method=self.CHAT_COMPLETION
                ),
                tools=instance.tools,
                tool_choice=(
                    asdict(instance.tool_choice)
                    if instance.tools and isinstance(instance.tool_choice, ToolChoice)
                    else None
                ),
                tool_choice_option=(
                    instance.tool_choice
                    if instance.tools and isinstance(instance.tool_choice, str)
                    else "none"
                ),
                params=TextChatParameters(**gen_kwargs),
            )

            # Step 5: If multiple choices requested per input
            # Step 5.a: Get requested choices count from parameters
            n = gen_kwargs.get("n", 1)

            # Step 5.b: Verify enough responses are generated per input
            if len(response["choices"]) != n:
                raise RuntimeError(
                    f"Number of responses ({len(response['choices'])}) does not match number of inputs (1) * n ({n})"
                )

            # Step 6: Iterate over each grouped response
            outputs = []
            addtl = {
                "completion_tokens": response["usage"]["completion_tokens"],
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "token_logprobs": [],
            }
            for choice in response["choices"]:
                outputs.append(choice["message"])

                token_logprobs = self._extract_token_log_probabilities(
                    choice=choice, method=self.CHAT_COMPLETION
                )
                if token_logprobs:
                    addtl["token_logprobs"].append(token_logprobs)

            self.update_instance_with_result(
                self.CHAT_COMPLETION,
                outputs if len(outputs) > 1 else outputs[0],
                instance,
                stop=gen_kwargs.get("stop", None),
                additional=addtl,
            )

            # Step 7: Notify the queue that the "work item" has been processed.
            queue.task_done()

            # Step 8: Update progress tracker
            update_progress_tracker()

    async def _execute_chat_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        **kwargs,
    ):
        # Step 1: Initialize necessary variables
        queue = asyncio.Queue()
        for request in requests:
            queue.put_nowait(request)

        # Step 2: Initialize progress tracker
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc=f"Running {self.CHAT_COMPLETION} requests",
        )

        # Step 3: Create generate tasks
        executors = []
        for _ in range(min(queue.qsize(), self._watsonx_resource.max_calls)):
            executors.append(
                self.async_chat_executor(
                    queue,
                    update_progress_tracker=lambda: pbar.update(1),
                )
            )

        # Step 4: Wait until all worker are finished
        await asyncio.gather(*executors, return_exceptions=True)

    def completion(self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs) -> None:
        # group requests by their generation_kwargs
        grouper = generator_utils.Grouper(requests, lambda x: str(x.gen_kwargs))

        # initialize progress tracker
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate requests",
        )

        # iterate over each group
        for _, reqs in grouper.get_grouped().items():
            # create request chunks based on maximum allowed batch size
            chunks: List[List[LMBlockData]] = generator_utils.chunks(
                reqs, n=min(self._watsonx_resource.max_calls, self.batch_size)
            )

            # Step 3.b: Iterate over each chunk
            for chunk in chunks:
                # fetch generation kwargs from 1st request since generation kwargs within a chunk are identical
                gen_kwargs = next(iter(chunk)).gen_kwargs

                # extract completion parameters from gen_kwargs
                gen_kwargs = self._adjust_completion_parameters(
                    params={**self._parameters, **gen_kwargs}
                )

                # adding this here to simplify downstream processing
                if gen_kwargs.get("logprobs") and gen_kwargs.get("top_logprobs") is None:
                    gen_kwargs["top_logprobs"] = 1

                # Step 3.b.vi: Execute generation routine
                responses = self._model.generate(
                    prompt=[
                        self._prepare_input(instance, gen_kwargs=gen_kwargs, method=self.COMPLETION)
                        for instance in chunk
                    ],
                    params=TextGenParameters(
                        **gen_kwargs,
                    ),
                )

                # Step 3.b.vii: Process generated outputs
                for idx, instance in enumerate(chunk):
                    addtl = {
                        "completion_tokens": responses[idx]["results"][0]["generated_token_count"],
                        "prompt_tokens": responses[idx]["results"][0]["input_token_count"],
                        "token_logprobs": [],
                    }
                    token_logprobs = self._extract_token_log_probabilities(
                        choice=responses[idx]["results"][0],
                        method=self.COMPLETION,
                    )
                    if token_logprobs:
                        addtl["token_logprobs"].append(token_logprobs)

                    self.update_instance_with_result(
                        self.COMPLETION,
                        responses[idx]["results"][0]["generated_text"],
                        instance,
                        gen_kwargs.get("stop_sequences", None),
                        additional=addtl,
                    )

                # Step 3.b.viii: Update progress counter
                pbar.update(1)

        # Step 4: Cleanup
        pbar.close()

    def chat_completion(
        self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs
    ) -> None:
        asyncio.run(
            self._execute_chat_requests(requests=requests, disable_tqdm=disable_tqdm, **kwargs)
        )
