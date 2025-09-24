# Standard
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Tuple
import asyncio
import logging

# Third Party
from httpx import HTTPError
from ollama import chat, generate, show
from tqdm import tqdm

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters
from fms_dgt.core.blocks.llm.openai import OpenAI
from fms_dgt.core.blocks.llm.utils import remap

# Disable third party logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
@dataclass(kw_only=True)
class OllamaCompletionParameters(Parameters):
    num_predict: int | NotGiven = NOT_GIVEN
    seed: int | NotGiven = NOT_GIVEN
    top_p: int | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    presence_penalty: float | NotGiven = NOT_GIVEN
    frequency_penalty: float | NotGiven = NOT_GIVEN
    stop: List[str] | NotGiven = NOT_GIVEN

    @classmethod
    def from_dict(cls, params: Dict):
        # map everything to canonical form
        remaped_params = remap(
            dictionary=dict(params),
            mapping={
                "num_predict": [
                    "max_tokens",
                ],
                "stop": ["stop_sequences"],
            },
        )

        # will filter out unused here
        field_names = [f.name for f in fields(cls)]
        eligible_params = {k: v for k, v in remaped_params.items() if k in field_names}

        return cls(**eligible_params)


@dataclass(kw_only=True)
class OllamaChatCompletionParameters(OllamaCompletionParameters):
    response_format: dict | NotGiven = NOT_GIVEN


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_block("ollama")
class Ollama(OpenAI):
    def __init__(
        self,
        base_url: str,
        **kwargs: Any,
    ):
        # Step 1: Initialize parent
        super().__init__(
            base_url=f"{base_url}/v1",
            api_key="ollama",
            **kwargs,
        )

        # Step 2: Set batch size, if None
        if not self.batch_size or self.batch_size > 1:
            self._batch_size = 1

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def max_length(self):
        # If max length manually set, return it
        if self._parameters.max_length:
            return self._parameters.max_length
        else:
            # Try auto-detecting max-length from the /v1/models API
            try:
                response = show(model=self.model_id_or_path)
                if response.modelinfo:
                    try:
                        return [v for k, v in response.modelinfo.items() if "context_length" in k][
                            0
                        ]
                    except (KeyError, IndexError):
                        return NOT_GIVEN
                else:
                    return NOT_GIVEN
            except HTTPError:
                return NOT_GIVEN

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_parameters(
        self, **kwargs
    ) -> Tuple[OllamaCompletionParameters, OllamaChatCompletionParameters]:
        return OllamaCompletionParameters.from_dict(
            kwargs
        ), OllamaChatCompletionParameters.from_dict(kwargs)

    def init_tokenizer(self, model_id_or_path: str = None):
        """Initializes a tokenizer

        Args:
            model_id_or_path (str, optional): Model to be used for initializing tokenizer. Defaults to None.
        """
        raise NotImplementedError(
            'Tokenization support is disabled for "Ollama". Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
        )

    def _extract_choice_content(self, choice: Any, method: str) -> str | Dict:
        # If choice is generated via chat completion
        if method == self.CHAT_COMPLETION:
            return choice.message.dict()

        # If choice is generated via text completion
        return choice.response

    async def async_executor(
        self,
        queue: asyncio.Queue,
        update_progress_tracker: Callable,
        method: str = LMProvider.COMPLETION,
    ):
        """
        Execute text completion or chat completion asynchronously.

        Args:
            queue (asyncio.Queue): instances to complete.
            update_progress_tracker (Callable): progress tracker update function
            method (str, optional): Either "completion" or "chat_completion". Defaults to LMProvider.COMPLETION.

        Raises:
            ValueError: If method outside "completion" or "chat_completion" is passed
            RuntimeError: If number of responses does not match number of inputs * n
        """
        while not queue.empty():
            # Get a "work item" out of the queue.
            instance = await queue.get()

            # Extract parameters
            params = (
                self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
            ).to_params(instance.gen_kwargs)

            # Trigger OpenAI completion functions
            if method == self.CHAT_COMPLETION:
                response = chat(
                    model=self.model_id_or_path,
                    messages=self._prepare_input(
                        instance,
                        method=method,
                        max_tokens=params.get("num_predict", None),
                    ),
                    tools=instance.tools,
                    options=params,
                    stream=False,
                )
            elif method == self.COMPLETION:
                response = generate(
                    model=self._model_id_or_path,
                    prompt=self._prepare_input(
                        instance,
                        method=self.COMPLETION,
                        max_tokens=params.get("num_predict", None),
                    ),
                    options=params,
                    stream=False,
                )
            else:
                raise ValueError(
                    f'Unsupported method ({method}). Only "{self.COMPLETION}" or "{self.CHAT_COMPLETION}" values are allowed.'
                )

            # Add output
            self.update_instance_with_result(
                method,
                self._extract_choice_content(response, method=method),
                instance,
                params.get("stop", None),
                {
                    "completion_tokens": response.eval_count,
                    "prompt_tokens": response.prompt_eval_count,
                    "token_logprobs": [],
                },
            )

            # Notify the queue that the "work item" has been processed.
            queue.task_done()

            # Update progress tracker
            update_progress_tracker()

    async def _execute_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        method: str = LMProvider.COMPLETION,
        **kwargs,
    ):
        # Initialize necessary variables
        queue = asyncio.Queue()

        # Add to queue
        for instance in requests:
            queue.put_nowait(instance)

        # Initialize progress tracker
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc=f"Running {method} requests",
        )

        # Create generate tasks
        executors = []
        for _ in range(min(queue.qsize(), self._call_limit)):
            executors.append(
                self.async_executor(
                    queue,
                    update_progress_tracker=lambda: pbar.update(1),
                    method=method,
                )
            )

        # Wait until all worker are finished
        await asyncio.gather(*executors, return_exceptions=True)

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def completion(self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs) -> None:
        asyncio.run(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.COMPLETION,
                **kwargs,
            )
        )

    def chat_completion(
        self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs
    ) -> None:
        asyncio.run(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.CHAT_COMPLETION,
                **kwargs,
            )
        )
