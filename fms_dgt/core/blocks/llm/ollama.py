# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Tuple
import logging

# Third Party
from httpx import HTTPError
from ollama import AsyncClient, Client, show

# Local
from fms_dgt.base.block import get_row_name
from fms_dgt.base.registry import register_block
from fms_dgt.base.telemetry import (
    payload_max_chars,
    payload_recording_enabled,
    record_llm_payload,
)
from fms_dgt.constants import NOT_GIVEN, NotGiven
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider, Parameters
from fms_dgt.core.blocks.llm.executor import AsyncLLMExecutor
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
    num_ctx: int | NotGiven = NOT_GIVEN
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
        base_url: str | None = None,
        **kwargs: Any,
    ):
        # Initialize parent
        super().__init__(
            base_url=base_url,
            api_key="ollama",
            **kwargs,
        )

        # Set batch size, if None
        if not self.batch_size or self.batch_size > 1:
            self._batch_size = 1

        # Sync client for non-async use; async client for the executor path
        self._client = Client(host=base_url)
        self._async_client = AsyncClient(host=base_url)

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
            return choice.message.model_dump()

        # If choice is generated via text completion
        return choice.response

    async def _process_item(
        self,
        instance: LMBlockData,
        update_progress: Callable,
        method: str = LMProvider.COMPLETION,
    ):
        """Process one Ollama request using the native async client.

        Uses ``self._async_client`` so parallel execution actually happens.
        The sync ``self._client`` is retained for non-async use cases only.

        Args:
            instance: The request to process.
            update_progress: Zero-argument callable to advance the progress bar.
            method: ``LMProvider.COMPLETION`` or ``LMProvider.CHAT_COMPLETION``.
        """
        params = (
            self._chat_parameters if method == self.CHAT_COMPLETION else self._parameters
        ).to_params(instance.gen_kwargs)

        _task_name = get_row_name(instance)
        async with self._llm_span(
            method=method,
            batch_size=1,
            params=params,
            task_names=[_task_name] if _task_name is not None else None,
        ) as span_attrs:
            if method == self.CHAT_COMPLETION:
                messages = self._prepare_input(
                    instance,
                    method=method,
                    max_tokens=params.get("num_predict", None),
                )
                # Map response_format (OpenAI convention) to Ollama's native
                # format parameter.  Ollama accepts either the string "json"
                # (JSON mode, no schema) or a raw JSON schema dict (structured
                # outputs).  The OpenAI wrapper is never passed through as-is.
                rf = params.pop("response_format", None)
                ollama_format = None
                if rf is not None:
                    t = rf.get("type")
                    if t == "json_schema":
                        ollama_format = rf.get("json_schema", {}).get("schema", {})
                    elif t == "json_object":
                        ollama_format = "json"
                    # type == "text" → leave ollama_format as None
                response = await self._async_client.chat(
                    model=self.model_id_or_path,
                    messages=messages,
                    tools=instance.tools,
                    options=params,
                    format=ollama_format,
                    stream=False,
                )
            elif method == self.COMPLETION:
                prompt = self._prepare_input(
                    instance,
                    method=self.COMPLETION,
                    max_tokens=params.get("num_predict", None),
                )
                response = await self._async_client.generate(
                    model=self._model_id_or_path,
                    prompt=prompt,
                    # options=params,
                    stream=False,
                )
            else:
                raise ValueError(
                    f'Unsupported method ({method}). Only "{self.COMPLETION}" or "{self.CHAT_COMPLETION}" values are allowed.'
                )

            span_attrs["prompt_tokens"] = response.prompt_eval_count or 0
            span_attrs["completion_tokens"] = response.eval_count or 0

            if payload_recording_enabled():
                if method == self.CHAT_COMPLETION:
                    rc = response.message.dict()
                else:
                    rc = response.response or ""
                record_llm_payload(
                    span_attrs,
                    method,
                    payload_max_chars(),
                    prompt=prompt if method == self.COMPLETION else None,
                    messages=messages if method == self.CHAT_COMPLETION else None,
                    response_completion=rc,
                )

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

        update_progress(1)

    async def _execute_requests(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        method: str = LMProvider.COMPLETION,
        **kwargs,
    ):
        await AsyncLLMExecutor.run(
            work_items=requests,
            process_item=lambda item, upd: self._process_item(item, upd, method=method),
            call_limit=self._call_limit,
            total_requests=len(requests),
            method=method,
            disable_tqdm=disable_tqdm,
        )

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def completion(self, requests: List[LMBlockData], disable_tqdm: bool = False, **kwargs) -> None:
        self.run_async(
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
        self.run_async(
            self._execute_requests(
                requests=requests,
                disable_tqdm=disable_tqdm,
                method=self.CHAT_COMPLETION,
                **kwargs,
            )
        )
