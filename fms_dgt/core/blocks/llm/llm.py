# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import abc
import asyncio
import contextvars
import hashlib
import json
import os
import threading
import time

# Third Party
from sqlitedict import SqliteDict
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# Local
from fms_dgt.base.block import DATASET_TYPE, Block, BlockData
from fms_dgt.base.concurrency import CredentialPool, _DualSemaphore
from fms_dgt.base.telemetry import (
    Span,
    payload_recording_enabled,
)
from fms_dgt.constants import NOT_GIVEN, NotGiven

MODEL_ID_OR_PATH = "model_id_or_path"


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass
class ToolChoiceFunction:
    name: str


@dataclass(kw_only=True)
class ToolChoice:
    type: Literal["function", "tool"] = "function"
    function: ToolChoiceFunction


@dataclass(kw_only=True)
class LMBlockData(BlockData):
    """Captures data needed to run instances of LMProvider"""

    input: str | List[Dict]
    gen_kwargs: Dict | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = "auto"
    continuation: str | None = None
    result: str | List[str] | List[dict] | None = None
    addtl: dict | List[dict] | None = None

    def __post_init__(self):
        if self.gen_kwargs is None:
            self.gen_kwargs = dict()


@dataclass
class Parameters:
    max_length: int | NotGiven = NOT_GIVEN

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls,
        params: Dict,
    ):
        raise NotImplementedError

    def to_params(self, kwargs: Dict | None = None) -> Dict:
        field_names = [f.name for f in fields(self)]

        params = {
            **{k: v for k, v in self.__dict__.items() if v is not NOT_GIVEN},
            **{k: v for k, v in kwargs.items() if k in field_names and v is not NOT_GIVEN},
        }
        return params


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
class LMProvider(Block):
    """Base Class for LLM Generators"""

    DATA_TYPE: LMBlockData = LMBlockData

    COMPLETION = "completion"
    CHAT_COMPLETION = "chat_completion"

    # Emitted at most once per process to avoid log spam.
    _payload_warning_emitted: bool = False

    def __new__(cls, *args: Any, **kwargs: Any):
        if "lm_cache" in kwargs and cls is not CachingLM:
            kwargs = dict(kwargs)
            force_cache = kwargs.pop("force_cache", False)
            lm_cache = kwargs.pop("lm_cache")
            # If __new__() does not return an instance of cls, then the new instance’s __init__() method will not be invoked.
            # Thus, we must call CachingLM.__init__
            return CachingLM(
                lm_cls=cls, force_cache=force_cache, lm_cache=lm_cache, *args, **kwargs
            )
        return super().__new__(cls)

    def __init__(
        self,
        model_id_or_path: str = NOT_GIVEN,
        batch_size: int = NOT_GIVEN,
        init_tokenizer: bool = False,
        **kwargs: Any,
    ):
        if "name" not in kwargs or not kwargs["name"]:
            kwargs["name"] = f"lm ({kwargs['type']})"
        super().__init__(**kwargs)

        self._cache_hook = CacheHook()

        if model_id_or_path is NOT_GIVEN:
            raise ValueError(f"Must specify model for Generator {self.name}")

        self._model_id_or_path: str = model_id_or_path
        self._batch_size = batch_size

        # extract parameters from kwargs
        self._parameters, self._chat_parameters = self.init_parameters(**kwargs)

        # only initialize tokenizer if user explicitly asks
        self._tokenizer = self.init_tokenizer() if init_tokenizer else None

        # Initialize usage tracker in profiler data
        self.profiler_data["usage"] = {"tokens": {"completion": 0, "prompt": 0}}

        # Persistent event loop: created once at init, reused across all calls.
        # Running on a dedicated daemon thread so it never blocks the caller.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name=f"dgt-lm-loop-{self._name}"
        )
        self._loop_thread.start()

        # Credential semaphore: None until a subclass calls _init_semaphore().
        # When set, every LLM call acquires it to enforce max_concurrent_requests.
        self._semaphore: Optional[_DualSemaphore] = None

        # Warn once per process when payload recording is on, since it fills
        # traces.jsonl quickly and should only be used for short debugging runs.
        if payload_recording_enabled() and not LMProvider._payload_warning_emitted:
            LMProvider._payload_warning_emitted = True
            self.logger.warning(
                "DGT_TELEMETRY_RECORD_PAYLOADS=1 is active: LLM prompts and completions "
                "will be written to traces.jsonl. Use this only for small debugging runs "
                "as it fills the telemetry directory quickly."
            )

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def model_id_or_path(self):
        return self._model_id_or_path

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def random_seed(self) -> int | None:
        return self._parameters.seed

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def init_tokenizer(self, model_id_or_path: str = None):
        """Initializes a tokenizer

        Args:
            model_id_or_path (str, optional): Model to be used for initializing tokenizer. Defaults to None.
        """
        try:
            return AutoTokenizer.from_pretrained(model_id_or_path or self.model_id_or_path)
        except (OSError, ValueError) as err:
            self.logger.warning(
                'Failed to initialize tokenizer for "%s" due to %s',
                model_id_or_path or self.model_id_or_path,
                err.args[0],
            )
            self.logger.warning(
                'Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
            )
            return None

    def _truncate(self, string: str, max_tokens: int = None) -> str:
        """Truncates string on the left to fit into specified length

        Args:
            string (str): string to truncate
            max_tokens (int): number of tokens that will be used for generation
        """
        # Step 1: Return entire string, if max_length is unspecified
        if self._parameters.max_length is NOT_GIVEN:
            return string

        # Step 2: If tokenizer is available,
        if self.tokenizer:
            # Step 2.a: Encode with specified tokenizer
            encoding = self.tokenizer.encode(string)

            # Step 2.b: Calculate truncation length
            truncate_len = self.max_length - (max_tokens or 0)

            # Step 2.c: Truncate
            encoding = encoding[-truncate_len:]

            # Step 2.d: Return decoded string
            return self.tokenizer.decode(encoding)
        else:
            # Step 2: Return entire string
            return string

    def _prepare_input(
        self,
        instance: LMBlockData,
        method: str,
        max_tokens: int = None,
    ):
        # Step 1: Initialize necessary variables
        prepared_input: str | Dict[str, Any] = None

        # Step 2: Process based in method
        # Step 2.a: "chat" method
        if method == self.CHAT_COMPLETION:
            prepared_input = instance.input

        # Step 2.b: "generate" method
        elif method == self.COMPLETION:
            # Step 2.b.i: string input
            if isinstance(instance.input, str):
                prepared_input = instance.input + (
                    instance.continuation if instance.continuation else ""
                )

            # Step 2.b.ii: list of dictionary input
            elif isinstance(instance.input, list):
                # Step 2.b.ii.*: Verify tokernizer is initialized
                if self.tokenizer is None:
                    raise ValueError(
                        "`init_tokenizer` must be set to `True` to auto apply chat template for non-string inputs."
                    )

                # Step 2.b.ii.**: Apply chat template
                prepared_input = self.tokenizer.apply_chat_template(instance.input, tokenize=False)
            else:
                raise ValueError(
                    f'Unsupported type ({type(instance.input)}) for "LMBlockData.input". Only string or list[dict] are allowed as "LMBlockData.input".',
                )

            # Step 2.b.iii: Truncate prepared input, if necessary
            prepared_input = self._truncate(string=prepared_input, max_tokens=max_tokens)
        else:
            raise ValueError(
                f"Unsupported method ({method}). Please use one of the folllowing. {self.COMPLETION}, {self.CHAT_COMPLETION}.",
            )

        # Step 3: Return
        return prepared_input

    def set_cache_hook(self, cache_hook) -> None:
        self._cache_hook = cache_hook

    def _init_semaphore(self, credential: str, max_concurrent_requests: int = 10) -> None:
        """Register this provider with the process-wide CredentialPool.

        Call this from a subclass ``__init__`` after the credential is known.
        All providers sharing the same credential (API key) will share one
        semaphore, capping total concurrent LLM requests across all of them.

        Args:
            credential: The API key or other credential string.  Only its hash
                is stored in the pool.
            max_concurrent_requests: Concurrency limit for this credential.
                Only applied on first registration; subsequent calls with the
                same credential return the existing semaphore.
        """
        self._semaphore = CredentialPool.get_instance().get(
            credential, max_concurrent_requests=max_concurrent_requests
        )

    @asynccontextmanager
    async def _llm_span(
        self,
        method: str,
        batch_size: int,
        params: Dict,
    ):
        """Async context manager that wraps one LLM API call with a ``dgt.llm_call`` span.

        Acquires the credential semaphore, then opens a ``Span`` whose
        ``duration_ms`` reflects only the API call latency (not the wait).
        The semaphore wait time is recorded as the ``semaphore_wait_ms``
        attribute on the span.

        Yields a mutable ``attrs`` dict.  The caller must populate:
        - ``prompt_tokens`` (int)
        - ``completion_tokens`` (int)

        Optionally (when payload recording is on):
        - ``prompt`` or ``messages`` (str / list)
        - ``completion`` (str / dict)
        - ``payload_truncated`` (bool)

        Args:
            method: ``"completion"`` or ``"chat_completion"``.
            batch_size: Number of individual prompts in this work item.
            params: The resolved generation parameters dict (for extracting
                ``temperature``, ``n``, ``max_tokens`` / ``max_completion_tokens``).
        """
        _temperature = params.get("temperature", None)
        _n = params.get("n", None)
        _max_tokens = params.get(
            "max_tokens",
            params.get(
                "max_completion_tokens",
                params.get("max_new_tokens", params.get("num_predict", None)),
            ),
        )
        attrs: Dict[str, Any] = {
            "provider": type(self).__name__,
            "model_id": self.model_id_or_path,
            "method": method,
            "batch_size": batch_size,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "semaphore_wait_ms": 0.0,
        }
        if _temperature is not None:
            attrs["temperature"] = _temperature
        if _n is not None:
            attrs["n"] = _n
        if _max_tokens is not None:
            attrs["max_tokens"] = _max_tokens

        # Measure how long we wait for the semaphore separately from API time.
        _wait_start = time.monotonic()
        async with self._semaphore:
            attrs["semaphore_wait_ms"] = round((time.monotonic() - _wait_start) * 1000, 2)
            # The Span context manager captures start/end time and writes on exit.
            # duration_ms therefore equals pure API latency.
            with Span(
                "dgt.llm_call",
                self._span_writer,
                parent_span_name="dgt.block",
                **attrs,
            ) as span:
                yield attrs
                # Flush any updates the caller made to attrs into the span record.
                for k, v in attrs.items():
                    span.set_attribute(k, v)

    def run_async(self, coro):
        """Run a coroutine on the persistent event loop and block until it completes.

        Copies the current contextvars context into the loop-thread task so
        that run_id/build_id from RunContextFilter are visible on log records
        emitted inside the coroutine.
        """
        ctx = contextvars.copy_context()

        async def _with_ctx():
            # create_task with an explicit context propagates the caller's
            # contextvars (run_id, build_id) into the coroutine's execution.
            return await self._loop.create_task(coro, context=ctx)

        future = asyncio.run_coroutine_threadsafe(_with_ctx(), self._loop)
        return future.result()

    def serve(self, *args: Any, **kwargs: Any):
        pass

    def release_model(self):
        pass

    def close(self):
        self.release_model()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        self._loop.close()

    def update_instance_with_result(
        self,
        method: str,
        output: Any,
        instance: LMBlockData,
        stop: str | List[str] | None = None,
        additional: dict[str, Any] | List[dict[str, Any]] | None = None,
    ):
        # reduce output, if output is string type and list of stop words are provided in stop sequences
        if stop is not None and isinstance(output, str):
            for term in stop:
                if len(term) > 0:
                    output = output.split(term)[0]

        # set result and addtl fields
        instance.result = output
        instance.addtl = additional

        # update cache hook
        self._cache_hook.add_partial(method, instance, self, output)

    def execute(
        self,
        inputs: Iterable[LMBlockData],
        *args,
        method: str = COMPLETION,
        input_map: List | Dict | None = None,
        output_map: List | Dict | None = None,
        **kwargs: Any,
    ):
        if method == self.COMPLETION:
            self.completion(inputs, **kwargs)
        elif method == self.CHAT_COMPLETION:
            self.chat_completion(inputs, **kwargs)
        else:
            err_str = (
                f"Unhandled method type: {method}"
                if method is not None
                else f"Must set 'method' kwarg to '{self.COMPLETION}' or '{self.CHAT_COMPLETION}'"
            )
            raise ValueError(err_str)

        # Record completion and prompt token usage
        completion_tokens = []
        prompt_tokens = []
        num_failed_inputs = 0
        for entry in inputs:
            if entry.addtl:
                completion_tokens.append(entry.addtl.get("completion_tokens", 0))
                prompt_tokens.append(entry.addtl.get("prompt_tokens", 0))
            else:
                num_failed_inputs += 1
                self.logger.debug("Failed to generate valid output for input: %s ", entry.input)

        if num_failed_inputs:
            self.logger.warning(
                "Prompt token usage count maybe incorrect due to %d failed instances",
                num_failed_inputs,
            )

        self.profiler_data["usage"]["tokens"]["completion"] += sum(completion_tokens)
        self.profiler_data["usage"]["tokens"]["prompt"] += sum(prompt_tokens)

        return inputs

    # ===========================================================================
    #                       ABSTRACT FUNCTIONS
    # ===========================================================================
    @abc.abstractmethod
    def init_parameters(self, **kwargs) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def completion(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> None:
        pass

    @abc.abstractmethod
    def chat_completion(
        self,
        requests: List[LMBlockData],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )


# ===========================================================================
#                       UTILITY FUNCTIONS
# ===========================================================================
def _hash_args(attr: str, request: LMBlockData, base_lm: LMProvider):
    dat = json.dumps([attr] + [request.input, request.gen_kwargs, base_lm.model_id_or_path])
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


# ===========================================================================
#                       UTILITY CLASSES
# ===========================================================================
class CacheHook:

    def __init__(self, dbdict: SqliteDict = None) -> None:
        self._dbdict: SqliteDict = dbdict

    def add_partial(self, attr: str, req: LMBlockData, lm: LMProvider, res: Any) -> None:
        if self._dbdict is None:
            return
        hsh = _hash_args(attr, req, lm)
        self._dbdict[hsh] = res


class CachingLM:
    def __init__(
        self,
        lm_cls: LMProvider,
        force_cache: bool,
        lm_cache: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm_cls: LM
            Underlying LM
        :param force_cache: bool
            Use the cache always, even when sampling
        :param lm_cache: str
            Path to cache db
        """

        self._lm: LMProvider = lm_cls(*args, **kwargs)

        self._cache_db = lm_cache
        if os.path.dirname(self._cache_db):
            os.makedirs(os.path.dirname(self._cache_db), exist_ok=True)
        self._dbdict = SqliteDict(self._cache_db, autocommit=True)

        self._force_cache = force_cache

        # add hook to lm
        self._lm.set_cache_hook(CacheHook(self._dbdict))

    @property
    def dbdict(self):
        return self._dbdict

    def __getattr__(self, attr):
        lm_attr = getattr(self._lm, attr)

        if attr not in ["generate", "chat"]:
            return lm_attr

        method = {
            "completion": LMProvider.COMPLETION,
            "chat_completion": LMProvider.CHAT_COMPLETION,
        }[attr]

        def fn(requests: List[LMBlockData]):
            res = []
            remaining_reqs: List[LMBlockData] = []
            # figure out which ones are cached and which ones are new
            self._lm.logger.info(
                "Loading '%s' responses from cache '%s' where possible...",
                method,
                self._cache_db,
            )

            for req in tqdm(requests, desc="Checking cached requests"):
                hsh = _hash_args(method, req, self._lm)
                if hsh in self._dbdict:
                    ob = self._dbdict[hsh]
                    assert ob is not None
                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)

            self._lm.logger.info(
                "Cached requests: %s, Requests remaining: %s",
                len(requests) - len(remaining_reqs),
                len(remaining_reqs),
            )

            # actually run the LM on the requests that do not have cached results
            getattr(self._lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req in remaining_reqs:
                while res[resptr] is not None:
                    resptr += 1
                res[resptr] = req.result

            # backup commit
            self._dbdict.commit()

            # now we store result
            for req, req_res in zip(requests, res):
                req.result = req_res

        return fn

    def __call__(
        self,
        inputs: DATASET_TYPE,
        *args,
        input_map: List | Dict | None = None,
        output_map: List | Dict | None = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """Copy of Block __call__ method"""

        input_map = input_map or self._lm._input_map
        output_map = output_map or self._lm._output_map

        transformed_inputs = map(lambda x: self._lm.transform_input(x, input_map), inputs)
        if isinstance(inputs, (list, tuple)):
            transformed_inputs = type(inputs)(transformed_inputs)

        outputs = self.execute(transformed_inputs, *args, **kwargs)

        transformed_outputs = map(lambda x: self._lm.transform_output(x, output_map), outputs)
        if isinstance(inputs, (list, tuple)):
            transformed_outputs = type(inputs)(transformed_outputs)

        return transformed_outputs

    def execute(
        self,
        inputs: DATASET_TYPE,
        method: str = LMProvider.COMPLETION,
        **kwargs: Any,
    ) -> None:

        if method == self._lm.COMPLETION:
            self._lm.completion(
                inputs,
                **kwargs,
            )
        elif method == self._lm.CHAT_COMPLETION:
            self._lm.chat_completion(inputs, **kwargs)
        else:
            err_str = (
                f"Unhandled method type: {method}"
                if method is not None
                else f"Must set 'method' kwarg to '{self._lm.COMPLETION}' or '{self._lm.CHAT_COMPLETION}'"
            )
            raise ValueError(err_str)

        return inputs
