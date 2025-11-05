# Standard
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union
import re

# Third Party
# Third-party
from tqdm import tqdm

# Local
from fms_dgt.base.databuilder import GenerationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import GenerationTask
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider
from fms_dgt.core.retrievers.unstructured_text.base import (
    UnstructuredTextRetriever,
)
from fms_dgt.core.retrievers.unstructured_text.web_search.base import (
    SearchEngineRetriever,
    SearchResult,
)
from fms_dgt.public.databuilders.secknowledge2.helper.categories import TemplateData
from fms_dgt.public.databuilders.secknowledge2.helper.schemas import PromptsSchema
from fms_dgt.public.databuilders.secknowledge2.task import (
    InputRow,
    IntermediateRow,
    OutputRow,
    SecKnowledge2Task,
)
from fms_dgt.utils import dgt_logger, read_json

TreeDict = Dict[str, Union["TreeDict", str]]


def build_dir_tree(parent_dir: Union[Path, str]) -> TreeDict:
    """
    Recursively builds a directory tree structure starting from the given parent directory.
    Args:
        parent_dir (Path | str): The root directory to start building the tree from.
                                 Can be a string or a Path object.
    Returns:
        TreeDict: A nested dictionary representing the directory structure.
                  - Keys are directory or file names.
                  - Values are:
                      - Nested dictionaries for subdirectories.
                      - File contents (as strings) for `.txt` files.
    Raises:
        ValueError: If the provided path is not a directory.
        ValueError: If the provided path does not exist.
    Notes:
        - Only `.txt` files are included in the tree, and their contents are read as strings.
        - Other file types are ignored.
    """

    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        raise ValueError(f"Path {parent_dir} is not a directory.")
    if not parent_dir.exists():
        raise ValueError(f"Path {parent_dir} does not exist.")

    tree = {}

    for item in parent_dir.iterdir():
        if item.is_dir():
            tree[item.name] = build_dir_tree(item)
        elif item.is_file() and item.suffix == ".txt":
            tree[item.stem] = item.read_text()

    return tree


def default_parse_output(x):
    if not isinstance(x, str):
        raise ValueError("Invalid output type")
    return x


def generate_until_parsing_output(
    generator: LMProvider,
    inputs: List[Dict],
    parse_output: Callable[[Union[str, List[str], List[Dict], None]], Any] = default_parse_output,
    max_parse_tries: int = 5,
    max_llm_tries: int = 2,
) -> List[Any]:
    """
    Generates outputs using a language model generator, ensuring that the outputs
    are restricted to a predefined list of allowed values.
    Args:
        generator (LMProvider): The language model generator instance used to produce outputs.
        inputs (List[Dict]): A list of input dictionaries to be processed by the generator.
        parse_output (Callable[[Union[str, List[str], List[Dict], None]], Any]): A function that attempts to parse the output in some way and throw a ValueError if unsuccessful.
        max_tries (int): The maximum number of retries for generating outputs.
    Returns:
        List[Any]: A list of generated outputs, each guaranteed to be within the allowed outputs. If no valid output is generated after max_tries, the corresponding entry will be None.
    """

    if max_parse_tries < 1:
        raise ValueError("max_parse_tries must be greater than 0.")
    if max_llm_tries < 1:
        raise ValueError("max_llm_tries must be greater than 0.")

    outputs: List[Any] = [None] * len(inputs)

    idx_to_parse_try = [0] * len(inputs)
    idx_to_llm_try = [0] * len(inputs)
    while None in outputs:
        remaining_indices = [
            i
            for i, output in enumerate(outputs)
            if output is None
            and idx_to_parse_try[i] < max_parse_tries
            and idx_to_llm_try[i] < max_llm_tries
        ]
        if len(remaining_indices) == 0:
            break
        llm_inputs = [inputs[i] for i in remaining_indices]
        try:
            llm_outputs: List[LMBlockData] = generator(
                llm_inputs, method=LMProvider.CHAT_COMPLETION
            )
        except Exception as e:
            dgt_logger.warning(
                f"!!FATAL!! Failed to generate batch outputs with error: {e}. Retrying..."
            )
        else:
            for inp_idx, llm_output in zip(remaining_indices, llm_outputs):
                # get output text
                output_text = None
                if llm_output["result"]:
                    output_text = llm_output["result"].get("content", None)

                # handle output text
                if not output_text:
                    idx_to_llm_try[inp_idx] += 1
                else:
                    try:
                        outputs[inp_idx] = parse_output(output_text)
                    except ValueError as e:
                        dgt_logger.warning(f'Failed to parse output: "{e}". Retrying...')
                        idx_to_parse_try[inp_idx] += 1

    num_errors = outputs.count(None)
    if num_errors > 0:
        dgt_logger.warning(
            f"{num_errors} outputs could not be generated after {max_llm_tries} tries for generating the responses, and {max_parse_tries} tries for parsing the output - consider increasing `max_tries` or checking the input data for issues."
        )

    return outputs


class SearchMethod(Enum):
    """Enum for the search methods."""

    INSTRUCTION = "instruction"
    LLM = "llm"
    HYBRID_LENGTH = "hybrid_length"
    HYBRID_JUDGE = "hybrid_judge"


@register_data_builder("secknowledge2")
class SecKnowledge2DataBuilder(GenerationDataBuilder):
    """Class for the implementation of the SecKnowledge2 pipeline as a data builder."""

    TASK_TYPE: GenerationTask = SecKnowledge2Task  # type: ignore

    # classifier is the LLM that will classify a given instruction to its corresponding task and subtask (if not given)
    classifier: LMProvider

    # rewriter is the LLM that will rewrite the response according to the selected format
    rewriter: LMProvider

    # judge is the LLM that will judge the rewritten response
    judge: LMProvider

    # query builder is the LLM that will generate search queries from a given instruction and structure
    query_builder: LMProvider

    def __init__(self, *args, **kwargs):
        specifications = kwargs["config"].pop("specifications")

        super().__init__(*args, **kwargs)

        prompts_dict = build_dir_tree(Path(__file__).resolve().parent / "prompts")
        self._prompts = PromptsSchema(**prompts_dict)  # type: ignore

        templates_path = Path(
            specifications.get("templates_path", "data/public/secknowledge2/templates/")
        )
        self._template_data = TemplateData.from_auto(templates_path)
        self._adaptive = specifications.get("adaptive", False)
        self._search_method = SearchMethod(specifications.get("search_method", "llm"))
        if file_path := specifications.get("search_queries_cache", None):
            self._search_queries_cache = read_json(file_path)
        else:
            self._search_queries_cache = None

    def _category_classification(self, instruction_data: List[InputRow]) -> List[InputRow]:
        replace_indices: List[int] = []
        llm_inputs: List[Dict] = []

        prompt = self._prompts.classifier.category

        # Find rows with missing or invalid categories
        for i, row in enumerate(instruction_data):
            if (
                row.subcategory in self._template_data.subcategories_names()
                or row.category in self._template_data.categories_names()
            ):
                # no classification needed
                continue
            else:
                # needs to find category
                replace_indices.append(i)
                llm_inputs.append(
                    {
                        "input": [
                            {
                                "role": "system",
                                "content": prompt.system.format(
                                    categories=self._template_data.categories_str()
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt.user.format(instruction=row.instruction),
                            },
                        ]
                    }
                )

        # classify categories
        category_names = self._template_data.categories_names()

        def parse_output(x):
            if not isinstance(x, str):
                raise ValueError("Invalid output type")
            if x not in category_names:
                raise ValueError(f"Output {x} is not a valid category.")
            return x

        dgt_logger.info("Classifyig categories...")
        categories = generate_until_parsing_output(
            generator=self.classifier,
            inputs=llm_inputs,
            parse_output=parse_output,
        )

        # replace the categories in the instruction data
        updated_rows = deepcopy(instruction_data)
        for idx, category in zip(replace_indices, categories):
            updated_rows[idx].category = category

        return updated_rows

    def classify(self, instruction_data: List[InputRow]) -> List[IntermediateRow]:
        # Ensure that all categories are filled
        rows_with_categories = self._category_classification(instruction_data)

        replace_indices: List[int] = []
        llm_inputs: List[Dict] = []

        prompt = self._prompts.classifier.subcategory

        # Find rows with missing or invalid subcategories
        for i, row in enumerate(rows_with_categories):
            if row.subcategory not in self._template_data.subcategories_names():
                # given the category, need to find subcategory
                replace_indices.append(i)

                category = self._template_data.get_category_by_name(row.category)  # type: ignore
                llm_inputs.append(
                    {
                        "input": [
                            {
                                "role": "system",
                                "content": prompt.system.format(
                                    sub_categories=category.subcategories_str()
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt.user.format(
                                    instruction=row.instruction, category=category.name
                                ),
                            },
                        ]
                    }
                )

        # classify subcategories
        subcategories_names = self._template_data.subcategories_names()

        def parse_output(x):
            if not isinstance(x, str):
                raise ValueError("Invalid output type")
            if x in subcategories_names:
                return x
            else:
                raise ValueError(f"Output {x} is not a valid category.")

        dgt_logger.info("Classifyig sub categories...")
        subcategories = generate_until_parsing_output(
            generator=self.classifier,
            inputs=llm_inputs,
            parse_output=parse_output,
        )

        # replace the subcategories in the instruction data
        rows_with_subcategories = deepcopy(rows_with_categories)
        for idx, subcategory in zip(replace_indices, subcategories):
            rows_with_subcategories[idx].subcategory = subcategory

        # Convert to IntermediateRow
        processed_rows = []
        for row in rows_with_subcategories:
            processed_rows.append(
                IntermediateRow(
                    task_name=row.task_name,
                    is_seed=row.is_seed,
                    instruction=row.instruction,
                    original_answer=row.answer,
                    category=self._template_data.get_category_by_name(row.category),  # type: ignore
                    subcategory=self._template_data.get_subcategory_by_name(
                        name=row.subcategory  # type: ignore
                    ),
                    search_results=None,
                    grounding_doc=row.grounding_doc,
                )
            )

        return processed_rows

    def _build_search_queries(
        self, rows: List[IntermediateRow], max_queries_per_instruction: int
    ) -> List[List[str]]:
        if self._search_method == SearchMethod.LLM:
            llm_build_query_condition = lambda row: True
        elif self._search_method == SearchMethod.HYBRID_LENGTH:
            llm_build_query_condition = lambda row: len(row.instruction) > 50
        elif self._search_method == SearchMethod.HYBRID_JUDGE:
            raise NotImplementedError()
        else:
            llm_build_query_condition = lambda row: False

        query_builder_prompt = self._prompts.search.query_builder
        query_filterer_prompt = self._prompts.search.query_filterer

        def extract_queries(x: Any) -> List[str]:
            if not isinstance(x, str):
                raise ValueError("Invalid output type")
            return re.findall(r"<query>(.*?)</query>", x, re.DOTALL)

        rows_to_build_queries = [
            row
            for row in rows
            if llm_build_query_condition(row)
            and (
                not self._search_queries_cache
                or not self._search_queries_cache.get(row.instruction)
            )
        ]

        # build queries based on instruction
        llm_inputs = [
            {
                "input": [
                    {
                        "role": "system",
                        "content": query_builder_prompt.system.format(
                            K=max_queries_per_instruction
                        ),
                    },
                    {
                        "role": "user",
                        "content": query_builder_prompt.user.format(
                            user_question=row.instruction,
                            K=max_queries_per_instruction,
                        ),
                    },
                ]
            }
            for row in rows_to_build_queries
        ]
        dgt_logger.info("Generating search queries...")
        initial_search_queriess = generate_until_parsing_output(
            generator=self.query_builder,
            inputs=llm_inputs,
            parse_output=extract_queries,
        )

        # filter out queries based on instruction, existing answer, and desired structure
        llm_inputs = [
            {
                "input": [
                    {
                        "role": "system",
                        "content": query_filterer_prompt.system,
                    },
                    {
                        "role": "user",
                        "content": query_filterer_prompt.user.format(
                            user_question=row.instruction,
                            draft_answer=row.original_answer,
                            structure=row.subcategory.structure,
                            search_queries="\n".join(initial_search_queries),
                        ),
                    },
                ]
            }
            for row, initial_search_queries in zip(rows_to_build_queries, initial_search_queriess)
        ]
        dgt_logger.info("Filtering search queries...")
        filtered_search_queriess = generate_until_parsing_output(
            generator=self.query_builder,
            inputs=llm_inputs,
            parse_output=extract_queries,
        )

        # collect results
        result = []
        for row in rows:
            if self._search_queries_cache and row.instruction in self._search_queries_cache:
                queries = self._search_queries_cache[row.instruction][:max_queries_per_instruction]
            elif llm_build_query_condition(row):
                queries = filtered_search_queriess.pop(0) or []
            else:
                queries = [row.instruction]
            result.append([query.strip() for query in queries if query])

        return result

    def _summarize_search_results(
        self,
        rows_search_results: List[Tuple[IntermediateRow, List[SearchResult]]],
        search: SearchEngineRetriever,
    ) -> List[List[SearchResult]]:
        prompt = self._prompts.search.webpage_summarizer

        if search.process_webpages:
            # summarize contents with LLM
            llm_inputs = [
                {
                    "input": [
                        {
                            "role": "system",
                            "content": prompt.system,
                        },
                        {
                            "role": "user",
                            "content": prompt.user.format(
                                document=search_result.text,
                                question=row.instruction,
                                structure=row.subcategory.structure,
                            ),
                        },
                    ]
                }
                for (row, search_results) in rows_search_results
                for search_result in search_results  # type: ignore
            ]
            dgt_logger.info("Summarizing search results...")
            summarized_contents = generate_until_parsing_output(
                generator=self.rewriter, inputs=llm_inputs
            )
            i = 0
            for row, search_results in rows_search_results:
                for search_result in search_results:
                    if summarized_contents[i] is not None:
                        search_result.text = summarized_contents[i]
                    i += 1

        return [search_result for (row, search_result) in rows_search_results]

    def _search_evidence_retrieval(
        self,
        classified_rows: List[IntermediateRow],
        search: SearchEngineRetriever,
        max_queries_per_instruction: int,
        summarize: bool,
    ) -> List[IntermediateRow]:

        rows_requires_search_indices = [
            i for i, row in enumerate(classified_rows) if row.subcategory.requires_search
        ]

        # build search queries and run
        rows_queries = self._build_search_queries(
            [classified_rows[i] for i in rows_requires_search_indices],
            max_queries_per_instruction,
        )
        flattened_queries = [
            (classified_rows[rows_requires_search_indices[i]], query)
            for i, row_queries in enumerate(rows_queries)
            for query in row_queries
        ]
        rows_results = [
            search(row_queries, disable_tqdm=True)
            for row_queries in tqdm(rows_queries, desc="Searching and processing evidence")
        ]
        flattened_search_results = [
            result for row_results in rows_results for result in row_results
        ]

        # summarize search results
        if summarize:
            flattened_summarized_search_results = self._summarize_search_results(
                [
                    (row, search_result)
                    for search_result, (row, query) in zip(
                        flattened_search_results, flattened_queries
                    )
                ],
                search,
            )
        else:
            flattened_summarized_search_results = flattened_search_results

        # repack the search results into the original structure
        current_row_search_results = []
        i = 0
        summarized_search_results = []
        for search_result in flattened_summarized_search_results:
            while i < len(rows_queries) and len(rows_queries[i]) == 0:
                summarized_search_results.append([])
                i += 1
            if i == len(rows_queries):
                break
            if len(current_row_search_results) < len(rows_queries[i]):
                # insert current search result into the current row
                current_row_search_results.append(search_result)
            if len(current_row_search_results) == len(rows_queries[i]):
                # current row is full - add bundle to the list, reset bundle, and move to next row
                summarized_search_results.append(current_row_search_results)
                current_row_search_results = []
                i += 1

        assert len(summarized_search_results) == len(
            rows_queries
        ), "Outer mismatch in number of search results and queries."
        assert all(
            [
                len(row_queries) == len(search_results)
                for row_queries, search_results in zip(rows_queries, summarized_search_results)
            ]
        ), "Inner Mismatch in number of search results and queries."

        retrieved_rows = deepcopy(classified_rows)
        for i, queries, search_results in zip(
            rows_requires_search_indices, rows_queries, summarized_search_results
        ):
            retrieved_rows[i].search_results = [
                search.result_to_str(query, search_result)
                for query, search_result in zip(queries, search_results)
            ]

        return retrieved_rows

    def _rag_evidence_retrieval(
        self,
        classified_rows: List[IntermediateRow],
        retriever: UnstructuredTextRetriever,
    ) -> List[IntermediateRow]:

        results = retriever([row.instruction for row in classified_rows])

        for row, result in zip(classified_rows, results):
            row.retrieval_results = [r.text for r in result]

        return classified_rows

    def evidence_retrieval(
        self,
        classified_rows: List[IntermediateRow],
        retriever: UnstructuredTextRetriever,
        max_queries_per_instruction: int,
        summarize_web_results: bool,
    ) -> List[IntermediateRow]:

        if isinstance(retriever, SearchEngineRetriever):
            retrieved_rows = self._search_evidence_retrieval(
                classified_rows,
                retriever,
                max_queries_per_instruction,
                summarize_web_results,
            )
        elif isinstance(retriever, UnstructuredTextRetriever):
            if max_queries_per_instruction is not None:
                dgt_logger.warning(
                    "max_queries_per_instruction is not applicable for UnstructuredTextRetriever. Ignoring."
                )
            retrieved_rows = self._rag_evidence_retrieval(classified_rows, retriever)
        else:
            raise ValueError(
                f"Unsupported retriever type: {type(retriever)}. "
                "Expected SearchEngineRetriever or UnstructuredTextRetriever."
            )

        return retrieved_rows

    def rewrite_answers(
        self, retrieved_rows: List[IntermediateRow], _try: int = 0
    ) -> List[OutputRow]:
        rewritten_rows: List[OutputRow] = [None] * len(retrieved_rows)  # type: ignore

        replace_indices: List[int] = []
        llm_inputs: List[Dict] = []

        prompts = self._prompts.rewriter

        for i, row in enumerate(retrieved_rows):
            if not row.subcategory.requires_rewrite:
                rewritten_rows[i] = OutputRow.from_intermediate(
                    intermediate_row=row,
                    rewritten_answer=row.original_answer,
                    judge_scores={},
                    evidence_str="",
                )
            else:
                if (
                    row.subcategory.requires_search or row.subcategory.requires_grounding_doc
                ) and row.get_evidence_str(remove_results=_try):
                    if self._adaptive:
                        system_prompt = prompts.retrieval.adaptive_system
                    else:
                        system_prompt = prompts.retrieval.non_adaptive_system
                else:
                    if self._adaptive:
                        system_prompt = prompts.no_retrieval.adaptive_system
                    else:
                        system_prompt = prompts.no_retrieval.non_adaptive_system

                replace_indices.append(i)
                llm_inputs.append(
                    {
                        "input": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": prompts.retrieval.user.format(
                                    question=row.instruction,
                                    response=row.original_answer,
                                    structure=row.subcategory.structure,
                                    # the following line will not take effect if no evidence is needed
                                    evidence=row.get_evidence_str(remove_results=_try),
                                ),
                            },
                        ]
                    }
                )

        def parse_output(x):
            if not isinstance(x, str):
                raise ValueError("Invalid output type")
            match = re.search(
                r"\[\s*Revised Response start\s*\](.*)\[\s*Revised Response end\s*\]",
                x,
                re.DOTALL,
            )
            if not match:
                raise ValueError(
                    f"Output '{x}' is not a valid rewritten answer. "
                    "Expected format: '[ Revised Response start ]... [ Revised Response end ]'."
                )
            return match.group(1).strip()

        # generate new answers
        if _try == 0:
            dgt_logger.info("Rewriting answers...")
        else:
            dgt_logger.info(f"Retrying rewriting answers with less evidence... (try {_try})")
        llm_outputs = generate_until_parsing_output(
            generator=self.rewriter, inputs=llm_inputs, parse_output=parse_output
        )
        to_retry = []
        for idx, llm_output in zip(replace_indices, llm_outputs):
            if llm_output:
                rewritten_rows[idx] = OutputRow.from_intermediate(
                    intermediate_row=retrieved_rows[idx],
                    rewritten_answer=llm_output,
                    judge_scores={},
                    evidence_str=retrieved_rows[idx].get_evidence_str(remove_results=_try),
                )
            elif _try < 3:
                to_retry.append(idx)
            else:
                rewritten_rows[idx] = OutputRow.from_intermediate(
                    intermediate_row=retrieved_rows[idx],
                    rewritten_answer="",
                    judge_scores={},
                    evidence_str="",
                )

        if len(to_retry) > 0:
            retry_results = self.rewrite_answers(
                [retrieved_rows[i] for i in to_retry], _try=_try + 1
            )
            for idx, rewritten_row in zip(to_retry, retry_results):
                rewritten_rows[idx] = rewritten_row

        return rewritten_rows

    def judge_answers(self, rewritten_rows: List[OutputRow]) -> List[OutputRow]:
        llm_inputs_factuality: List[Dict] = []
        llm_inputs_readability: List[Dict] = []

        for i, row in enumerate(rewritten_rows):
            if row.rewritten_answer:
                # factuality judge
                factuality_prompt = self._prompts.judge["factuality"]
                llm_inputs_factuality.append(
                    {
                        "input": [
                            {"role": "system", "content": factuality_prompt.system},
                            {
                                "role": "user",
                                "content": factuality_prompt.user.format(
                                    question=row.instruction,
                                    ref_answer=row.original_answer,
                                    answer=row.rewritten_answer,
                                ),
                            },
                        ]
                    }
                )

                # readability judge
                readability_prompt = self._prompts.judge["readability"]
                llm_inputs_readability.append(
                    {
                        "input": [
                            {
                                "role": "system",
                                "content": readability_prompt.system,
                            },
                            {
                                "role": "user",
                                "content": readability_prompt.user.format(
                                    question=row.instruction,
                                    answer_a=row.original_answer,
                                    answer_b=row.rewritten_answer,
                                ),
                            },
                        ]
                    }
                )
                llm_inputs_readability.append(
                    {
                        "input": [
                            {
                                "role": "system",
                                "content": readability_prompt.system,
                            },
                            {
                                "role": "user",
                                "content": readability_prompt.user.format(
                                    question=row.instruction,
                                    answer_a=row.rewritten_answer,
                                    answer_b=row.original_answer,
                                ),
                            },
                        ]
                    }
                )

        def factuality_output_parser(x) -> int:
            if not isinstance(x, str):
                raise ValueError("Invalid output type")
            match = re.search(r"\[\[\d+\]\]", x)
            if not match:
                raise ValueError(f"Output '{x}' is not a valid factuality score.")
            return int(match.group(0)[2:-2])

        def readability_output_parser(x) -> str:
            if not isinstance(x, str):
                raise ValueError("Invalid output type")
            match = re.search(r"\[\[[ABC]\]\]", x)
            if not match:
                raise ValueError(f"Output '{x}' is not a valid readability score.")
            return match.group(0)[2:-2]

        # generate judge scores
        dgt_logger.info("Assessing rewritten answers factuality...")
        llm_outputs_factuality = generate_until_parsing_output(
            generator=self.judge,
            inputs=llm_inputs_factuality,
            parse_output=factuality_output_parser,
        )

        dgt_logger.info("Choosing final answers based on readability...")
        llm_outputs_readability = generate_until_parsing_output(
            generator=self.judge,
            inputs=llm_inputs_readability,
            parse_output=readability_output_parser,
        )

        judged_rows = deepcopy(rewritten_rows)
        j = 0
        for i, row in enumerate(judged_rows):
            if row.rewritten_answer:
                judged_rows[i].judge_scores["factuality"] = llm_outputs_factuality[j]
                readability = (
                    llm_outputs_readability[j * 2],
                    llm_outputs_readability[j * 2 + 1],
                )
                if readability[0] and readability[1]:
                    if readability[0] == "A" and readability[1] == "B":
                        result = "original"
                    elif readability[0] == "B" and readability[1] == "A":
                        result = "rewritten"
                    elif readability[0] == "C" and readability[1] == "C":
                        result = "tie"
                    else:
                        result = f"inconsistent ({readability[0]} vs {readability[1]})"
                    judged_rows[i].judge_scores["readability"] = result
                else:
                    judged_rows[i].judge_scores["readability"] = None
                j += 1

        return judged_rows

    def call_with_task_list(self, tasks: List[SecKnowledge2Task], request_idx: int) -> List[OutputRow]:
        output = []
        for task in tasks:
            data_pool = task.get_batch_examples()
            retriever = task.retriever
            max_queries_per_instruction = task.max_queries_per_instruction
            summarize_web_results = task.summarize_web_results
            output.extend(
                self(
                    request_idx,
                    data_pool,
                    retriever,
                    max_queries_per_instruction,
                    summarize_web_results,
                )
            )
        return output

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[InputRow],
        retriever: UnstructuredTextRetriever,
        max_queries_per_instruction: int,
        summarize_web_results: bool,
    ) -> List[OutputRow]:
        inputs = [data for data in instruction_data if isinstance(data, InputRow)]

        # Step 1. Classification to subcategories (where needed)
        classified_rows = self.classify(inputs)

        # Step 2. Retrieving evidence
        retrieved_rows = self.evidence_retrieval(
            classified_rows,
            retriever,
            max_queries_per_instruction,
            summarize_web_results,
        )

        # Step 3. Rewriting the answers
        rewritten_rows = self.rewrite_answers(retrieved_rows)

        # Step 4. Judging the answers
        judged_rows = self.judge_answers(rewritten_rows)

        return judged_rows
