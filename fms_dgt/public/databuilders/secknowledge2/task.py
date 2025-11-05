# Standard
from dataclasses import dataclass
from typing import List, Optional

# Local
from fms_dgt.base.task import DataPoint, GenerationTask
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.retrievers.registry import get_unstructured_text_retriever
from fms_dgt.public.databuilders.secknowledge2.helper.categories import (
    Category,
    SubCategory,
)


@dataclass(kw_only=True)
class InputRow(DataPoint):
    """Class representing a row of data in an input dataset for the SecKnowledge2 pipeline."""

    instruction: str
    answer: str
    grounding_doc: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None


@dataclass(kw_only=True)
class IntermediateRow(DataPoint):
    """Class representing a row of data after the classification and retrieval."""

    instruction: str
    original_answer: str
    category: Category
    subcategory: SubCategory
    search_results: Optional[List[str]] = None
    retrieval_results: Optional[List[str]] = None
    grounding_doc: Optional[str] = None

    def get_evidence_str(self, remove_results: int = 0) -> str:
        """Get the evidence and grounding doc as one string."""
        output = ""
        if self.search_results:
            search_results = (
                self.search_results[:-remove_results]
                if remove_results > 0
                else self.search_results
            )
            if len(search_results) > 0:
                output += f"# Useful Search Results\n\n"
                for search_result in search_results:
                    output += search_result
                    output += "\n\n---\n"
        if self.retrieval_results:
            retrieval_results = (
                self.retrieval_results[:-remove_results]
                if remove_results > 0
                else self.retrieval_results
            )
            if len(retrieval_results) > 0:
                output += f"# Relevant Document{'s' if len(self.retrieval_results) > 1 else ''}\n\n"
                for retrieval_result in retrieval_results:
                    output += retrieval_result
                    output += "\n\n---\n"
        if self.grounding_doc:
            output += f"# Grounding Document\n\n{self.grounding_doc}"
            if self.search_results:
                output += "\n\n---\n"

        return output.strip()


@dataclass(kw_only=True)
class OutputRow(IntermediateRow):
    """Class representing a row of data in an output dataset for the SecKnowledge2 pipeline."""

    category: str
    subcategory: str
    search_results: str
    rewritten_answer: str
    judge_scores: dict

    @staticmethod
    def from_intermediate(
        intermediate_row: IntermediateRow,
        rewritten_answer: str,
        judge_scores: dict,
        evidence_str: str,
    ) -> "OutputRow":
        """Convert an IntermediateRow to an OutputRow."""

        return OutputRow(
            task_name=intermediate_row.task_name,
            is_seed=intermediate_row.is_seed,
            instruction=intermediate_row.instruction,
            original_answer=intermediate_row.original_answer,
            category=intermediate_row.category.name,
            subcategory=intermediate_row.subcategory.name,
            search_results=evidence_str,
            retrieval_results=intermediate_row.retrieval_results,
            grounding_doc=intermediate_row.grounding_doc,
            rewritten_answer=rewritten_answer,
            judge_scores=judge_scores,
        )


class SecKnowledge2Task(GenerationTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = InputRow
    OUTPUT_DATA_TYPE = OutputRow

    def __init__(
        self,
        *args,
        templates_dir: str,
        retriever: dict,
        summarize_web_results: bool = False,
        max_queries_per_instruction: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.templates_dir = templates_dir
        self.retriever = get_unstructured_text_retriever(
            retriever[TYPE_KEY],
            **{k: v for k, v in retriever.items() if k not in [TYPE_KEY]},
        )
        self.summarize_web_results = summarize_web_results
        self.max_queries_per_instruction = max_queries_per_instruction
