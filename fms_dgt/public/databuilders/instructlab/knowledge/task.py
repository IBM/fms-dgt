# Standard
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

# Local
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.registry import get_dataloader, get_datastore
from fms_dgt.base.task import GenerationTask
from fms_dgt.constants import TYPE_KEY
from fms_dgt.public.databuilders.instructlab.knowledge.constants import (
    _CONTENT_KEY,
    _DATASTORE_CONFIG,
    _FIELDS,
)
from fms_dgt.public.databuilders.instructlab.knowledge.data_objects import (
    KnowledgeData,
)
from fms_dgt.public.databuilders.instructlab.knowledge.utils import (
    extract_docs,
    prepare_documents_for_generation,
)
from fms_dgt.utils import dgt_logger


class KnowledgeTask(GenerationTask):

    INPUT_DATA_TYPE = KnowledgeData
    OUTPUT_DATA_TYPE = KnowledgeData

    def __init__(
        self,
        *args: Any,
        domain: str = None,
        chunk_size: int = -1,
        loop_over: bool = False,
        documents: List[Dict[str, Any]] = None,
        knowledge: Optional[Dict] = None,
        num_icl_examples_per_prompt: int = 3,
        num_docs_per_iteration: int = 100,
        prompt_templates_dir: str | None = None,
        question_style: str = "FRQ",
        criteria: List[str] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        # Initialize Knowledge task variables
        self.domain = domain if domain else self.name
        self.chunk_size = chunk_size
        self.documents = documents
        self.knowledge = knowledge
        self.loop_over = loop_over
        self.num_icl_examples_per_prompt = num_icl_examples_per_prompt
        self.num_docs_per_iteration = num_docs_per_iteration
        self.question_style = question_style
        self.criteria = criteria

        # Initialize Knowledge (either through `knowledge` or `documents`)
        # and its previous state (if any)
        state_datastore = get_datastore(
            "default",
            **{
                "store_name": os.path.join(self._store_name, "knowledge_dataloader_state"),
                "output_dir": self._output_dir,
                "restart": self._restart_generation,
            },
        )
        if knowledge:
            if _DATASTORE_CONFIG not in knowledge:
                raise ValueError(f"Must specify {_DATASTORE_CONFIG} in knowledge")
            self.knowledge_dataloader = get_dataloader(
                dataloader_name="default",
                datastore=self._get_knowledge_ds(knowledge.get(_DATASTORE_CONFIG)),
                state_datastore=state_datastore,
                fields=knowledge.get(_FIELDS),
                loop_over=loop_over,
            )
        elif documents:
            documents_flattened = []
            for doc_group, docs in documents.items():
                _ = doc_group
                docs = extract_docs(docs)
                documents_flattened.extend(docs)

            all_docs = ({_CONTENT_KEY: doc} for doc in documents_flattened)

            self.knowledge_dataloader = get_dataloader(
                dataloader_name="default",
                iterators=[all_docs],
                state_datastore=state_datastore,
                loop_over=loop_over,
            )
        else:
            raise ValueError("Must provide either `knowledge` or `documents` in task.yaml")

        self.load_knowledge_dataloader_state()  # load knowledge state

        # Load prompts
        if prompt_templates_dir:
            prompt_templates_dir = os.path.expandvars(prompt_templates_dir)

            if not os.path.exists(prompt_templates_dir):
                dgt_logger.warning(
                    'Failed to locate prompt templates directory at "%s". Loading default templates from "%s"',
                    prompt_templates_dir,
                    Path(Path(__file__).parent, "prompt_templates"),
                )
                prompt_templates_dir = Path(Path(__file__).parent, "prompt_templates")
        else:
            prompt_templates_dir = Path(Path(__file__).parent, "prompt_templates")

        self.prompt_templates = {}
        for template_path in Path(prompt_templates_dir).glob("*.txt"):
            self.prompt_templates[template_path.name[:-4]] = JinjaPromptTemplate(
                template_path=template_path, stop=["</s>"]
            )

    def _get_knowledge_ds(self, knowledge_datastore: Dict):
        dgt_logger.info(
            "Initializing knowledge datastore with parameters %s",
            knowledge_datastore,
        )
        return get_datastore(
            knowledge_datastore.get(TYPE_KEY),
            **{
                "output_dir": self._output_dir,
                **knowledge_datastore,
            },
        )

    def get_documents(self, docs_per_batch=None):
        """
        This method returns document chunks from the specified knowledge source and prepares them for generation.

        Args:
            docs_per_batch (Optional[int]): No. of documents to prepare per batch. Defaults to using all documents.

        Returns:
            list: A list of document chunks.
        """
        docs = []
        for i in count():
            if docs_per_batch:
                if i >= docs_per_batch:
                    break
            try:
                docs.append(next(self.knowledge_dataloader))
            except StopIteration:
                break

        document_chunks = prepare_documents_for_generation(
            knowledge_source=docs,
            domain=self.domain,
            chunk_size=self.chunk_size,
        )
        return document_chunks

    def save_knowledge_dataloader_state(self):
        """Saves the state of the knowledge_dataloader"""
        curr_state = self.knowledge_dataloader.get_state()
        self.knowledge_dataloader.state_datastore.save_data([curr_state])

    def load_knowledge_dataloader_state(self):
        """Loads the state of the knowledge_dataloader"""
        prev_state = self.knowledge_dataloader.state_datastore.load_data()
        if prev_state:
            self.knowledge_dataloader.set_state(prev_state[-1])

    def instantiate_input_example(self, **kwargs: Any):
        domain = kwargs.get("domain", self.domain)
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            task_description=self.task_description,
            domain=domain,
            question=kwargs.get("question"),
            answer=kwargs.get("answer"),
        )
