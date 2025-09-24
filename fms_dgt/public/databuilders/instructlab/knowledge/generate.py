# Standard
from typing import Dict, List, Literal, Optional
import random

# Local
from fms_dgt.base.databuilder import GenerationDataBuilder
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import GenerationTask
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.blocks.validators.lm_judge import LMJudgeValidator
from fms_dgt.public.blocks.magpie.tag import MagpieTagger
from fms_dgt.public.databuilders.instructlab.knowledge.data_objects import (
    KnowledgeData,
)
from fms_dgt.public.databuilders.instructlab.knowledge.task import (
    KnowledgeTask,
)
from fms_dgt.utils import dgt_logger
import fms_dgt.public.databuilders.instructlab.knowledge.utils as utils


@register_data_builder("instructlab/knowledge")
class KnowledgeDataBuilder(GenerationDataBuilder):
    """Class for generating instruction-response pairs using data from the `knowledge` and `foundational_skills` branch of InstructLab's taxonomy"""

    TASK_TYPE: GenerationTask = KnowledgeTask

    # generator is the main generator that will produce the synthetic examples
    generator: LMProvider

    # validator is the validator which checks a generated api call is correct w.r.t. the input specification
    validator: LMJudgeValidator

    # tagger is the Magpie based tagger to rate synthetic examples
    tagger: Optional[MagpieTagger]

    def call_with_task_list(
        self,
        tasks: List[KnowledgeTask],
        request_idx: int,
    ):
        _ = request_idx
        outputs: List[KnowledgeData] = []
        for task in tasks:
            dgt_logger.info("=" * 99)
            dgt_logger.info('\t\tTask: "%s"', task.name)
            dgt_logger.info("=" * 99)

            outputs.extend(
                self(
                    domain=task.domain,
                    documents=task.get_documents(docs_per_batch=task.num_docs_per_iteration),
                    seed_data=task.get_batch_examples(),
                    prompt_templates=task.prompt_templates,
                    num_icl_examples_per_prompt=task.num_icl_examples_per_prompt,
                    question_style=task.question_style,
                    criteria=task.criteria,
                )
            )
            task.save_knowledge_dataloader_state()

            dgt_logger.info("=" * 99)

        return outputs

    def __call__(
        self,
        domain: str,
        documents: Dict,
        seed_data: List[KnowledgeData],
        prompt_templates: Dict[str, JinjaPromptTemplate],
        num_icl_examples_per_prompt: int = 3,
        question_style: str = "FRQ",
        criteria: List[str] | None = None,
    ) -> List[KnowledgeData]:
        # Generation
        dgt_logger.info("Generating question-answer pairs for the domain %s ...", domain)
        generated_question_answer_pairs = self._generate_question_answer_pairs(
            seed_data=seed_data,
            domain=domain,
            documents=documents,
            prompt_templates=prompt_templates,
            num_icl_examples_per_prompt=num_icl_examples_per_prompt,
            question_style=question_style,
        )

        # Verification
        dgt_logger.info("Validating generated question-answer pairs...")
        validated_question_answer_pairs = self._validate_question_answer_pairs(
            question_answer_pairs=generated_question_answer_pairs,
            prompt_templates=prompt_templates,
            criteria=criteria,
        )

        # Step 3: Magpie Tags Generation
        if hasattr(self, "tagger"):
            # Step 3.a: Run tagger
            tagged_question_answer_pairs = self.tagger(
                validated_question_answer_pairs, disable_tqdm=False
            )

            # Step 3.b: Return
            return tagged_question_answer_pairs
        else:
            return validated_question_answer_pairs

    def _generate_question_answer_pairs(
        self,
        seed_data: List[KnowledgeData],
        domain: str,
        documents: List[Dict],
        prompt_templates: Dict[str, JinjaPromptTemplate],
        num_icl_examples_per_prompt: int,
        question_style: Literal["FRQ", "MCQ"] = "FRQ",
    ) -> List[KnowledgeData]:

        # Build question generator inputs
        question_generator_inputs: List[Dict] = []
        for document in documents:
            # Create new seed data list and shuffle it
            shuffled_seed_data = seed_data + []  # Trick to create new list
            random.shuffle(shuffled_seed_data)

            # Build question generator input based on the requested question style
            if question_style == "MCQ":
                question_generator_inputs.append(
                    {
                        "input": prompt_templates["mcq_question_generation"].encode(
                            render_dict={"document": document["content"]}
                        ),
                        "gen_kwargs": {"stop": prompt_templates["mcq_question_generation"].stop},
                        "document": document,
                        "reference": shuffled_seed_data,
                    }
                )
            else:
                for start_idx in range(0, len(shuffled_seed_data), num_icl_examples_per_prompt):
                    icl_examples = shuffled_seed_data[
                        start_idx : start_idx + num_icl_examples_per_prompt
                    ]

                    question_generator_inputs.append(
                        {
                            "input": prompt_templates["frq_question_generation"].encode(
                                render_dict={
                                    "domain": domain,
                                    "document": document["content"],
                                    "icl_qa_pairs": "\n\n".join(
                                        [
                                            f"[Question]\n{example.question}\n[Answer]\n{example.answer}\n[End]"
                                            for example in icl_examples
                                        ]
                                    ),
                                }
                            ),
                            "gen_kwargs": {
                                "stop": prompt_templates["frq_question_generation"].stop
                            },
                            "document": document,
                            "reference": icl_examples,
                        }
                    )

        # Invoke generator
        generator_outputs = self.generator(question_generator_inputs)

        # Process generator outputs
        outputs = []
        for generator_output in generator_outputs:
            generated_question_answer_pairs = utils.clean_generated_data(generator_output["result"])
            reference: KnowledgeData = generator_output["reference"][0]
            for question_answer_pair in generated_question_answer_pairs:

                outputs.append(
                    KnowledgeData(
                        task_name=reference.task_name,
                        is_seed=False,
                        task_description=reference.task_description,
                        domain=reference.domain,
                        question=question_answer_pair["question"],
                        answer=question_answer_pair["answer"],
                        context=generator_output["document"],
                    )
                )

        # Return
        return outputs

    def _validate_question_answer_pairs(
        self,
        question_answer_pairs: List[KnowledgeData],
        prompt_templates: Dict[str, JinjaPromptTemplate],
        criteria: List[str] | None = None,
    ):
        # Set default criteria, if necessary
        if criteria is None:
            criteria = ["faithfulness", "relevancy", "question_verification"]

        # Initialize necessary variable
        valid_question_answer_pairs: List[KnowledgeData] = question_answer_pairs

        # Validate question answer pairs against each criterion
        for criterion in criteria:
            # Build validator inputs
            validator_inputs: List[Dict] = [
                {
                    "store_names": self.get_block_store_names(
                        block_name=self.validator.name,
                        task_name=question_answer_pair.task_name,
                    ),
                    "reference": question_answer_pair,
                    **self._build_validator_input(
                        prompt_templates=prompt_templates,
                        criterion=criterion,
                        parameters={
                            attr: getattr(question_answer_pair, attr)
                            for attr in ["question", "answer", "context"]
                        },
                    ),
                }
                for question_answer_pair in valid_question_answer_pairs
            ]

            # Invoke validator
            validator_outputs = self.validator(validator_inputs)

            # Retain valid question answer pairs
            valid_question_answer_pairs = [
                validator_output["reference"] for validator_output in validator_outputs
            ]

        # Return valid question answer pairs
        return valid_question_answer_pairs

    def _build_validator_input(
        self,
        prompt_templates: Dict[str, JinjaPromptTemplate],
        criterion: str,
        parameters: Dict,
    ) -> Dict:
        if criterion == "faithfulness":

            def is_faithful(text):
                metadata = {"criterion": criterion}
                if utils.get_faithfulness_score(text) == 1:
                    return True, metadata
                return False, metadata

            return {
                "input": prompt_templates["faithfulness_criterion"].encode(render_dict=parameters),
                "success_func": is_faithful,
                "gen_kwargs": {"stop": prompt_templates["faithfulness_criterion"].stop},
            }

        elif criterion == "relevancy":

            def is_relevant(text):
                metadata = {"criterion": criterion}
                if utils.get_relevancy_score(text) == 1:
                    return True, metadata
                return False, metadata

            return {
                "input": prompt_templates["relevancy_criterion"].encode(render_dict=parameters),
                "success_func": is_relevant,
                "gen_kwargs": {"stop": prompt_templates["relevancy_criterion"].stop},
            }

        elif criterion == "question_verification":

            def is_verified(text):
                metadata = {"criterion": criterion}
                if utils.get_question_verify_rating(text) == 1:
                    return True, metadata
                return False, metadata

            return {
                "input": prompt_templates["question_verification_criterion"].encode(
                    render_dict=parameters
                ),
                "success_func": is_verified,
                "gen_kwargs": {"stop": prompt_templates["question_verification_criterion"].stop},
            }
        else:
            raise ValueError(
                f'Unsupported criterion: {criterion}. Please use one of the following: "faithfulness", "relevancy" or "question_verification".'
            )
