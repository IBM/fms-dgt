# Standard
from typing import Dict

# Third Party
from pydantic import BaseModel


class Prompt(BaseModel):
    """Class representing the schema of a prompt."""

    system: str
    user: str


class RewriterPrompt(BaseModel):
    """Class representing the schema of a rewriter prompt."""

    adaptive_system: str
    non_adaptive_system: str
    user: str


class Classifier(BaseModel):
    """Class representing the schema of the classifier prompts directory."""

    category: Prompt
    subcategory: Prompt


class Rewriter(BaseModel):
    """Class representing the schema of the rewriter prompts directory."""

    no_retrieval: RewriterPrompt
    retrieval: RewriterPrompt


class Search(BaseModel):
    """Class representing the schema of the search prompts directory."""

    webpage_summarizer: Prompt
    query_builder: Prompt
    query_filterer: Prompt


class PromptsSchema(BaseModel):
    """Class representing the schema of the prompts directory."""

    classifier: Classifier
    rewriter: Rewriter
    search: Search
    judge: Dict[str, Prompt]  # can be any judges the user wants
