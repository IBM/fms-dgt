# Standard
from typing import Dict, List
import re

# Local
from fms_dgt.public.databuilders.instructlab.knowledge.constants import (
    _CONTENT_KEY,
)


def remove_references(knowledge_docs: List[dict]):
    """
    Remove references from a list of knowledge documents

    Parameters:
    - knowledge_docs (list): List of knowledge documents.

    Returns:
    List(dict): A modified list with the 'References' column removed.
    """
    for doc in knowledge_docs:
        doc[_CONTENT_KEY] = (
            doc[_CONTENT_KEY][doc[_CONTENT_KEY].index("References") :]
            if "References" in doc[_CONTENT_KEY]
            else doc[_CONTENT_KEY]
        )
    return knowledge_docs


def fuse_texts(text_list, short_length_threshold=100):
    """
    Fuses multiple short texts into a single long text.

    Parameters:
    - text_list (list): A list of short texts.
    - short_length_threshold (int): The length of each short text.

    Returns:
    list: A list of fused texts.
    """
    fused_texts = []
    previous_long_text = ""

    for text in text_list:
        word_count = len(text.split())

        if word_count <= short_length_threshold and previous_long_text:
            # Append the short text to the last long text
            fused_texts[-1] += " " + text
        else:
            # This is a long text, so add it to the list and remember it
            fused_texts.append(text)
            previous_long_text = text

    return fused_texts


def chunk_doc_based_on_words(knowledge_docs: List[dict], chunk_size=1000):
    """
    Chunk documents based on words.

    Parameters:
    - knowledge_docs (list): List of knowledge documents.
    - chunk_size (int): Number of tokens per chunk.

    Returns:
    List(dict): A new dataset with documents chunked into specified size chunks.
    """
    final_ds = []
    for doc in knowledge_docs:
        chapter = doc[_CONTENT_KEY].split()
        doc_copy = {**doc}
        if len(chapter) > chunk_size:
            chunks = [
                " ".join(chapter[span : min(span + chunk_size, len(chapter))])
                for span in range(0, len(chapter), chunk_size)
            ]
            chunks = fuse_texts(chunks, short_length_threshold=300)
            for chunk in chunks:
                doc_copy[_CONTENT_KEY] = chunk
                doc_copy["document_len"] = len(chunk.split(" "))
                final_ds.append(doc_copy)
        else:
            doc_copy["document_len"] = len(chapter)
            final_ds.append(doc_copy)
    return final_ds


def extract_docs(docs):
    """
    Extracts and returns all documents from a given input.

    Parameters:
        docs (str | list | dict): The input document(s). If a string, it will be returned as a single-item list.
            If a list, each item in the list will be recursively processed using the `extract_docs` function.
            If a dictionary, the "content" key will be extracted and returned as a single-item list.

    Returns:
        list: A list of all documents extracted from the given input.
    """
    if isinstance(docs, str):
        return [docs]
    elif isinstance(docs, list):
        return [d for doc in docs for d in extract_docs(doc)]
    elif isinstance(docs, dict):
        if not docs.get(_CONTENT_KEY):
            raise ValueError(
                "When docs are provided as dictionaries, each dictionary must have a 'content' key"
            )
        return [docs.get(_CONTENT_KEY)]
    else:
        raise ValueError("Document provided was not a string, list, or dict")


def prepare_documents_for_generation(
    knowledge_source: List[Dict],
    chunk_size: int = -1,
    domain: str = None,
):
    """
    Prepare knowledge documents for generation.

    Parameters:
    - knowledge_source (List[Dict]): List of knowledge objects { chapter(s): text | list[text] } pairs.
    - chunk_size (int): Number of tokens per chunk.
    - domain (str): The domain of the knowledge documents.
    """
    knowledge_docs = []
    for document in knowledge_source:
        knowledge_docs.append(
            (
                {**document, "domain": domain}
                if isinstance(document, dict)
                else {"content": document, "domain": domain}
            ),
        )
    knowledge_docs = remove_references(knowledge_docs)
    if chunk_size >= 1:
        knowledge_docs = chunk_doc_based_on_words(knowledge_docs, chunk_size=chunk_size)
    return knowledge_docs


## TODO Clean this up
def parse_qa_v2(text):
    """
    Parse Q&A pairs from text.

    Parameters:
    - text (str): Text to parse.

    Returns:
    list: A list of Q&A pairs.
    """
    # 1. Adjusted pattern to match both formats
    pattern = r"\[Question\]\s*(.*?)\s*\[Answer\]\s*(.*?)\s*(?=\[Question\]|$)"

    # 2. Find all matches in the text using DOTALL to match across lines
    matches = re.findall(pattern, text, re.DOTALL)
    matches = list(matches)
    if len(matches) > 1:
        # 2.a Drop last sample if it doesn't have `[END]` key
        if ("[End]" not in matches[-1][1]) and ("[END]" not in matches[-1][1]):
            matches = matches[:-1]

    # 3. Convert matches to a list of dictionaries
    qa_list = []
    for match in matches:
        question = match[0].replace("[Question]", "").replace("[/INST]", "").strip()
        answer = (
            match[1]
            .replace("[End]", "")
            .replace("[END]", "")
            .replace("[/INST]", "")
            .strip()
        )
        # 3.a If we encounter a [End]/[END] tag in the question, then it wasn't parsed correctly, so we skip this example
        if ("[End]" in question) or ("[END]" in question):
            continue
        # 3.b Only take those samples where both question and answer are not empty strings
        if (question != "") and (answer != ""):
            qa_list.append({"question": question, "answer": answer})

    # 4. Return
    return qa_list


def clean_generated_data(model_response):
    """
    Cleans generated data.

    Parameters:
    - model_response (str): generated response

    Returns:
    qa_list: A cleaned dataset with the response column removed.
    """
    model_response = "[Question]\n" + model_response
    qa_list = parse_qa_v2(model_response)
    return qa_list


def get_faithfulness_score(model_response: str):
    """
    Assigns faithfulness score for a text
    Parameters:
    - model_response (str): Model response
    Return:
    faithfulness_rating: The faithfulness score
    """
    rating_regex_1 = re.compile(r"\*\*Answer:\*\*\s*(\w+)")
    rating_regex_2 = re.compile(r"Answer:\s*(\w+)")

    def extract_faithfulness_score(text):
        try:
            rating = rating_regex_1.findall(text)[0]
            return rating
        # pylint: disable=broad-exception-caught
        except Exception:
            try:
                rating = rating_regex_2.findall(text)[0]
                return rating
            # pylint: disable=broad-exception-caught
            except Exception:
                return "-1"

    faithfulness_rating = extract_faithfulness_score(model_response)
    faithfulness_rating = 1 if faithfulness_rating.lower() == "yes" else 0

    return faithfulness_rating


def get_relevancy_score(model_response: str):
    """
    Assign relevancy score for a text
    Parameters:
    - model_response (str): Model response
    Return:
    relevancy_rating: The relevancy score
    """

    def extract_total_score(text):
        match = re.search(r"Total Score: (\d)/2", text)
        if match:
            rating = int(match.group(1))
            if rating == 2:
                return 1
        return 0

    relevancy_rating = extract_total_score(model_response)
    return relevancy_rating


def get_question_verify_rating(model_response: str):
    """
    Extracts the question rating from the assigned rating by the teacher model
    Parameters:
    - model_response (str): Model response
    Return:
    question_verify_rating: Question rating
    """

    def extract_rating(text):
        # Regular expression to find the pattern "Rating: <number>"
        match = re.search(r"Rating:\s+(\d+)", text)
        if match:
            # Extract and return the rating number
            return int(match.group(1))
        return 0

    question_verify_rating = extract_rating(model_response)
    return question_verify_rating
