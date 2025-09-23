# Standard
import ast
import re


def post_process_questions(questions: str):
    raw_questions = re.split(r"### Question \d+:", questions)
    return [raw_q.strip() for raw_q in raw_questions if raw_q.strip() != ""]


def post_process_context(context: str):
    # Regular expression to find context
    pattern = re.compile(r"\[Start of Context\](.*?)(?=\[End of Context\])", re.DOTALL)

    # Extracting contexts into a list
    matches = pattern.findall(context)

    # Check if matches are found
    if matches:
        contexts = [match.strip() for match in matches]
        return contexts[0]

    return None


def parse_response_string(judgement):
    """
    Parses the 'judgement' field of the input sample dictionary and extracts the rating.
    Parameters:
    - sample (string): A string containing response information with a 'judgement' field.
    Returns:
    int: The input sample dictionary with an additional 'qrating' field. If the 'judgement' field does not contain a 'Rating', the 'qrating' field is set to -1.
    Note:
    - The function uses regex to search for the rating pattern in the 'judgement' field.
    - If 'Rating' is found, the function tries to extract the score using a primary and a backup pattern.
    - The extracted rating is stored in the 'qrating' field of the input sample.
    - If no rating is found or the extraction fails, the 'qrating' field is set to -1.
    """

    one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")
    if "Rating" in judgement:
        match = re.search(one_score_pattern, judgement)
        if not match:
            match = re.search(one_score_pattern_backup, judgement)
        if match:
            return float(ast.literal_eval(match.groups()[0]))
    return -1
