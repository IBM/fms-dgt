# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import List, Optional
import re

# Third Party
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
import ftfy
import numpy as np

# Define patterns for IP addresses and email addresses
PATTERNS = [
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP addresses
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
    r"\b(?:[A-Z]\.){1,3}\s[A-Z][a-z]+",  # J.D. Salinger
    r"(?<!\w)([A-Z]\.[A-Z]\.)(?!\s[A-Z][a-z])",  # r'\b(?:[A-Z]\.){1,3}(?=\s+[A-Z][a-z])'#r'\b(?:[A-Z]\.){1,3}\s[A-Z][a-z]+', #Initials
]
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\s?[AaPp]\.?[Mm]\.?)?\b")
STR_PUNCT = r"""!"#$%&'()*+,:;<=>?[\]^_`{|}~"""
PUNC_TRANS_TABLE = str.maketrans(STR_PUNCT, " " * len(STR_PUNCT))
STOP_WORDS = set(stopwords.words("english"))
END_PUNCT = [".", ";", ",", "?", "!"]
abbrev_list = {
    "e.g",
    "i.e",
    "dr",
    "mr",
    "mrs",
    "ms",
    "vs",
    "gen",
    "prof",
    "rev",
    "hon",
    "etc",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "u.n",
    "u.s",
    "st",
    "apt",
    "a.m",
    "p.m",
    "capt",
    "sen",
    "st",
    "p.p",
    "cf",
    "n.b",
    "inc",
    "u.k",
    "e.u",
    "sr",
    "jr",
    "ltd",
    "co",
    "corp",
    "mt",
    "sun",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
}
COMBINED_PATTERNS = "|".join(PATTERNS)
punkt_param = PunktParameters()
punkt_param.abbrev_types = abbrev_list
tokenizer = PunktSentenceTokenizer(punkt_param)


# --- Text identity and IDentiText classes (formerly text.py) ---


@dataclass
class TextIdentity:
    """
    Represents a unique identifier for a text object.

    Attributes
    ----------
    main_id : str | int
        The primary identifier of the text.

    secondary_id : str | int
        An optional secondary identifier for the text.

    collection : str | int
        The collection or group to which the text belongs.
    """

    main_id: str | int = field(init=True, default=None)
    secondary_id: Optional[str | int] = field(init=True, default=None)
    collection: Optional[str | int] = field(init=True, default=None)

    def __hash__(self):
        """
        Computes a unique hash for the identity.

        Returns
        -------
        int
            The computed hash value.
        """
        return hash((self.main_id, self.secondary_id, self.collection))

    def __lt__(self, other):
        """
        Compares two TextIdentity objects for ordering.

        Parameters
        ----------
        other : TextIdentity
            The other TextIdentity instance to compare.

        Returns
        -------
        bool
            True if the current instance is less than the other, otherwise False.
        """
        if not isinstance(other, TextIdentity):
            return NotImplemented
        return (self.collection, self.main_id, self.secondary_id) < (
            other.collection,
            other.main_id,
            other.secondary_id,
        )

    def __eq__(self, other):
        """
        Checks if two TextIdentity objects are equal.

        Parameters
        ----------
        other : TextIdentity
            The other TextIdentity instance to compare.

        Returns
        -------
        bool
            True if both instances have the same attributes, otherwise False.
        """
        if not isinstance(other, TextIdentity):
            return False
        return (
            other.main_id == self.main_id
            and other.secondary_id == self.secondary_id
            and other.collection == self.collection
        )


@dataclass
class IDentiText:
    """
    Represents a text object with an associated identity.

    Attributes
    ----------
    text : str
        The textual content.

    identity : TextIdentity
        The identity associated with the text.

    logprobs : list[float]
        A list of log probabilities related to text processing.

    _norm_sentences : List[str]
        A list of normalized sentences extracted from the text (internal use).

    _norm_masked_sentences: List[str]
        A list of normalized masked sentences

    _sentences : List[str]
        A list of tokenized sentences from the text (internal use).
    """

    text: Optional[str] = field(init=True, default=None)
    identity: TextIdentity = field(init=True, default_factory=TextIdentity)
    logprobs: Optional[List[float]] = field(init=True, default=None)

    _norm_sentences: List[str] = field(init=False, default=None)
    _norm_masked_sentences: List[str] = field(init=False, default=None)
    _sentences: List[str] = field(init=False, default=None)

    def __post_init__(self):
        self.text = ftfy.fix_text(self.text)

    def __eq__(self, other):
        """
        Checks if two IDentiText objects are equal based on identity collection and text.

        Parameters
        ----------
        other : IDentiText
            The other IDentiText instance to compare.

        Returns
        -------
        bool
            True if both instances have the same collection and text, otherwise False.
        """
        if not isinstance(other, IDentiText):
            return False
        return self.identity.collection == other.identity.collection and self.text == other.text

    def __hash__(self):
        """
        Computes a unique hash for the text object.

        Returns
        -------
        int
            The computed hash value.
        """
        return hash((self.identity.collection, self.text))

    def __repr__(self):
        """
        Generates a string representation of the IDentiText instance.

        Returns
        -------
        str
            A formatted string representation.
        """
        secondary_id = " - " + str(self.identity.secondary_id) if self.identity.secondary_id else ""
        return f"Collection: {self.identity.collection}, Id:{self.identity.main_id}{secondary_id}, Text: {self.text}"

    def split_to_sentences(self, include_stop_words: bool = True, split=True):
        """
        Splits the text into sentences and normalizes them.

        Parameters
        ----------
        include_stop_words : bool, optional
            Whether to include stop words in the normalized sentences (default is True).
        """

        if split:
            self._sentences = sent_tokenize(self.text)
        else:
            self._sentences = [self.text]

        self._norm_sentences = [
            normalize_text(ref, include_stop_words=include_stop_words) for ref in self._sentences
        ]

        self._norm_masked_sentences = self._norm_sentences

    @classmethod
    def create_from(
        cls,
        text: str,
        main_id: str,
        secondary_id: Optional[str] = None,
        collection: Optional[str] = None,
        logprobs: Optional[List[float]] = None,
    ):
        """
        Creates an IDentiText instance from provided values.

        Parameters
        ----------
        text : str
            The text content.

        main_id : str
            The primary identifier.

        secondary_id : str, optional
            The secondary identifier.

        collection : str, optional
            The collection the text belongs to.

        logprobs : list[float], optional
            Log probabilities associated with the text processing.

        Returns
        -------
        IDentiText
            A new instance of IDentiText.
        """
        return IDentiText(text, TextIdentity(main_id, secondary_id, collection), logprobs)


# --- Text utility functions (formerly text_util.py) ---


def is_stop_word(word: str) -> bool:
    return word in STOP_WORDS


def remove_punc(text, date_pattern=DATE_PATTERN):

    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'[."]+', " ", text)
    dates = date_pattern.findall(text)

    for i, date in enumerate(dates):
        text = text.replace(date, f"DATEPLACEHOLDER{i}")

    ip_addresses = re.findall(IP_PATTERN, text)
    for i, ip in enumerate(ip_addresses):
        text = text.replace(ip, f"IP{i}")

    times = re.findall(TIME_PATTERN, text)
    for i, time in enumerate(times):
        text = text.replace(time, f"TIME{i}")

    text = text.translate(PUNC_TRANS_TABLE)

    for i, time in enumerate(times):
        text = text.replace(f"TIME{i}", time)

    for i, ip in enumerate(ip_addresses):
        text = text.replace(f"IP{i}", ip)

    for i, date in enumerate(dates):
        text = text.replace(f"DATEPLACEHOLDER{i}", date)

    text = remove_ending_punc(text)
    return text


def remove_ending_punc(text):
    if text and len(text) > 1 and text[-1] in END_PUNCT:
        return text[:-1]
    return text


def white_space_fix(text: str) -> str:
    """
    Removes extra white space.
    text : str
        string/text to be fixed
    Returns:
    --------
        string/text with extra whitespaces removed

    """
    return " ".join(text.split())


def normalize_text(text: str, include_stop_words=False) -> str:
    """
    Lower text, remove punctuation as well as \\n and \\t, removes extra white space and remove stop words if required
    text : str
        string/text to be normalized
    include_stop_words : bool
        this flag allows to normalize text with or without stop words. Default is do not include stop words
    Returns:
    --------
        string/text without punctuation and without tabs and \\n
    """

    def remove_stop_words(text: str) -> str:
        """
        Remove stop words
        text : str
            string/text to be normalized

        Returns:
        --------
            string/text without stop words
        """
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
        new_string = " ".join(filtered_words)
        return new_string

    def lower(text: str) -> str:
        """
        Lower text

        text : str
            string/text to be lowered

        Returns:
        --------
            string/text with lower case
        """
        return text.lower()

    return (
        white_space_fix(lower(remove_punc(text)))
        if include_stop_words
        else white_space_fix(remove_stop_words(lower(remove_punc(text))))
    )


def preprocess_text(text, combined_pattern=COMBINED_PATTERNS):
    if not text:
        return text
    text = re.sub(r"(?<=\w)\-", " ", text)  # Remove hyphens, etc. that are inside words
    # Protect known exceptions like A.M. and P.M.
    protected = re.sub(
        r"(?<!\w)(A\.M\.|P\.M\.|U\.S\.|U\.K\.|E\.U\.|U\.N\.)(?!\w)",
        lambda m: m.group(0).replace(".", "<ignoreprd>"),
        text,
    )
    protected = re.sub(combined_pattern, lambda m: m.group(0).replace(".", "<prd>"), protected)
    # Restore A.M. and P.M.
    protected = protected.replace("<ignoreprd>", ".")
    return protected


def fallback_split(text):
    return re.split(
        r"(?<=\b(?:[Aa]\.[Mm]\.|[Pp]\.[Mm]\.|[Uu]\.[Kk]\.|[Ee]\.[Uu]\.|[Ee]tc\.)) (?=(?![A-Z]{2,5}\b)[A-Z])",
        text,
    )


def normalize_ellipses(text):
    return re.sub(r"\.{2,}", ".", text)


def sent_tokenize(text: str) -> List[str]:
    sentences = tokenizer.tokenize(preprocess_text(normalize_ellipses(text)))
    final_sentences = []
    for sent in sentences:
        sub_sents = fallback_split(sent)
        final_sentences.extend(sub_sents)
    return [sentence.replace("<prd>", ".") for sentence in final_sentences]


def clean_special_punct(text):
    text = (
        text.replace(" ..", " ")
        .replace(".. ", " ")
        .replace("..'", " ")
        .replace("\u00bb", " ")
        .replace("''", " ")
        .replace("..", " ")
        .replace('."', " ")
        .replace(',"', " ")
        .replace('"', " ")
        .replace("__", " ")
        .replace(";", " ")
        .replace("..!", " ")
        .replace(".!", " ")
        .replace("!", " ")
        .replace("' ',", " ")
        .replace("? ", " ")
    )
    text = text.replace("\n", " ").replace("\t", " ")
    if text and text[-1] in [".", ":", ";", "-", ",", "'", '"']:
        text = text[:-1]
    return text.strip().lstrip()


def find_common_non_overlapping_ngrams(target_words, source_words, min_len=2) -> list:
    """
    Finds non-overlapping ngram-words

    Returns
    -------
    list
        non-overlapping ngram-words
    """

    max_n = min(len(target_words), len(source_words))

    used1 = np.zeros(len(target_words), dtype=bool)
    used2 = np.zeros(len(source_words), dtype=bool)
    results = []

    for n in range(max_n, 0, -1):
        if len(target_words) < n or len(source_words) < n:
            continue

        win1 = np.lib.stride_tricks.sliding_window_view(target_words, n)
        win2 = np.lib.stride_tricks.sliding_window_view(source_words, n)

        for i in range(len(win1)):
            if used1[i : i + n].any():
                continue

            for j in range(len(win2)):
                if used2[j : j + n].any():
                    continue

                if np.array_equal(win1[i], win2[j]):
                    # Return the list of indices for the n-gram match (indexes only, no words)
                    index_tuple = list(range(i, i + n))
                    if len(index_tuple) >= min_len:
                        results.append(index_tuple)
                    used1[i : i + n] = True
                    used2[j : j + n] = True
                    break

    return results
