# Standard
from dataclasses import dataclass, field
from typing import List, Optional

# Third Party
import ftfy

# Local
from fms_dgt.public.blocks.validators.privacy.metrics.text_util import (
    normalize_text,
    sent_tokenize,
)


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
