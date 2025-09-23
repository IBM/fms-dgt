# Standard
from copy import deepcopy
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, Set, Tuple, Type
import collections
import itertools
import time

# Local
from fms_dgt.utils import dgt_logger


def retry(
    on_exceptions: Tuple[Type[Exception]],
    max_retries: int = 3,
    backoff_time: float = 10.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Optional[Callable[[Exception, float], Any]] = None,
):
    """Retry on an LLM Provider's rate limit error with exponential backoff
    For example, to use for OpenAI, do the following:
    ```
    from openai import RateLimitError

    # Recommend specifying max_retries to avoid infinite loops!
    @retry_on_specific_exceptions([RateLimitError], max_retries=3)
    def completion(...):
        # Wrap OpenAI completion function here
        ...
    ```
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize necessary variables
            sleep_timer = backoff_time
            attempt = 0

            # Keep retrying till max retries are attempted
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except on_exceptions as e:
                    # Trigger exception callback
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_timer)

                    # Wait for sleep timer before retrying
                    time.sleep(sleep_timer)

                    # Increament sleep timer and attempt counter
                    sleep_timer *= backoff_multiplier
                    attempt += 1

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Initialize necessary variables
            sleep_timer = backoff_time
            attempt = 0

            # Keep retrying till max retries are attempted
            while attempt < max_retries:
                try:
                    return await func(*args, **kwargs)
                except on_exceptions as e:
                    # Trigger exception callback
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_timer)

                    # Wait for sleep timer before retrying
                    time.sleep(sleep_timer)

                    # Increament sleep timer and attempt counter
                    sleep_timer *= backoff_multiplier
                    attempt += 1

        return async_wrapper if iscoroutinefunction(func) else wrapper

    return decorator


def chunks(iter, n: int = 0, fn=None):
    """
    Divides an iterable into chunks of specified size or based on a given function.
    Useful for batching

    Parameters:
    - iter: The input iterable to be divided into chunks.
    - n: An integer representing the size of each chunk. Default is 0.
    - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

    Returns:
    An iterator that yields chunks of the input iterable.

    Example usage:
    ```
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for chunk in chunks(data, 3):
        print(chunk)
    ```
    Output:
    ```
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]
    ```
    """
    arr = []
    for i, x in enumerate(iter):
        arr.append(x)
        if len(arr) == (fn(i, iter) if fn else n):
            yield arr
            arr = []

    if arr:
        yield arr


class Grouper:
    """
    takes an array `arr` and function `fn` and returns a dictionary
    with keys fn(ob) for each ob in `arr` and with values `self.arr[key]` a list of all
    objects in `arr` satisfying `key == fn(ob)`.
    """

    def __init__(self, arr, fn) -> None:
        # make copy
        arr = arr + []
        # self.orig_arr = arr
        self.size = len(arr)
        arr = list(enumerate(arr))

        def group_return_dict(arr, fn):
            res = collections.defaultdict(list)

            for ob in arr:
                res[fn(ob)].append(ob)
            return res

        arr = group_return_dict(arr, lambda x: fn(x[1]))

        # self.arr has format Dict[Tuple[int, <entry from orig. arr>]]
        self.arr = arr
        self._grouped = None

    def get_grouped(self):
        # return the contents but not indices for our grouped dict.
        if self._grouped:
            return self._grouped
        grouped = {}
        for key in self.arr.keys():
            # drop the index from each element of self.arr
            grouped[key] = [y[1] for y in self.arr[key]]
        self._grouped = grouped
        return grouped

    def get_original(self, grouped_dict):
        # take in a grouped dictionary with e.g. results for each key listed
        # in the same order as the instances in `self.arr`, and
        # return the results in the same (single list) order as `self.orig_arr`.
        res = [None] * self.size
        cov = [False] * self.size
        # orig = [None] * self.size

        assert grouped_dict.keys() == self.arr.keys()

        for key in grouped_dict.keys():
            for (ind, _), v in zip(self.arr[key], grouped_dict[key]):
                res[ind] = v
                cov[ind] = True
                # orig[ind] = _

        assert all(cov)
        # assert orig == self.orig_arr

        return res


def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """

    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]


def remap(
    dictionary: dict, mapping: dict, override: bool = False, raised: Set[str] = set()
) -> dict:
    """
    Shrink the dictionary via remapping from_fields -> to_fields in a dictionary.

    Args:
        dictionary (dict): dictionary to adjust
        mapping (dict): {"to_field_1": ["from_field_1", ...], "to_field_2": ["from_field_*", ...]}
        override (bool, optional): override existing field value in the dictionary. Defaults to False.
        raised (Set[str], optional): retains previously raised warning messages.

    NOTE: Retaining previously raised warning via a set object (inherently shared across invocations) is not a recommended practice. Avoid repeating this pattern.

    Returns:
        (dict): Remapped dictionary
    """
    # Step 1: Create deep copy of dictionary to remap
    remapped_dictionary = deepcopy(dictionary)

    # Step 2: Iterate over mappings
    for to_field, from_fields in mapping.items():
        # Step 2.a: Skip remapping, if the to_field already exists in the dictionary and override is set to false
        if remapped_dictionary.get(to_field) and not override:
            warning_msg = f"Retaining detected value for {to_field}"
            if warning_msg not in raised:
                dgt_logger.warning(warning_msg)
                raised.add(warning_msg)
            continue

        # Step 2.b: Remap to first from_field and remove other from_fields
        remapped: bool = False
        for from_field in from_fields:
            if remapped_dictionary.get(from_field):
                value_to_remap = remapped_dictionary.pop(from_field)

                if not remapped:
                    remapped_dictionary[to_field] = value_to_remap
                    remapped = True

    # Step 3: Return
    return remapped_dictionary
