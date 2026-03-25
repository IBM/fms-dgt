# Standard
from itertools import groupby
from types import GeneratorType
from typing import Iterable
import math

NDIGITS = 4


def type_check(*types, allow_none=False, **variables):
    """

    Check for acceptable types for a given object.  If the type check fails, a log message
    will be omitted at the error level on the log channel associated with this handler and a
    `TypeError` exception will be raised with an appropriate message.  This check should be used
    to check for appropriate variable types.  For example, to verify that an argument passed to
    a function that expects a string is actually an instance of a string.

    *types:  type or None
        Variadic arguments containing all acceptable types for `variables`.  If any values
        of `variable` are not any of `*types` then a log message will be omitted and a
        `TypeError` will be raised.  Multiple types may be specified as separate arguments.
        If no types are specified, then a `RuntimeError` will be raised.

    allow_none:  bool
        If `True` then the values of `variables` are allowed to take on the value of `None`
        without causing the type check to fail.  If `False` (default) then `None` values
        will cause the type check to fail.

    **variables:  object
        Variadic keyword arguments to be examined for acceptable type.  The name of the
        variable is used in log and error messages while its value is actually check against
        `types`.  Multiple keyword variables may be specified.  If no variables are
        specified, then a `ValueError` will be raised.

    Examples:
        # this will raise a `TypeError` because `foo` is not `None` or a `list` or `tuple`
        > type_check(None, list, tuple, foo='hello world')

        # this type check verifies that `foo` and `bar` are both strings
        > type_check(str, foo=foo, bar=bar)
    """

    if not types:
        raise ValueError("invalid type check: no types specified")

    if not variables:
        raise ValueError("invalid type check: no variables specified")

    for name, variable in variables.items():
        if allow_none and variable is None:
            continue

        # check if variable is an instance of one of `types`
        if not isinstance(variable, types):
            type_name = type(variable).__name__
            valid_type_names = tuple(typ.__name__ for typ in types)
            if allow_none:
                valid_type_names += (type(None).__name__,)

            valid_type_names = (
                valid_type_names[0] if len(valid_type_names) == 1 else list(valid_type_names)
            )
            # raise an appropriate exception
            raise TypeError(
                "type check failed: variable `{}` has type `{}`. Allowed type(s): `{}`".format(
                    name, type_name, valid_type_names
                )
            )


def type_check_all(*types, allow_none=False, **variables):
    """

    This type check is similar to `.type_check` except that it verifies that each variable
    in `**variables` is either a `list` or a `tuple` and then checks that *all* of the items
    they contain are instances of a type in `*types`.  If `allow_none` is set to `True`, then
    the variable is allowed to be `None`, but the items in the `list` or `tuple` are not.
    If 'allow_none' is set to False the input should not be empty or None.

    Examples:
        # this type check will verify that foo is a `list` or `tuple` containing only `int`s
        > foo = (1, 2, 3)
        > type_check(int, foo='hello world')

        # this type check allows `foo` to be `None`
        > type_check(None, foo=None)

        # this type check fails because `foo` contains `None`
        > type_check(None, int, foo=(1, 2, None, 3, 4))

        # this type check fails because `bar` contains a `str`
        # but not for any other reason
        > foo = [1, 2, 3]
        > bar = [4, 5, 'x']
        > baz = None
        > type_check('<COR40818868E>', None, int, foo=foo, bar=bar, baz=None)
    """

    if not types:
        raise ValueError("invalid type check: no types specified")

    if not variables:
        raise ValueError("invalid type check: no variables specified")

    top_level_types = (Iterable,)
    invalid_types = (
        str,
        GeneratorType,
    )  # top level types that will fail the type check

    for name, variable in variables.items():

        if allow_none and variable is None:
            continue

        if not allow_none and variable is not None and len(variable) == 0:
            raise ValueError("value check failed: variable `{}` cannot be empty`.".format(name))

        # log and raise if variable is not an Iterable
        if not isinstance(variable, top_level_types) or isinstance(variable, invalid_types):
            type_name = type(variable).__name__
            valid_type_names = tuple(typ.__name__ for typ in top_level_types)
            if allow_none:
                valid_type_names += (type(None).__name__,)

            valid_type_names = (
                valid_type_names[0] if len(valid_type_names) == 1 else list(valid_type_names)
            )
            raise TypeError(
                "type check failed: variable `{}` has type `{}`. Allowed types are: `{}`".format(
                    name, type_name, valid_type_names
                )
            )

        # raise if any item is not in list of valid types
        for item in variable:
            if not isinstance(item, types):
                type_name = type(item).__name__
                valid_type_names = list(typ.__name__ for typ in types)
                valid_type_names = (
                    valid_type_names[0] if len(valid_type_names) == 1 else valid_type_names
                )

                raise TypeError(
                    "type check failed: element of `{}` has type `{}` not in `{}`".format(
                        name, type_name, valid_type_names
                    )
                )


def value_check_one_of(item, one_of_list, name):
    if item not in one_of_list:
        raise NotImplementedError(f"{name} should be one of {one_of_list}")


def value_check(condition, msg, *args):
    """
    Check for acceptable values for a given object.  If this check fails, a log message will
    be omitted at the error level on the log channel associated with this handler and a
    `ValueError` exception will be raised with an appropriate message.  This check should be
    used for checking for appropriate values for variable instances.  For example, to check that
    a numerical value has an appropriate range.

    condition:  bool
        A boolean value that should describe if this check passes `True` or fails `False`.
        Upon calling this function, this is typically provided as an expression, e.g.,
        `0 < variable < 1`.

    msg:  str
        A string message describing the value check that failed.  If the empty string is
        provided (default) then no additional information will be provided.  Note that
        string interpolation can be lazily performed on `msg` using `{}` format syntax by
        passing additional arguments.  This is the preferred method for performing string
        interpolation on `msg` so that it is only done if an error condition is encountered.

    *args:
        Additional `{}`-style string formatting arguments to be lazily interpolated into
        the `msg` argument.
    """

    if condition is None or condition == "" or condition == []:
        interpolated_msg = msg.format(*args)
        raise ValueError("value check failed: {}".format(interpolated_msg))


def is_range_correct(value: float) -> bool:
    """
    The method checks that value is in the range [0..1]

    value : float
           a value

    Returns:
    -------
        True if a float value is in range [0..1] otherwise False
    """
    return 0.0 <= value <= 1.0


def is_positive(count: float) -> bool:
    """
    The method checks that a value is positive

    value : float
           a value

    Returns:
    -------
        True if a value is positive otherwise False
    """
    return count >= 0.0


def calculate_aggregated_score(scores: list[float], alpha: float = 1.0) -> float:
    """
    Calculate the aggregated score based on a list of individual scores.

    Parameters:
    - scores (list): A list of scores (float values between 0.0 and 1.0).
    - alpha (float): A scaling parameter controlling the influence of additional scores.

    Returns:
    - float: The aggregated score.
    """
    if not scores:
        return 0.0

    max_score = max(scores)
    remaining_scores = scores[:]
    remaining_scores.remove(max_score)

    # Contribution from other scores scaled by alpha
    additional_contribution = sum(
        (1 - max_score) * (score / len(scores)) * (1 - math.exp(-alpha))
        for score in remaining_scores
    )

    # Ensure the aggregated score is at least the max score plus contributions
    aggregated_score = max_score + additional_contribution

    # Cap the result at 1.0 (since scores are normalized between 0 and 1)
    return min(round(aggregated_score, NDIGITS), 1.0)


def select_highest_scores_from_group(matches_list, score_f, id_fields):
    # Sorting the data by the provided ID fields and then by score in descending order
    matches_list.sort(
        key=lambda x: tuple(getattr(x, id_f) for id_f in id_fields) + (-getattr(x, score_f),)
    )
    # Grouping by the provided ID fields
    highest_scores = [
        next(group)
        for _, group in groupby(
            matches_list, key=lambda x: tuple(getattr(x, id_f) for id_f in id_fields)
        )
    ]
    return highest_scores
