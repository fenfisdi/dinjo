"""Some helper functions and variables used to execute the examples of the
cmodels package.
"""
from typing import Union
from numbers import Number


def param_sample_range(
    param: Union[int, float],
    param_range_width: Union[int, float]
):
    """Define the bounds of some parameters."""
    error_message = (
        "param_sample_range(): param and param_range width must be"
        "numbers greater than zero"
    )

    if not (
        isinstance(param, Number) and isinstance(param_range_width, Number)
    ):
        raise TypeError(error_message)

    if param < 0 or param_range_width < 0:
        raise ValueError(error_message)

    return [param / param_range_width, param * param_range_width]
