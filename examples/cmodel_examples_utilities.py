"""Some helper functions and variables used to execute the examples of the
cmodels package.
"""
from datetime import datetime, timedelta
import os
import csv
from pathlib import Path

from typing import Any, List, Union
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


def setup_csv(
    file_complete_path: str,
    first_row: List[Any] = None
):
    """If ``file_complete_path`` does not exist, creates a [CSV] file and adds
    a first row to it."""

    file_complete_path = Path(file_complete_path)

    file_parent_directory_path = file_complete_path.parent
    if not os.path.isdir(file_parent_directory_path):
        raise ValueError(
            'setup_csv(): file_complete_path parameter.'
            f'{file_parent_directory_path.parent} must be a directory.'
        )

    if os.path.isfile(file_complete_path):
        return None

    if first_row and (not isinstance(first_row, list) or isinstance(first_row, tuple)):
        raise TypeError(
            'setup_csv(): first_row kwarg must be a list or a tuple.'
        )

    with open(file_complete_path, 'w', newline='') as file:
        if first_row:
            csv_writer = csv.writer(file)
            csv_writer.writerow(first_row)

    return True


def int_to_str_date(i: int, initial_date: datetime) -> str:
    return (datetime.utcnow() + timedelta(days=i)).strftime('%Y-%m-%d')
