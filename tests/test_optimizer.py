import os
import sys
from typing import List
from _pytest.python_api import approx

import pytest
from cmodel import seirv_model

this_dir_path = os.path.dirname(__file__)
sys.path.append(this_dir_path)

import cmodel.seirv_model as build_model
import cmodel.optimizer as optimizer
from test_seirv_model import (
    state_variables_source, state_variables, parameters_source, parameters,
    model_SEIRV,
)


@pytest.fixture
def infected(state_variables: build_model.StateVariable):
    infected = state_variables[2]
    return infected


@pytest.fixture
def reference_values():
    """
    Number of Infections in Colombia by day
    """
    return [
        1, 1, 1, 1, 3, 9, 9, 13, 22, 34, 54, 65, 93, 102, 128, 196, 231, 277,
        378, 470, 491, 539, 608, 702, 798, 906, 1065, 1161, 1267, 1406, 1485,
        1579, 1780, 2054, 2223, 2473, 2709, 2776, 2852, 2979, 3105, 3233,
        3439, 3439, 3792, 3977, 4149, 4356, 4561, 4881, 5142, 5379, 5597,
        5949, 6207, 6507, 7006, 7285, 7668, 7973, 8613, 8959, 9456, 10051,
        10495, 11063, 11613, 12272, 12930, 13610, 14216, 14939, 15574, 16295,
        16935, 17687, 18330, 19131, 20177, 21175, 21981, 23003, 24104, 24141,
        25406, 26734, 27219, 29384, 30593, 31935, 33466, 36759, 36759, 38149,
        40847, 40847, 42206, 43810, 45344, 46994, 48896, 53211, 53211, 55083,
        57202, 60387, 63454, 68836, 71367, 73760, 73760, 77313, 80811, 84660,
        91995, 91995, 95269, 98090, 102261, 106392, 109793, 113685, 117412,
        120281, 124494, 128638, 133973, 140776, 145362, 150445, 154277,
        159898, 165169, 182140, 190700, 197278, 204005, 211038, 218428,
        226373, 233541, 240795, 240795, 257101, 267385, 276055, 286020,
        295508, 306181, 317651, 327850, 334979, 345714, 357710, 367204,
        376870, 387481, 397623, 410453, 422519, 433805, 445111, 456689,
        468332, 476660, 489122, 502178, 513719, 522138, 522138, 541139, 551688
    ]


def test_optimizer_initialization(
    model_SEIRV: build_model.CompartmentalModel,
    infected: build_model.StateVariable,
    reference_values: List[float]
):
    try:
        seirv_optimizer = optimizer.Optimizer(
            model=model_SEIRV,
            reference_state_variable=infected,
            reference_values=reference_values,
            reference_t_values=model_SEIRV.t_eval
        )
    except AttributeError:
        assert False
    else:
        assert True


@pytest.fixture
def seirv_optimizer(
    model_SEIRV: build_model.CompartmentalModel,
    infected: build_model.StateVariable,
    reference_values: List[float]
) ->  optimizer.Optimizer:
    """
    docstring
    """
    seirv_optimizer = optimizer.Optimizer(
        model=model_SEIRV,
        reference_state_variable=infected,
        reference_values=reference_values,
        reference_t_values=model_SEIRV.t_eval
    )

    return seirv_optimizer


def test_optimizer_cost_function(
    seirv_optimizer: optimizer.Optimizer,
    model_SEIRV: build_model.CompartmentalModel
):

    root_mean_square = seirv_optimizer.cost_function(
        model_SEIRV.parameters_init_vals
    )
    print(root_mean_square)
    # Value calculated with Boris' definition of RootMeanSquare in Mathematica script
    assert root_mean_square == approx(6878.03, rel=0.99)


def test_optimizer_cost_function_cost_method_value_error(
    seirv_optimizer: optimizer.Optimizer,
    model_SEIRV: build_model.CompartmentalModel
):
    """Test error handling in
    :method:`cmodel.optimizer.Optimizer.cost_function`
    """
    try:
        root_mean_square = seirv_optimizer.cost_function(
            model_SEIRV.parameters_init_vals,
            cost_method='some_random_method'
        )
    except ValueError:
        assert True
    else:
        assert False


def test_optimizer_minimize_global(
    seirv_optimizer: optimizer.Optimizer,
):
    """Test if method:`cmodel.Optimize.minimize_global` method works with
    easiest optimization: fixed parameters via bounds.
    """
    minimization_algorithms = ['dual_annealing'] #, 'shgo', , 'brute',]
    parameters_init_vals = seirv_optimizer.model.parameters_init_vals
    for algorithm in minimization_algorithms:
        optimization = seirv_optimizer.minimize_global(algorithm=algorithm)
        assert optimization.x == approx(parameters_init_vals, rel=0.99)