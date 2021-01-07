from typing import List

import pytest
import numpy as np

from cmodel import model, optimizer
from .test_seirv_model import (
    state_variables_source, state_variables, parameters_source, parameters,
    model_SEIRV,
)


@pytest.fixture
def infected(state_variables: model.StateVariable):
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
    model_SEIRV: model.CompartmentalModel,
    infected: model.StateVariable,
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
    except:
        assert False
    else:
        assert True


@pytest.mark.skip(
    "Error Handling should be improved in optimizer module. "
    "For example, AttributeError is raised inappropiately"
)
def test_optimizer_initialization_error_handling(
    model_SEIRV: model.CompartmentalModel,
    infected: model.StateVariable,
    reference_values: List[float]
):
    pass


@pytest.fixture
def seirv_optimizer(
    model_SEIRV: model.CompartmentalModel,
    infected: model.StateVariable,
    reference_values: List[float]
) ->  optimizer.Optimizer:
    """
    Optimizer for seirv model.
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
    model_SEIRV: model.CompartmentalModel
):

    root_mean_square = seirv_optimizer.cost_function(
        model_SEIRV.parameters_init_vals
    )
    print(root_mean_square)
    # Value calculated with Boris' definition of RootMeanSquare in Mathematica script
    assert root_mean_square == pytest.approx(6878.03, rel=0.99)


def test_optimizer_cost_function_cost_method_value_error(
    seirv_optimizer: optimizer.Optimizer,
    model_SEIRV: model.CompartmentalModel
):
    """Test error handling in
    :method:`cmodel.optimizer.Optimizer.cost_function` passing unsupported
    value for cost_method kwarg to cost_function() method in optimizer class
    """
    try:
        seirv_optimizer.cost_function(
            model_SEIRV.parameters_init_vals,
            cost_method='some_random_method'
        )
    except ValueError:
        assert True
    else:
        assert False


###############################################################################
#       Test optimizer minimize_global with a very simple one-parameter       #
#                       harmonic oscillator model                             #
###############################################################################


@pytest.fixture
def state_variables_oscillator() -> List[model.StateVariable]:
    """List of state variables for optimizator tests.
    """
    q = model.StateVariable(
        name='position', representation='q', initial_value=1.0
    )
    p = model.StateVariable(
        name='momentum', representation='p', initial_value=0.0
    )
    return [q, p]


@pytest.fixture
def parameters_oscillator() -> List[model.Parameter]:
    """
    List of parameters for oscillator optimizator tests.
    """
    omega = model.Parameter(
        name='frequency', representation='w', initial_value=5, bounds=[4, 8]
    )
    return [omega]


@pytest.fixture
def t_span_oscillator(
    parameters_oscillator: List[model.Parameter]
) -> List[float]:
    """Time span for oscillator optimizer tests."""
    omega = parameters_oscillator[0]
    t_span = [0, 2 * np.pi / omega.initial_value]
    return t_span


@pytest.fixture
def t_steps_oscillator() -> int:
    """Number of time seteps for oscillator optimizer tests."""
    t_steps = 50
    return int(t_steps)


@pytest.fixture
def model_oscillator(
    parameters_oscillator: List[model.Parameter],
    state_variables_oscillator: List[model.StateVariable],
    t_span_oscillator: List[float],
    t_steps_oscillator: int
) -> model.CompartmentalModel:
    """Build  compartmental model of Harmonic Oscillator."""

    # Build oscillator model
    class ModelOscillator(model.CompartmentalModel):
        def build_model(self, t, y, w):
            """Harmonic Oscillator differential equations
            """
            q, p  = y

            # Hamilton's equations
            dydt = [
                p,
                - (w ** 2) * q
            ]
            return dydt

    # Instantiate Model
    model_oscillator = ModelOscillator(
        state_variables=state_variables_oscillator,
        parameters=parameters_oscillator,
        t_span=t_span_oscillator,
        t_steps=t_steps_oscillator
    )

    return model_oscillator


@pytest.fixture
def solution_oscillator(model_oscillator: model.CompartmentalModel):
    """Numerical solution of oscillator model with initial value of
    parameters_oscillator.
    """
    return model_oscillator.run_model()


@pytest.fixture
def fake_position_reference_values_oscillator(
    solution_oscillator,
    t_steps_oscillator: int
) -> np.ndarray:
    """Generate fake position reference values by adding noise to a numerical
    solution.
    """
    # Get the position solution
    solution_position = solution_oscillator.y[0]

    # Build fake observation data from the solution (to test optimizer)
    noise_factor = 0.15         # Keep this number less than 0.2 or some tests will fail
    oscillator_fake_position_reference_values: np.ndarray = (
        solution_position
        + (2 * np.random.random(t_steps_oscillator) - 1) * noise_factor
    )

    return oscillator_fake_position_reference_values


@pytest.fixture
def optimizer_oscillator(
    model_oscillator: model.CompartmentalModel,
    state_variables_oscillator: List[model.StateVariable],
    fake_position_reference_values_oscillator: np.ndarray,
    solution_oscillator,
):
    """
    docstring
    """
    position_oscillator: model.StateVariable = state_variables_oscillator[0]
    optimizer_oscillator = optimizer.Optimizer(
        model=model_oscillator,
        reference_state_variable=position_oscillator,
        reference_values=fake_position_reference_values_oscillator,
        reference_t_values=solution_oscillator.t
    )
    return optimizer_oscillator


def test_optimizer_minimize_global_oscillator(
    optimizer_oscillator: optimizer.Optimizer,
):
    """
    Test :method:`Optimizer.minimize_global` mehtod using harmonic Oscillator
    example.
    """

    # Algorithms to test
    minimize_global_algorithms = ['differential_evolution', 'shgo']

    # 'dual_annealing' is very slow (approx 13 seconds for a one-parameter optimization).
    test_dual_annealing = False
    if test_dual_annealing:
        minimize_global_algorithms.append('dual_annealing')

    # Rune one assert statement for each algorithm
    for algorithm in minimize_global_algorithms:
        optimal_parameters_oscillator = optimizer_oscillator.minimize_global(
            algorithm=algorithm
        )
        expected_parameters_oscillator = optimizer_oscillator.model.parameters_init_vals

        assert optimal_parameters_oscillator.x == pytest.approx(
            expected_parameters_oscillator, rel=0.9
        )