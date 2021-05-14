from typing import List

import pytest
import numpy as np

from dinjo import model, optimizer
from .test_seirv_model import (                                                # noqa: F401
    state_variables_source, state_variables, parameters_source, parameters,
    model_SEIRV,
)
from examples.seirv_model_colombia import infected_reference_col


@pytest.fixture
def infected(state_variables: model.StateVariable):                            # noqa: F811
    infected: model.StateVariable = state_variables[2]
    return infected


@pytest.fixture
def reference_values():
    """
    Number of Infections in Colombia by day
    """
    return infected_reference_col


def test_optimizer_initialization(
    model_SEIRV: model.CompartmentalModel,                                     # noqa: F811
    infected: model.StateVariable,
    reference_values: List[float]
):
    try:
        optimizer.Optimizer(
            model=model_SEIRV,
            reference_state_variable=infected,
            reference_values=reference_values,
            reference_t_values=model_SEIRV.t_eval
        )
    except AttributeError:
        assert False
    except Exception:
        assert False
    else:
        assert True


@pytest.mark.skip(
    "Error Handling should be improved in optimizer module. "
    "For example, AttributeError is raised inappropiately"
)
def test_optimizer_initialization_error_handling(
    model_SEIRV: model.CompartmentalModel,                                     # noqa: F811
    infected: model.StateVariable,
    reference_values: List[float]
):
    raise NotImplementedError


@pytest.fixture
def seirv_optimizer(
    model_SEIRV: model.CompartmentalModel,                                     # noqa: F811
    infected: model.StateVariable,
    reference_values: List[float]
) -> optimizer.Optimizer:
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
    model_SEIRV: model.CompartmentalModel                                      # noqa: F811
):

    root_mean_square = seirv_optimizer.cost_function(
        model_SEIRV.parameters_init_vals
    )
    print(root_mean_square)
    # Value calculated with Boris' definition of RootMeanSquare in Mathematica script
    assert root_mean_square == pytest.approx(6878.03, rel=0.99)


def test_optimizer_cost_function_cost_method_value_error(
    seirv_optimizer: optimizer.Optimizer,
    model_SEIRV: model.CompartmentalModel                                      # noqa: F811
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
            q, p = y

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
    test_dual_annealing: bool = False
):
    """
    Test :method:`Optimizer.minimize_global` mehtod using harmonic Oscillator
    example.
    """

    # Algorithms to test
    minimize_global_algorithms = ['differential_evolution', 'shgo']

    # 'dual_annealing' is very slow (approx 13 seconds for a one-parameter optimization).
    if test_dual_annealing:
        minimize_global_algorithms.append('dual_annealing')

    # Rune one assert statement for each algorithm
    for algorithm in minimize_global_algorithms:
        optimal_parameters_oscillator = optimizer_oscillator.minimize(
            algorithm=algorithm
        )
        expected_parameters_oscillator = optimizer_oscillator.model.parameters_init_vals

        assert optimal_parameters_oscillator.x == pytest.approx(
            expected_parameters_oscillator, rel=0.9
        )
