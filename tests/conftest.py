from typing import Any, List
import pytest

from numpy import pi
from numpy.random import random

from dinjo.model import StateVariable, Parameter
from dinjo.optimizer import Optimizer
from dinjo.predefined.physics import ModelOscillator


@pytest.fixture(scope='session')
def ho_state_vars():
    # Harmonic Oscillator Initial Value Problem
    q = StateVariable(
        name='position', representation='q', initial_value=1.0
    )
    p = StateVariable(
        name='momentum', representation='p', initial_value=0.0
    )
    return [q, p]


@pytest.fixture(scope='session')
def ho_params():
    # Define Paramters
    omega = Parameter(
        name='frequency', representation='w', initial_value=2 * pi, bounds=[4, 8]
    )
    return [omega]


@pytest.fixture(scope='session')
def t_span():
    return [0, 1]


@pytest.fixture(scope='session')
def t_steps():
    return 50


@pytest.fixture(scope='session')
def model_oscillator(ho_state_vars, ho_params, t_span, t_steps):
    # Instantiate the IVP class with appropiate State Variables and Parameters
    return ModelOscillator(
        state_variables=ho_state_vars,
        parameters=ho_params,
        t_span=t_span,
        t_steps=t_steps
    )


@pytest.fixture(scope='session')
def oscillator_solution(model_oscillator: ModelOscillator):
    return model_oscillator.run_model()


@pytest.fixture(scope='session')
def ho_mock_values(oscillator_solution: Any, t_steps: List[float]):
    noise_factor = 0.3
    return (
        oscillator_solution.y[0]
        + (2 * random(t_steps) - 1) * noise_factor
    )


@pytest.fixture(scope='session')
def ho_optimizer(
    model_oscillator: ModelOscillator,
    oscillator_solution: Any,
    ho_mock_values: List[float],
    ho_state_vars: List[StateVariable]
):
    return Optimizer(
        model_oscillator,
        ho_state_vars[0],
        ho_mock_values,
        oscillator_solution.t
    )
