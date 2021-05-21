from typing import Any, List

import pytest

from dinjo.model import ModelIVP, Parameter, StateVariable
from dinjo.optimizer import Optimizer
from dinjo.predefined.physics import ModelOscillator


def test_bad_reference_state_variable(
    model_oscillator: ModelOscillator,
    oscillator_solution: Any,
    ho_mock_values: List[float],
):
    with pytest.raises(ValueError):
        Optimizer(
            model_oscillator,
            StateVariable('foo', 'foo', 0.0),
            ho_mock_values,
            oscillator_solution.t
        )


def test_reference_values_bad_length(
    model_oscillator: ModelOscillator,
    ho_state_vars: List[StateVariable],
    ho_mock_values: List[float],
    oscillator_solution: Any
):
    longer_ref_values = list(ho_mock_values[:])
    longer_ref_values.append(0)
    with pytest.raises(ValueError):
        Optimizer(
            model_oscillator,
            ho_state_vars[0],
            longer_ref_values,
            oscillator_solution.t
        )


def test_reference_t_values_bad_length(
    model_oscillator: ModelOscillator,
    ho_state_vars: List[StateVariable],
    ho_mock_values: List[float],
    oscillator_solution: Any
):
    longer_t_vlaues = list(oscillator_solution.t[:])
    longer_t_vlaues.append(0)
    with pytest.raises(ValueError):
        Optimizer(
            model_oscillator,
            ho_state_vars[0],
            ho_mock_values,
            longer_t_vlaues
        )


def test_unordered_reference_t_values(
    model_oscillator: ModelOscillator,
    ho_state_vars: List[StateVariable],
    ho_mock_values: List[float],
    oscillator_solution: Any
):
    longer_t_vlaues = list(oscillator_solution.t[:])
    longer_t_vlaues[1] = 10
    with pytest.raises(ValueError):
        Optimizer(
            model_oscillator,
            ho_state_vars[0],
            ho_mock_values,
            longer_t_vlaues
        )


def test_reference_t_values_bad_limits(
    model_oscillator: ModelOscillator,
    ho_state_vars: List[StateVariable],
    ho_mock_values: List[float],
    oscillator_solution: Any
):
    longer_t_vlaues = list(oscillator_solution.t[:])
    longer_t_vlaues[0] = 0.01
    with pytest.raises(ValueError):
        Optimizer(
            model_oscillator,
            ho_state_vars[0],
            ho_mock_values,
            longer_t_vlaues
        )


def test_reference_t_values_property(ho_optimizer: Optimizer):
    assert ho_optimizer.reference_t_values.any()


def test_cost_function_root_mean_square(
    model_oscillator: ModelOscillator,
    ho_optimizer: Optimizer
):
    ho_optimizer.cost_function(
        model_oscillator.parameters_init_vals,
        cost_method='root_mean_square'
    )


def test_cost_function_bad_cost_method(
    model_oscillator: ModelOscillator,
    ho_optimizer: Optimizer
):
    with pytest.raises(ValueError):
        ho_optimizer.cost_function(
            model_oscillator.parameters_init_vals,
            cost_method='foo'
        )


def test_optimize_algorithm_diff_evol(
    ho_optimizer: Optimizer
):
    ho_optimization = ho_optimizer.optimize(
        algorithm='differential_evolution'
    )
    assert ho_optimization.success


def test_optimize_algorithm_shgo(
    ho_optimizer: Optimizer
):
    ho_optimization = ho_optimizer.optimize(
        algorithm='shgo'
    )
    assert ho_optimization.success


@pytest.mark.skip('Takes more time. Comment this line to run this test.')
def test_optimize_algorithm_dual_annealing(
    ho_optimizer: Optimizer
):
    ho_optimization = ho_optimizer.optimize(
        algorithm='dual_annealing'
    )
    assert ho_optimization.success


def test_optimize_bad_algorithm_name(
    ho_optimizer: Optimizer
):
    with pytest.raises(ValueError):
        ho_optimizer.optimize(
            algorithm='foo'
        )


def test_optimize_bad_algotrithm_keyword_arguments(
    ho_optimizer: Optimizer
):
    with pytest.raises(ValueError):
        ho_optimizer.optimize(
            algorithm_kwargs={'foo': 'foo'}
        )


def test_optimize_runtime_error():
    class Foo(ModelIVP):
        def build_model(self, t, y, foo):
            return None
    variable_params = ('foo', 'foo', 0.0)
    foo_var = StateVariable(*variable_params)
    foo_param = Parameter(*variable_params)
    foo_model = Foo([foo_var], [foo_param], [0, 1], 2)
    foo_optimizer = Optimizer(foo_model, foo_var, [0, 1], [0, 1])
    with pytest.raises(RuntimeError):
        foo_optimizer.optimize()
