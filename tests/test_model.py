import pytest

from dinjo.model import Variable, StateVariable, Parameter, ModelIVP
from dinjo.predefined.physics import ModelOscillator

VAR_PARAMS = 'foo', 'foo', 0.0


def test_variable():
    Variable(*VAR_PARAMS)
    assert True


def test_state_variable():
    StateVariable(*VAR_PARAMS)
    assert True


def test_parameter():
    Parameter(*VAR_PARAMS, bounds=[0.0, 1.0])


def test_parameter_bounds_not_explicit():
    Parameter(*VAR_PARAMS)


def test_get_parameter():
    p = Parameter(*VAR_PARAMS)
    p.bounds


def test_parameter_bounds_not_list():
    with pytest.raises(ValueError):
        Parameter(*VAR_PARAMS, bounds=(1.0, 0.0))


def test_parameter_bounds_not_number():
    with pytest.raises(ValueError):
        Parameter(*VAR_PARAMS, bounds=['a', 0.0])


def test_parameter_bounds_reversed():
    with pytest.raises(ValueError):
        Parameter(*VAR_PARAMS, bounds=[1.0, 0.0])


def test_parameter_init_val_not_in_bounds():
    with pytest.raises(ValueError):
        Parameter(*VAR_PARAMS, bounds=[0.1, 1.0])


class TestModelIVP:
    def setup(self):
        self.v = StateVariable(*VAR_PARAMS)
        self.p = Parameter(*VAR_PARAMS)
        self.model_ivp = ModelIVP(
            [self.v], [self.p], t_span=[0., 1.], t_steps=50
        )

    def test_state_vars_init_vals(self):
        sv_iv = self.model_ivp.state_variables_init_vals
        assert sv_iv == [self.v.initial_value]

    def test_parameters_init_vals(self):
        params_iv = self.model_ivp.state_variables_init_vals
        assert params_iv == [self.p.initial_value]

    def test_build_model_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.model_ivp.build_model(0, 0)

    def test_run_model(self, model_oscillator: ModelOscillator):
        model_oscillator.run_model()

    def test_run_model_params_not_list(
        self, model_oscillator: ModelOscillator
    ):
        with pytest.raises(TypeError):
            model_oscillator.run_model(parameters=set((0, 1)))
