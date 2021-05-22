from typing import List
from pytest import approx

from dinjo.model import ModelIVP, StateVariable, Parameter
from dinjo.predefined.epidemiology import (
    ModelSEIR, ModelSimpleSEIRV, ModelSIR, ModelSEIRV
)


class TestEpidemiologyModels:
    def setup(self):
        S = StateVariable('Succeptible', 'S', 0.5)
        E = StateVariable('Exposed', 'E', 0.5)
        Inf = StateVariable('Infected', 'I', 0.5)
        R = StateVariable('recovered', 'R', 0.5)
        V = StateVariable('Virus', 'V', 0.5)

        Lmbd = Parameter('Lambda', r'\Lambda', 1 / 2)
        mu = Parameter('mu', r'\mu', 1 / 2)
        inv_alpha = Parameter('inverse_alpha', r'\frac{1}{\alpha}', 1 / 2)
        omega = Parameter('omega', r'\omega', 1 / 2)
        gamma = Parameter('gamma', r'\gamma', 1 / 2)
        xi_E = Parameter('xi_E', r'\xi_E', 1 / 2)
        xi_I = Parameter('xi_I', r'\xi_I', 1 / 2)
        sigma = Parameter('sigma', r'\sigma', 1 / 2)
        beta_E = Parameter('beta_E', r'\beta_E', 1 / 2)
        beta_I = Parameter('beta_I', r'\beta_I', 1 / 2)
        beta_V = Parameter('beta_V', r'\beta_V', 1 / 2)
        c_E = Parameter('c_E', r'\c_E', 1 / 2)
        c_I = Parameter('c_I', r'\c_I', 1 / 2)
        c_V = Parameter('c_V', r'\c_V', 1 / 2)

        chi = Parameter('chi', r'\chi', 1 / 2)
        eta = Parameter('eta', r'\eta', 1 / 2)
        Pi = Parameter('Pi', r'\Pi', 1 / 2)
        tau = Parameter('tau', r'\tau', 1 / 2)

        self.sir_state_vars = [S, Inf, R]
        self.sir_params = [Lmbd, mu, omega, gamma, chi, eta, Pi, tau]

        self.seir_state_vars = [S, E, Inf, R]
        self.seir_params = [
            Lmbd, mu, omega, gamma, inv_alpha, chi, beta_E, beta_I
        ]

        self.seirv_state_vars = [S, E, Inf, R, V]
        self.seirv_params = [
            Lmbd, mu, inv_alpha, omega, gamma, xi_E, xi_I, sigma, beta_E,
            beta_I, beta_V, c_E, c_I, c_V
        ]

        self.seirv_simple_params = [
            Lmbd, mu, omega, gamma, inv_alpha, xi_E, xi_I, sigma,
            beta_E, beta_I, beta_V
        ]

        self.t_span = [0, 1]
        self.t_steps = 2

    def assert_approx_dydt(
        self,
        model: ModelIVP,
        approx_dydt: List[float],
        rel_tol: float = 0.99
    ):
        dydt = model.build_model(
            self.t_span[0],
            model.state_variables_init_vals,
            *model.parameters_init_vals
        )

        print(dydt)

        assert dydt == approx(approx_dydt, rel_tol)

    def test_seir(self):
        model_seir = ModelSEIR(
            self.seir_state_vars, self.seir_params, self.t_span, self.t_steps
        )
        self.assert_approx_dydt(model_seir, [-0.25, -0.25, -0.5, 0.25])

    def test_seirv(self):
        model_seirv = ModelSEIRV(
            self.seirv_state_vars, self.seirv_params, self.t_span, self.t_steps
        )
        self.assert_approx_dydt(model_seirv, [-0.05, -0.2, -0.5, 0.0, 0.25])

    def test_seirv_fixed(self):
        model_seirv_fixed = ModelSimpleSEIRV(
            self.seirv_state_vars, self.seirv_simple_params, self.t_span,
            self.t_steps
        )
        self.assert_approx_dydt(
            model_seirv_fixed,
            [-0.125, -0.125, -0.5, 0.0, 0.25]
        )

    def test_sir(self):
        model_sir = ModelSIR(
            self.sir_state_vars, self.sir_params, self.t_span, self.t_steps
        )
        self.assert_approx_dydt(model_sir, [0.125, -0.875, -0.25])
