from typing import List

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Variable:
    """Represents a state variable"""
    def __init__(
        self, name: str, representation: str, initial_value: float = 0,
        *args, **kwargs
    ) -> None:
        self.name = name
        self.representation = representation
        self.initial_value = initial_value


class StateVariable(Variable):
    pass


class Parameter(Variable):
    pass


class CompartmentalModel:
    pass


class ModelSEIRV(CompartmentalModel):
    def __init__(
        self,
        state_variables: List[StateVariable],
        parameters: List[Parameter],
        t_span: List[float] = [0, 10],
        t_steps: int = 50,
        t_eval: List[float] = None
    ) -> None:
        self.state_variables = state_variables
        self.parameters = parameters
        self.t_span = t_span
        self.t_steps = t_steps
        self.t_eval = t_eval if t_eval else list(np.linspace(*t_span, t_steps))
    
    def build_model(
        self, t, y,
        Lambda, mu, alpha, omega, gamma, xi1, xi2, sigma, b1, b2, b3, c1, c2, c3
    ) -> List[float]:
        """Returns the vector field dy/dt evaluated at a given point in phase space"""
        
        S, Ex, If, R, V = y

        def beta(x, b, c):
            return b / (1. + c * x)

        principal_flux = S * (beta(Ex, b1, c1) * Ex + beta(If, b2, c2) * If + beta(V, b3, c3) * V)

        dydt = [
            Lambda - principal_flux - S * mu,
            principal_flux - (alpha + mu) * Ex,
            alpha * Ex - (omega + gamma + mu) * If,
            gamma * If - mu * R,
            xi1 * Ex + xi2 * If - sigma * V
        ]

        return dydt

    def run_model(self, method:str = 'RK45'):
        """Integrate model using ``scipy.integrate.solve_ivp``"""
        
        initial_conditions = [sv.initial_value for sv in self.state_variables]
        parameters = tuple([param.initial_value for param in self.parameters])

        solution = solve_ivp(
            fun=self.build_model,
            t_span=self.t_span,
            y0=initial_conditions,
            method=method,
            t_eval=self.t_eval,
            args=parameters
        )

        return solution
