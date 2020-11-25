import numbers
from typing import List, Optional

import numpy as np
from numpy.lib.arraysetops import isin
from scipy.integrate import solve_ivp


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
    def __init__(
        self, name: str, representation: str, initial_value: float = 0,
        *args, bounds: Optional[List[float]] = None, **kwargs
    ) -> None:
        super().__init__(name, representation, initial_value)
        self.bounds = bounds if bounds else [initial_value, initial_value]
    
    @property
    def bounds(self):
        """Bounds of the parameters. Needed for optimization.
        """
        return self._bounds
    
    @bounds.setter
    def bounds(self, bounds_input):
        attr_err_message = "bounds must be a list of two increasing numbers."
        init_val_not_in_bounds_range = "initial_value must be in the range defined by bounds."
       
        type_check: bool = (
            isinstance(bounds_input, list)
            and len(bounds_input) == 2
            and isinstance(bounds_input[0], numbers.Number)
            and isinstance(bounds_input[1], numbers.Number)
        )

        if not type_check:
            raise AttributeError(attr_err_message)

        order_check: bool = bounds_input[0] <= bounds_input[1]
        
        if not order_check:
            raise AttributeError(attr_err_message)

        if not (
            bounds_input[0] <= self.initial_value 
            and bounds_input[1] >= self.initial_value
        ):
            raise AttributeError(init_val_not_in_bounds_range)
        
        self._bounds = bounds_input


class CompartmentalModel:
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
    
    def _get_variable_init_vals(
        self, variables: List[Variable]
    ) -> List[float]:
        return [var.initial_value for var in variables]

    @property
    def state_variables_init_vals(self) -> List[float]:
        """Get the values of the model's state variables initial values in the
        order they are currently stored in ``self.state_variables``."""
        return self._get_variable_init_vals(self.state_variables)

    @property
    def parameters_init_vals(self):
        """Get the values of the model's parameters initial values in the
        order they are currently stored in ``self.parameters``."""
        return self._get_variable_init_vals(self.parameters)

    def build_model(self, t, y, *args):
        pass

    def run_model(
        self,
        parameters: List[float] = None,
        method:str = 'RK45'
    ):
        """Integrate model using ``scipy.integrate.solve_ivp``"""
        
        parameters = (
            parameters if parameters is not None else self.parameters_init_vals
        )

        parameters_permitted_types = (list, tuple, np.ndarray)
        parameters_type_is_permitted = True
        
        for permitted_type in parameters_permitted_types:
            parameters_type_is_permitted += isinstance(parameters, permitted_type)
        
        if not parameters_type_is_permitted:
            raise TypeError(
                "parameters must be a list, tuple or numpy.ndarray"
            )

        solution = solve_ivp(
            fun=self.build_model,
            t_span=self.t_span,
            y0=self.state_variables_init_vals,
            method=method,
            t_eval=self.t_eval,
            args=parameters
        )

        return solution


class ModelSEIRV(CompartmentalModel):
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
