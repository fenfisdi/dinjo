import copy
from typing import Callable, List, Dict

import math
import numpy as np

from . import seirv_model as build_model

class Optimizer:
    def __init__(
        self,
        model: build_model.CompartmentalModel,
        reference_state_variable: build_model.StateVariable,
        reference_values: List[float],
        reference_t_values: List[float],
        integration_method: str = 'RK45'
    ) -> None:
        self.model = copy.deepcopy(model)

        self.reference_state_variable = reference_state_variable
        self.reference_values = reference_values

        self.reference_t_values = reference_t_values

        self.integration_method = integration_method

        # Index of reference state variable in state variable list
        try:    
            self._reference_state_variable_index: int = \
                model.state_variables.index(
                    reference_state_variable
                )
        except ValueError:
            raise AttributeError(
                "self.reference_state_variable must be in model.state_variables"
            )

    @property
    def reference_t_values(self):
        return self._reference_t_values
    
    @reference_t_values.setter
    def reference_t_values(self, reference_t_values_input: List[float]):
        if len(self.reference_values) != len(reference_t_values_input):
            raise AttributeError(
                "self.reference_values and self.reference_t_values must have the same length"
            )

        for i, t in enumerate(reference_t_values_input[:-1]):
            if not t < reference_t_values_input[i+1]:
                raise AttributeError(
                    "self.reference_t_values must be a list of floats in increasing order"
                )

        rel_tol = 0.999

        t_span_condition = math.isclose(
                self.model.t_span[0], reference_t_values_input[0], rel_tol=rel_tol
            ) and math.isclose(
                self.model.t_span[1], reference_t_values_input[-1], rel_tol=rel_tol
            )

        if not t_span_condition:
            raise AttributeError(
                "self.model.t_span and self.reference_t_values initial and "
                "final entries must coincide."
            )

        self._reference_t_values = reference_t_values_input


    def cost_function(
        self,
        parameters: List[float],
        cost_method: str = 'root_mean_square'
    ):
        """Function to be minimized by the optimizer.
        Initially this will be the root mean square of the difference between
        the observations and the numerical solution.

        Parameters
        ----------
        parameters : str
            parameters of the model to be minimized
        cost_method : str
            Must be one of ``'root_mean_square'``, etc.
        """
        self.model.t_eval = self.reference_t_values
        solution = self.model.run_model(
            parameters=parameters, method=self.integration_method
        )
        
        # Get appropiate state variable solution in order to compare with
        # reference values
        solution_reference = solution.y[self._reference_state_variable_index]

        def rms(solution_reference):
            """Root mean square of the difference between ``reference_values``
            and ``solution_reference``.
            
            """
            diff_squared: np.ndarray = (
                (np.array(solution_reference) - np.array(self.reference_values)) ** 2
            )
            root_mean_square: float = np.mean(diff_squared) ** 0.5
            return root_mean_square

        cost_method_dict: Dict[str, Callable] = {
            'root_mean_square': rms,
        }
        
        try:
            cost_func = cost_method_dict[cost_method]
        except KeyError:
            raise ValueError(
                f"cost_method value '{cost_method}' not permitted. "
                f"Valid options options are {cost_method_dict.keys()}."
            )

        return cost_func(solution_reference)
