import copy
from typing import List

import math

import cmodel.seirv_model as build_model

class Optimizer:
    def __init__(
        self,
        model: build_model.CompartmentalModel,
        reference_state_variable: build_model.StateVariable,
        reference_values: List[float],
        reference_t_values: List[float]
    ) -> None:
        self.model = copy.deepcopy(model)

        self.reference_state_variable = reference_state_variable
        self.reference_values = reference_values

        self.reference_t_values = reference_t_values

    @property
    def reference_t_values(self):
        return self._reference_t_values
    
    @reference_t_values.setter
    def reference_t_values(self, reference_t_values_input: List[float]):
        if len(self.reference_values) != len(self.reference_t_values_input):
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
                self.model.t_span[0], self.reference_t_values[0], rel_tol=rel_tol
            ) and math.isclose(
                self.model.t_span[1], self.reference_t_values[-1], rel_tol=rel_tol
            )

        if not t_span_condition:
            raise AttributeError(
                "self.model.t_span and self.reference_t_values initial and "
                "final entries must coincide."
            )

        self._reference_t_values = reference_t_values_input


    def cost_function(self):
        """Function to be minimized by the optimizer.
        Initially this will be the 
        """
        self.model.t_eval = self.reference_t_values
        # solution = self.model.run_model
        pass
