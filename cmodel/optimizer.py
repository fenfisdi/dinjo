from typing import List

import cmodel.seirv_model as build_model
from cmodel.seirv_model import CompartmentalModel

class Optimizer:
    def __init__(
        self,
        model: build_model.CompartmentalModel,
        reference_state_variable: build_model.StateVariable,
        reference_values: List[float],
        reference_t_values: List[float]
    ) -> None:
        self.model = model

        self.reference_state_variable = reference_state_variable
        self.reference_values = reference_values

        self._reference_t_values = None
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
        self._reference_t_values = reference_t_values_input


    def cost_function(self):
        """Function to be minimized by the optimizer
        """

        pass