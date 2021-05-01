from typing import List

from cmodel.model import CompartmentalModel


class SEIR_model(CompartmentalModel):

    def build_model(
        self, t, y,
        Lmbd, mu, omega, gamma, alpha, chi, beta_E, beta_I
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, E, I, R = y

        dydt = [
            Lmbd - beta_E * S * E - beta_I * S * I - (chi + mu) * S,
            beta_E * S * E + beta_I * S * I - E / alpha - mu * E,
            E / alpha - (gamma + omega + mu) * I,
            gamma * I + chi * S - mu * R
        ]

        return dydt
