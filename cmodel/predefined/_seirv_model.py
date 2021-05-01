from typing import List

from cmodel.model import CompartmentalModel


class ModelSEIRV(CompartmentalModel):

    def build_model(
        self, t, y,
        Lambda, mu, alpha, omega, gamma, xi_E, xi_I, sigma, beta_E, beta_I, beta_V, c1, c2, c3
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, E, I, R, V = y

        def beta(x, b, c):
            return b / (1. + c * x)

        principal_flux = S * (beta(E, beta_E, c1) * E + beta(I, beta_I, c2) * I + beta(V, beta_V, c3) * V)

        dydt = [
            Lambda - principal_flux - S * mu,
            principal_flux - E / alpha - mu * E,
            E / alpha - (omega + gamma + mu) * I,
            gamma * I - mu * R,
            xi_E * E + xi_I * I - sigma * V
        ]

        return dydt
