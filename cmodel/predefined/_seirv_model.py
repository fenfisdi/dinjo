from typing import List

from cmodel.model import CompartmentalModel


class ModelSEIRV(CompartmentalModel):

    def build_model(
        self, t, y,
        Lambda, mu, alpha, omega, gamma, xi_E, xi_I, sigma, b1, b2, b3, c1, c2, c3
    ) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, Ex, If, R, V = y

        def beta(x, b, c):
            return b / (1. + c * x)

        principal_flux = S * (beta(Ex, b1, c1) * Ex + beta(If, b2, c2) * If + beta(V, b3, c3) * V)

        dydt = [
            Lambda - principal_flux - S * mu,
            principal_flux - (alpha + mu) * Ex,
            alpha * Ex - (omega + gamma + mu) * If,
            gamma * If - mu * R,
            xi_E * Ex + xi_I * If - sigma * V
        ]

        return dydt
