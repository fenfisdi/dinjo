from typing import List

from cmodel.model import CompartmentalModel

class ModelSIR(CompartmentalModel):
    def build_model(
        self, t, y,
        lmbd, gI, omega, gamma, gB, gP, gV, mu,
    ) -> List[float]:
        """Returns the vector field dy/dt evaluated at a given point in phase space"""

        S, If, R = y

        principal_flux = gI * S * If / (S + If + R)

        dydt = [
            lmbd + gP * R - gV * S - principal_flux - mu * S,
            principal_flux - gB * R - (gamma + omega + mu ) * I,
            gamma * If + gV * S - ( gB + gP ) * If
        ]

        return dydt
