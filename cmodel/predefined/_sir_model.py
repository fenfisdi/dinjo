from typing import List

from cmodel.model import CompartmentalModel

class SIR_model(CompartmentalModel):
    
    def build_model( self, t, y,
                    Lmbd, mu, omega, gamma, chi, eta, Pi, tau) -> List[float]:
        """
        Returns the vector field dy/dt evaluated at a given point in phase space
        """
        S, I, R = y
    
        dS = Lmbd + Pi*R - tau*S*I - (chi+mu)*S
        dI = tau*S*I - eta*R - (gamma+omega+mu)*I
        dR = gamma*I + chi*S - (eta+Pi+mu)*R
    
        return [dS, dI, dR]
