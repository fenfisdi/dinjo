from ...model import ModelIVP


class ModelOscillator(ModelIVP):
    def build_model(self, t, y, w):
        """Harmonic Oscillator differential equations
        """
        q, p = y

        # Hamilton's equations
        dydt = [
            p,
            - (w ** 2) * q
        ]

        return dydt
