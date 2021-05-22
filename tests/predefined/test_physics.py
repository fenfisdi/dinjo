from pytest import approx
from numpy import pi

from dinjo.model import StateVariable, Parameter
from dinjo.predefined.physics import ModelOscillator


def test_model_oscillator():
    q = StateVariable('position', 'q', 1.0)
    p = StateVariable('momentum', 'p', 0.0)
    omega = Parameter('frequency', 'w', 2 * pi, [4, 8])
    oscillator_model = ModelOscillator([q, p], [omega], [0, 1], 50)
    dydt = oscillator_model.build_model(0, [1/2, 1/2], 1/2)            # noqa
    assert dydt == approx([1/2, - 1/8], 0.99)                           # noqa
