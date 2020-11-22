import sys
import os
from time import time

import matplotlib.pyplot as plt

# Add project root directory to path
this_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(this_file_dir, '..')
sys.path.append(project_root_dir)

from cmodel import seirv_model as seirv


# Instantiation of state variables and parameters

# In the implementation in which the model is built from scratch
# StateVariable and Parameter instances initialization might be a bit
# different, but the idea is the same

# The values of the parameters were obtained from Boris' Mathematica script
state_variables = [
    ("suceptible", "S", 50372424),
    ("exposed", "Ex", 0),
    ("infected", "If", 1),
    ("recovered", "R", 0),
    ("vaccinated", "V", 0),
]
parameters = [
    ("Lambda", "Lambda", 2083.62),
    ("mu", "mu", 0.0000150767),
    ("alpha", "alpha", 0.172414),
    ("omega", "omega", 0.0114),
    ("gamma", "gamma", 0.0666667),
    ("xi1", "xi1", 1.09736),
    ("xi2", "xi2", 6.32609),
    ("sigma", "sigma", 1.),
    ("b1", "b1", 2.91823E-9),
    ("b2", "b2", 2.67472E-9),
    ("b3", "b3", 4.635E-9),
    ("c1", "c1", 0.0000855641),
    ("c2", "c2", 2.94906E-7),
    ("c3", "c3", 0.159107),
]
state_variables = [seirv.StateVariable(*sv) for sv in state_variables]
parameters = [seirv.Parameter(*param) for param in parameters]

# Time span and time steps in days
t_span = [0, 171]
t_steps = 172

# Instantiate the model 
model_SEIRV = seirv.ModelSEIRV(state_variables, parameters, t_span, t_steps)

# In the next iteration of cmodel's API the user will need to define the
# model here manually by specifying the fluxes of each state variable using
# some interface such as model_SEIRV.add_flux(...)

# Run the model
t0 = time()
solution = model_SEIRV.run_model(method='RK45')
print(f"Time calculating solution: {time() - t0}")

# Data: Colombia
col_obs = [
    1, 1, 1, 1, 3, 9, 9, 13, 22, 34, 54, 65, 93, 102, 128, 196, 231, 277,
    378, 470, 491, 539, 608, 702, 798, 906, 1065, 1161, 1267, 1406, 1485,
    1579, 1780, 2054, 2223, 2473, 2709, 2776, 2852, 2979, 3105, 3233,
    3439, 3439, 3792, 3977, 4149, 4356, 4561, 4881, 5142, 5379, 5597,
    5949, 6207, 6507, 7006, 7285, 7668, 7973, 8613, 8959, 9456, 10051,
    10495, 11063, 11613, 12272, 12930, 13610, 14216, 14939, 15574, 16295,
    16935, 17687, 18330, 19131, 20177, 21175, 21981, 23003, 24104, 24141,
    25406, 26734, 27219, 29384, 30593, 31935, 33466, 36759, 36759, 38149,
    40847, 40847, 42206, 43810, 45344, 46994, 48896, 53211, 53211, 55083,
    57202, 60387, 63454, 68836, 71367, 73760, 73760, 77313, 80811, 84660,
    91995, 91995, 95269, 98090, 102261, 106392, 109793, 113685, 117412,
    120281, 124494, 128638, 133973, 140776, 145362, 150445, 154277,
    159898, 165169, 182140, 190700, 197278, 204005, 211038, 218428,
    226373, 233541, 240795, 240795, 257101, 267385, 276055, 286020,
    295508, 306181, 317651, 327850, 334979, 345714, 357710, 367204,
    376870, 387481, 397623, 410453, 422519, 433805, 445111, 456689,
    468332, 476660, 489122, 502178, 513719, 522138, 522138, 541139, 551688
]

plt.figure()
plt.plot(solution.t, solution.y[2], "ko-", label='SEIRV model')
plt.plot(solution.t, col_obs, "ro-", label='Data from INS')
plt.xlabel('Days since case 1')
plt.ylabel('Confirmed infected people')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.close()