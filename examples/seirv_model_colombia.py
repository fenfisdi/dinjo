import sys
import os
from time import time
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt

# Add project root and this file's directories to path in order to find cmodel
# package
this_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(this_file_dir, '..')
sys.path.append(this_file_dir)
sys.path.append(project_root_dir)

from cmodel import model, predefined
from cmodel_examples_utilities import param_sample_range


# Instantiation of state variables and parameters

# In the next implementation (in which the model is built from scratch)
# StateVariable and Parameter instances initialization ought be a bit
# different, but the idea is the same

# The values of the parameters were obtained from Boris' Mathematica script
# (name, representation, initial_value)
seirv_state_variables_colombia_src = [
    ("suceptible", "S", 50372424),
    ("exposed", "Ex", 0),
    ("infected", "If", 1),
    ("recovered", "R", 0),
    ("vaccinated", "V", 0),
]

# List of instances of state variables
seirv_state_variables_colombia = [
    model.StateVariable(*sv) for sv in seirv_state_variables_colombia_src
]

# Parameters not variated by Boris' Mathematica script
Lambda = 2083.62
mu = 0.0000150767
alpha = 0.172414
omega = 0.0114
gamma = 0.0666667
xi2 = 6.32609
sigma = 1.0

# Optimal parameters found by Boris' Mathematica script
xi1_opt = 1.09736
b1_opt = 2.91823E-9
b2_opt = 2.67472E-9
b3_opt = 4.635E-9
c1_opt = 0.0000855641
c2_opt = 2.94906E-7
c3_opt = 0.159107

# Parameters used to define their own bounds via the function
# param_sample_range
xi1 = 2.3
b1, b2, b3 = 1.5 * 3.11e-9, 1.5 * 0.62e-9, 1.5 * 1.03e-9

# Bounds of c and xi2 parameters used in Boris' Mathematica script
c_bounds = [0, 1]
xi2_bounds = [0, 10]

# param_range_width used in Boris' Mathematica script
param_range_width = 3

# Each element of the list refers to a parameter
# (name, representation, initial_value, bounds)
seirv_parameters_colombia_src = [
    ("Lambda", "Lambda", Lambda, [Lambda, Lambda]),
    ("mu", "mu", mu, [mu, mu]),
    ("alpha", "alpha", alpha, [alpha, alpha]),
    ("omega", "omega", omega, [omega, omega]),
    ("gamma", "gamma", gamma, [gamma, gamma]),
    ("xi1", "xi1", xi1_opt, param_sample_range(xi1, param_range_width)),
    ("xi2", "xi2", xi2, xi2_bounds),
    ("sigma", "sigma", sigma, [sigma, sigma]),
    ("b1", "b1", b1_opt, param_sample_range(b1, param_range_width)),
    ("b2", "b2", b2_opt, param_sample_range(b2, param_range_width)),
    ("b3", "b3", b3_opt, param_sample_range(b3, param_range_width)),
    ("c1", "c1", c1_opt, c_bounds),
    ("c2", "c2", c2_opt, c_bounds),
    ("c3", "c3", c3_opt, c_bounds),
]
seirv_parameters_colombia_src = [
    {
        'name': param_info[0],
        'representation': param_info[1],
        'initial_value': param_info[2],
        'bounds': param_info[3]
    }
    for param_info in seirv_parameters_colombia_src
]

# List of instances of parameters
seirv_parameters_colombia = []
for param_info in seirv_parameters_colombia_src:
    seirv_parameters_colombia.append(
        model.Parameter(**param_info)
    )

# Time span and time steps in days
t_span_col = [0, 171]
t_steps_col = 172

infected_reference_col = [
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
    468332, 476660, 489122, 502178, 513719, 522138, 522138, 541139, 551688,
]


def seirv_model_example(
    state_variables: List[model.StateVariable] = seirv_state_variables_colombia,
    parameters: List[model.Parameter] = seirv_parameters_colombia,
    t_span: List[Union[float, int]] = t_span_col,
    t_steps: int = t_steps_col,
    infected_reference: List[int] = infected_reference_col,
    monitor_computation_time: bool = False,
    plot_solution: bool = False,
) -> Dict[str, Any]:

    # Instantiate the model
    model_SEIRV = predefined.ModelSEIRV(
        state_variables=state_variables,
        parameters=parameters,
        t_span=t_span,
        t_steps=t_steps
    )

    # In the next iteration of cmodel's API the user will need to define the
    # model here manually by specifying the fluxes of each state variable using
    # some interface such as model_SEIRV.add_flux(...)

    # Run the model
    t0 = time()
    solution = model_SEIRV.run_model(method='RK45')
    if monitor_computation_time:
        print(f"Time calculating solution: {time() - t0}")

    if plot_solution:
        plt.figure()
        plt.plot(solution.t, solution.y[2], "ko-", label='SEIRV model')
        plt.plot(solution.t, infected_reference, "ro-", label='Data from INS')
        plt.xlabel('Days since case 1')
        plt.ylabel('Confirmed infected people')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    seirv_model_example = {
        'model': model_SEIRV,
        'solution': solution
    }

    return seirv_model_example


if __name__ == "__main__":
    seirv_model_example(
        monitor_computation_time=True,
        plot_solution=True
    )
