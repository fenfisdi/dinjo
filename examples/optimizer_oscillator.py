import os
from os import stat
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add project root directory to path
this_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(this_file_dir, '..')
sys.path.append(project_root_dir)

from cmodel import  seirv_model as seirv
from cmodel import optimizer


class Oscillator(seirv.CompartmentalModel):
    def build_model(self, t, y, m, k):
        """Harmonic Oscillator differential equations
        """
        q, p  = y

        # Hamilton's equations
        dydt = [
            p / m,
            - k * q
        ]

        return dydt


if __name__ == '__main__':

    # Define State Variables
    q = seirv.StateVariable(
        name='position', representation='q', initial_value=1.0
    )
    p = seirv.StateVariable(
        name='momentum', representation='p', initial_value=0.0
    )

    # Define Paramters
    m = seirv.Parameter(
        name='mass', representation='m', initial_value=6.0, bounds=[5, 7]
    )
    k = seirv.Parameter(
        name='force constant', representation='k', initial_value=.5, bounds=[0., 1]
    )

    t_span = [0, 2 * np.pi * (m.initial_value / k.initial_value)**0.5]
    t_steps = 50

    # Instantiate Model
    oscillator_model = Oscillator(
        state_variables=[q, p],
        parameters=[m, k],
        t_span=t_span,
        t_steps=t_steps
    )

    # Define Model fluxes (This Will be done in the next iteration)

    # Run the model
    oscillator_solution = oscillator_model.run_model()

    # Build fake observation data from the solution (to test optimizer)
    noise_factor = 0.30
    oscillator_fake_position_data = \
        oscillator_solution.y[0] + (2 * np.random.random(t_steps) - 1) * noise_factor


    # Optimizer using fake noisy data
    oscillator_optimizer = optimizer.Optimizer(
        model=oscillator_model,
        reference_state_variable=q,
        reference_values=oscillator_fake_position_data,
        reference_t_values=oscillator_solution.t
    )

    # Optimize parameters
    parameters_optimization = \
        oscillator_optimizer.minimize_global(algorithm='differential evolution')

    # Calculate differential equation solution using optimized parameters
    if parameters_optimization.success:
        oscillator_optimal_solution = \
            oscillator_model.run_model(parameters=parameters_optimization.x)
    else:
        oscillator_optimal_solution = None
        print("Parameter optimization did not succed.")


    # Plot solution
    plt.figure()
    plt.plot(
        oscillator_solution.t, oscillator_solution.y[0],
        'k-', label='Exact Solution'
    )
    plt.plot(
        oscillator_solution.t, oscillator_fake_position_data,
        'ro', label='Noisy fake data'
    )
    try:
        plt.plot(
            oscillator_optimal_solution.t, oscillator_optimal_solution.y[0],
            'k-*',
            label='Optimized solution from noisy data\n'
                + f'$m={parameters_optimization.x[0]:.3f}$,  '
                + f'$k={parameters_optimization.x[1]:.3f}$   '
                + f'$m/k={parameters_optimization.x[0]/parameters_optimization.x[1]:.3f}$   '
        )
    except:
        pass
    plt.xlabel('t')
    plt.ylabel('q(t)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()
