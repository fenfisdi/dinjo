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


class ModelOscillator(seirv.CompartmentalModel):
    def build_model(self, t, y, w):
        """Harmonic Oscillator differential equations
        """
        q, p  = y

        # Hamilton's equations
        dydt = [
            p,
            - (w ** 2) * q
        ]

        return dydt


def oscillator_optimizer_example(
    q0=1.0,
    p0=0.0,
    w0=2 * np.pi,
    w_bounds=[4, 8],
    t_span=[0, 1],
    t_steps=50,
    fake_data_noise_factor=0.30,
    minimization_algorithm='differential_evolution',
    plot_solution=False
):

    # Define State Variables
    q = seirv.StateVariable(
        name='position', representation='q', initial_value=q0
    )
    p = seirv.StateVariable(
        name='momentum', representation='p', initial_value=p0
    )

    # Define Paramters
    omega = seirv.Parameter(
        name='frequency', representation='w', initial_value=w0, bounds=w_bounds
    )

    # Instantiate Model
    oscillator_model = ModelOscillator(
        state_variables=[q, p],
        parameters=[omega],
        t_span=t_span,
        t_steps=t_steps
    )

    # Define Model fluxes (This Will be done in the next iteration)

    # Run the model
    oscillator_solution = oscillator_model.run_model()

    # Build fake observation data from the solution (to test optimizer)
    oscillator_fake_position_data = (
        oscillator_solution.y[0] 
        + (2 * np.random.random(t_steps) - 1) * fake_data_noise_factor
        )

    # Optimizer using fake noisy data
    oscillator_optimizer = optimizer.Optimizer(
        model=oscillator_model,
        reference_state_variable=q,
        reference_values=oscillator_fake_position_data,
        reference_t_values=oscillator_solution.t
    )

    # Optimize parameters
    oscillator_parameters_optimization = \
        oscillator_optimizer.minimize_global(algorithm=minimization_algorithm)

    # Calculate differential equation solution using optimized parameters
    if oscillator_parameters_optimization.success:
        oscillator_optimal_solution = oscillator_model.run_model(
            parameters=oscillator_parameters_optimization.x
        )
    else:
        oscillator_optimal_solution = None
        print("Parameter optimization did not succed.")

    # Plot solution
    if plot_solution:        
        plt.figure()
        plt.plot(
            oscillator_solution.t, oscillator_solution.y[0],
            'k-',
            label='Exact Solution using '
                f'$\omega={omega.initial_value:.3f}$'
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
                    f'using {minimization_algorithm} algorithm\n'
                    f'$\omega={oscillator_parameters_optimization.x[0]:.3f}$'
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

    oscillator_optimization_example = {
        'model': oscillator_model,
        'solution': oscillator_solution,
        'fake_position_data': oscillator_fake_position_data,
        'optimizer': oscillator_optimizer,
        'parameters_optimization': oscillator_parameters_optimization,
        'optimal_solution': oscillator_optimal_solution,
    }

    return oscillator_optimization_example


if __name__ == '__main__':
    oscillator_optimizer_example(plot_solution=True)