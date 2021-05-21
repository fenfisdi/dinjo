import os
from typing import Any, Dict, List, Union
import pickle
from datetime import datetime, timedelta


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dinjo import model, optimizer


this_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(this_file_dir, '..')


class ModelOscillator(model.ModelIVP):
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


def oscillator_optimizer_example(
    q0: Union[float, int] = 1.0,
    p0: Union[float, int] = 0.0,
    w0: Union[float, int] = 2 * np.pi,
    w_bounds: List[Union[float, int]] = [4, 8],
    t_span: List[Union[float, int]] = [0, 1],
    t_steps: int = 50,
    fake_data_noise_factor: float = 0.30,
    minimization_algorithm: str = 'differential_evolution',
    save_optimization_results_pickle: bool = False,
    plot_solution: bool = False,
    save_csv_file: bool = False,
    date_fmt: bool = False
) -> Dict[str, Any]:

    # Define State Variables
    q = model.StateVariable(
        name='position', representation='q', initial_value=q0
    )
    p = model.StateVariable(
        name='momentum', representation='p', initial_value=p0
    )

    # Define Paramters
    omega = model.Parameter(
        name='frequency', representation='w', initial_value=w0, bounds=w_bounds
    )

    # Instantiate Model
    oscillator_model = ModelOscillator(
        state_variables=[q, p],
        parameters=[omega],
        t_span=t_span,
        t_steps=t_steps
    )

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
        oscillator_optimizer.optimize(algorithm=minimization_algorithm)

    date_equivalent = [
        datetime.now() + timedelta(days=i) for i in range(len(oscillator_solution.t))
    ]

    # Save pickle
    if save_optimization_results_pickle or save_csv_file:
        files_directory_path = os.path.join(
            this_file_dir,
            'generated_files'
        )

        if not os.path.isdir(files_directory_path):
            try:
                os.mkdir(files_directory_path)
            except OSError:
                print("Creation of the directory %s failed" % files_directory_path)
            else:
                print("Successfully created the directory %s " % files_directory_path)

                if save_optimization_results_pickle:
                    pickle_path = os.path.join(
                        files_directory_path,
                        'oscillator_parameters_optimization.pickle'
                    )
                    pickle.dump(
                        oscillator_parameters_optimization,
                        open(pickle_path, 'wb')
                    )

    # Calculate differential equation solution using optimized parameters
    if oscillator_parameters_optimization.success:
        oscillator_optimal_solution = oscillator_model.run_model(
            parameters=oscillator_parameters_optimization.x
        )
        if save_csv_file:
            csv_path = os.path.join(
                files_directory_path,
                'oscillator_csv_example.csv'
            )
            pd.DataFrame(
                {
                    'date': [
                        date.strftime('%Y-%m-%d') for date in date_equivalent
                    ],
                    'user_data_variable_X': oscillator_fake_position_data,
                    'optimal_solution_variable_X': oscillator_optimal_solution.y[0]
                }
            ).to_csv(csv_path, header=True, index=False)
    else:
        oscillator_optimal_solution = None
        print("Parameter optimization did not succed.")

    # Plot solution
    if plot_solution:
        oscillator_solution.t = date_equivalent if date_fmt else oscillator_solution.t
        oscillator_optimal_solution.t = date_equivalent if date_fmt else oscillator_solution.t
        xdate_val = True if date_fmt else False

        plt.figure()
        if date_fmt:
            plt.xticks(rotation=70)
        plt.plot_date(
            oscillator_solution.t, oscillator_solution.y[0],
            'k-',
            label='Exact Solution using '
                  f'$\omega={omega.initial_value:.3f}$',
            xdate=xdate_val
        )
        plt.plot_date(
            oscillator_solution.t, oscillator_fake_position_data,
            'ro', label='Noisy fake data', xdate=xdate_val
        )
        try:
            plt.plot_date(
                oscillator_optimal_solution.t, oscillator_optimal_solution.y[0],
                'k-*',
                label='Optimized solution from noisy data\n'
                      f'using {minimization_algorithm} algorithm\n'
                      f'$\omega={oscillator_parameters_optimization.x[0]:.3f}$',
                xdate=xdate_val
            )
        except Exception:
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
    oscillator_optimizer_example(
        plot_solution=True,
        save_optimization_results_pickle=True,
        save_csv_file=True,
        date_fmt=False
    )
