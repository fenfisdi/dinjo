# Optimize the seirv model for colombia
# This script may take several minutes to execute
import os
import pickle
from time import time
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

import dinjo.optimizer as optimizer
from seirv_model_colombia import (
    seirv_state_variables_colombia, seirv_model_example, infected_reference_col
)
from cmodel_examples_utilities import int_to_str_date, setup_csv


this_file_dir = os.path.dirname(__file__)
project_root_dir = os.path.join(this_file_dir, '..')


def optimizer_seirv_model_colombia_example(
    minimization_algorithm: str = 'differential_evolution',
    algorithm_kwargs={
        'popsize': 20,
        'disp': True,
        'tol': 0.0015,
        'maxiter': 100,
        'mutation': [0.3, 0.7],
        'atol': 200
    },
    print_optimization_log: bool = False,
    save_optimization_results: bool = False,
    generated_files_directory_name: str = 'generated_files',
    file_base_name: str = 'seirv_model_colombia_parameters_optimization',
    plot_results: bool = False,
    save_optimal_solution: bool = False,
    csv_optimal_solution_path: str = ''
) -> Dict[str, Any]:

    seirv_model_example_col = seirv_model_example()
    optimizer_seirv_model_colombia = optimizer.Optimizer(
        model=seirv_model_example_col["model"],
        reference_state_variable=seirv_state_variables_colombia[2],
        reference_values=infected_reference_col,
        reference_t_values=seirv_model_example_col['solution'].t
    )

    # Optimization using the desired algorith
    t0 = time()
    seirv_colombia_parameters_optimization = \
        optimizer_seirv_model_colombia.optimize(
            cost_method='root_mean_square',
            algorithm=minimization_algorithm,
            algorithm_kwargs=algorithm_kwargs
        )
    computation_time = time() - t0

    line_split = '=============================================\n'

    # Construct log message
    optimization_log = (
        line_split
        + 'SEIRV Model Colombia Parematers Optimization:\n'
        + line_split
        + f'Computation time: {computation_time}\n'
        + line_split
        + str(seirv_colombia_parameters_optimization)
        + '\n\nParam \tOptimized \tExpected\n'
        + line_split
    )

    optimal_solution_mathematica = \
        optimizer_seirv_model_colombia.model.parameters_init_vals
    parameters = optimizer_seirv_model_colombia.model.parameters

    for i, param in enumerate(seirv_colombia_parameters_optimization.x):
        optimization_log += (
            f"{parameters[i].representation}\t"
            f"{param:.4e}\t"
            f"{optimal_solution_mathematica[i]:.4e}\n"
        )

    if print_optimization_log:
        print(optimization_log)

    if save_optimization_results:
        # Directory name (full path). Created in the same folder as this file.
        generated_files_directory_path = os.path.join(
            this_file_dir,
            generated_files_directory_name
        )
        # File base name for pickle and txt (full path).
        file_base_path = os.path.join(
            generated_files_directory_path,
            file_base_name
        )

        # If directory does not exist, create it
        if not os.path.isdir(generated_files_directory_path):
            try:
                os.mkdir(generated_files_directory_path)
            except OSError:
                print(
                    "Creation of the directory "
                    f"{generated_files_directory_path} failed"
                )
            else:
                print(
                    "Successfully created the directory "
                    f"{generated_files_directory_path}"
                )

        csv_file_path = file_base_path + '.csv'

        # Create CSV file with proper column names if one does not exist
        param_names = [
            param.name for param in optimizer_seirv_model_colombia.model.parameters
        ]
        csv_column_names = [
            'index',
            'comp_time', 'algorithm',
            'success', 'optimizer_message', 'fun', 'nfev', 'nit',
            *param_names
        ]
        setup_csv(csv_file_path, first_row=csv_column_names)

        # Organize optimal parameters in dictionary
        optimal_parameters_dict = {}
        for i, param_opt_value in enumerate(seirv_colombia_parameters_optimization.x):
            optimal_parameters_dict[param_names[i]] = param_opt_value

        # Add optimal parameters to the CSV file using pandas
        try:
            df_optimization_log = pd.read_csv(
                csv_file_path, index_col=csv_column_names[0]
            )
        except (EmptyDataError, ValueError):
            df_optimization_log = pd.DataFrame({csv_column_names[1]: [0.]})

        optimization_csv_row = {
            csv_column_names[1]: computation_time,
            csv_column_names[2]: minimization_algorithm,
            csv_column_names[3]: seirv_colombia_parameters_optimization.success,
            csv_column_names[4]: seirv_colombia_parameters_optimization.message,
            csv_column_names[5]: seirv_colombia_parameters_optimization.fun,
            csv_column_names[6]: seirv_colombia_parameters_optimization.nfev,
            csv_column_names[7]: seirv_colombia_parameters_optimization.nit,
            **optimal_parameters_dict
        }
        df_optimization_log = df_optimization_log.append(
            optimization_csv_row, ignore_index=True
        )
        df_optimization_log.to_csv(csv_file_path, index_label=csv_column_names[0])

        # Save pickle
        pickle.dump(
            seirv_colombia_parameters_optimization,
            open(file_base_path + '.pickle', 'wb')
        )

    if plot_results or save_optimal_solution:
        optimal_solution = optimizer_seirv_model_colombia.model.run_model(
            parameters=seirv_colombia_parameters_optimization.x
        )

    if plot_results:

        plt.figure()
        plt.plot(
            optimal_solution.t, optimal_solution.y[2],
            "ko-", label='Optimization of SEIRV model'
        )
        plt.plot(
            optimal_solution.t, infected_reference_col,
            "ro-", label='Data from INS'
        )
        plt.xlabel('Days since case 1')
        plt.ylabel('Confirmed infected people')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    optimizer_seirv_model_colombia_example = {
        'optimizer': optimizer_seirv_model_colombia,
        'optimization_results': seirv_colombia_parameters_optimization,
        'optimization_log': optimization_log,
        'computation_time': computation_time,
    }

    if save_optimal_solution:
        t_dates = []
        initial_date = datetime.utcnow()

        for i in range(len(optimal_solution.t)):
            t_dates.append(int_to_str_date(i, initial_date))

        df = pd.DataFrame(
            {
                'date': t_dates,
                'I_reference': infected_reference_col,
                'S_optimal': optimal_solution.y[0],
                'E_optimal': optimal_solution.y[1],
                'I_optimal': optimal_solution.y[2],
                'R_optimal': optimal_solution.y[3],
                'V_optimal': optimal_solution.y[4],
            }
        )

        df.set_index('date', inplace=True)

        df.to_csv(csv_optimal_solution_path)

    return optimizer_seirv_model_colombia_example


if __name__ == "__main__":
    csv_optimal_solution_path = os.path.join(
        this_file_dir,
        'generated_files',
        'seirv_model_colombia_optimal_solution.csv'
    )

    optimizer_seirv_model_colombia_example(
        minimization_algorithm='differential_evolution',
        print_optimization_log=True,
        save_optimization_results=True,
        plot_results=True,
        save_optimal_solution=False,
        csv_optimal_solution_path=csv_optimal_solution_path
    )
