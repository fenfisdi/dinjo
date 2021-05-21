.. _start-examples:

========
Examples
========

Harmonic Oscillator
===================

Lets take the unit mass harmonic oscillator example to show you how to use our
library.

The initial value problem can be stated in terms of the Hamilton equations as

.. math::

   \frac{dq}{dt} =& \,\, p

   \frac{dp}{dt} =& - \omega ^ 2  q,


and an initial condition :math:`q(t_0) = q_0`,  :math:`p(t_0) = p_0`, where
:math:`q` represents the position of the particle and :math:`p` the momentum.

**The problem we will solve:** find the value of :math:`\omega` that best fits
to the IVP, given an experimental –noisy– solution.

**Note** that in this case we could find the optimal value of :math:`\omega` by
directly fitting the noisy data to the parametrized solution of the harmonic
oscillator because it is a well known solution. However, in most other problems,
the exact parametrized solution is not known and thus the problem could not be
solved that way. In that sense, this is a pedagogical example.

The step by step process is given as follows:

1. Define your IVP using the class :py:class:`dinjo.model.ModelIVP`.

      .. code-block:: python

         from dinjo.model import ModelIVP

         # Define the IVP
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

2. Note that the method  ``ModelOscillator.build_model`` implicitely uses the
   state variables ``p`` and ``q`` and the parameter ``w``. So, the
   next step is defining our State Variables and Parameters as dinjo objects:

      .. code-block:: python

         from dinjo.model import StateVariable, Parameter
         import numpy as np
         

         # Define State Variables
         q = StateVariable(
             name='position', representation='q', initial_value=1.0
         )
         p = StateVariable(
             name='momentum', representation='p', initial_value=0.0
         )

         # Define Paramters
         omega = Parameter(
             name='frequency', representation='w',
             initial_value=2 * np.pi, bounds=[4, 8]
         )

3. The State Variables and parameters must be encapsulated into lists in the
   same order implicetely defined in ``ModelOscillator.build_model``: the
   Parameters must be in the same order as the method's signature and the State
   Variables must be in the same order in which they are unpacked from the ``y``
   parameter. In this case:

      .. code-block:: python

         state_vars = [q, p]
         params = [omega]

4. Now, we instantiate the ``ModelOscillator`` class which will contain all the
   information of the oscillator IVP. The initial values are implicitely
   assumed at time ``t0 = t_span[0]``. So, the initial values are
   ``p(t_span[0]) = p.initial_value`` and ``q(t_span[0]) = q.initial_value``.
   In this case :math:`q(0) = 1` and :math:`p(0) = 0`.

      .. code-block:: python

         t_span = [0, 1]
         t_steps = 50

         # Instantiate the IVP class with appropiate State Variables and Parameters
         oscillator_model = ModelOscillator(
             state_variables=state_vars,
             parameters=params,
             t_span=t_span,
             t_steps=t_steps
         )

5. At this point you can play with the IVP itself. For example, you can
   integrate the equations using the method :py:meth:`~dinjo.model.run_model`.
   The resulting object is the same as the return value of
   scipy.integrate.solve_ivp_

      .. code-block:: python

         # Run the model
         oscillator_solution = oscillator_model.run_model()

6. Now we will build our :py:class:`dinjo.optimizer.Optimizer` instance.
   Ideally you may have some experimental (reference) data to use here. But as
   we do not, lets just generate some noisy data from the previous exact
   solution and use it to mock the experimental data. At this point we need to
   tell the optimizer what state variable corresponds to the observation data
   and also the times associated to the reference values.

      .. code-block:: python

         from dinjo.optimizer import Optimizer

         fake_data_noise_factor = 0.3

         # Build fake observation data from the solution
         oscillator_fake_position_data = (
             oscillator_solution.y[0]
             + (2 * np.random.random(t_steps) - 1) * fake_data_noise_factor
         )

         # Instantiate Optimizer using your data
         oscillator_optimizer = Optimizer(
             model=oscillator_model,
             reference_state_variable=q,
             reference_values=oscillator_fake_position_data,
             reference_t_values=oscillator_solution.t
         )

7. Finally we can find the value of :math:`\omega` that best fits to the
   solution of the IVP by using the
   :py:meth:`dinjo.optimizer.Optimizer.optimize` method.

      .. code-block:: python

         minimization_algorithm = 'differential_evolution'

         # Optimize parameters
         oscillator_parameters_optimization = \
             oscillator_optimizer.optimize(algorithm=minimization_algorithm)

         # The attribute oscillator_parameters_optimization.x contains the
         # optimal parameters
         print(f'Optimal value of $\omega$ = {oscillator_parameters_optimization.x[0]}')

8. That's it, but just for fun, lets plot the optimal solution

      .. code-block:: python

         import matplotlib.pyplot as plt

         oscillator_optimal_solution = oscillator_model.run_model(
             parameters=oscillator_parameters_optimization.x
         )

         plt.figure()
         plt.plot(
             oscillator_solution.t, oscillator_solution.y[0],
             'k-',
             label='Exact Solution using '
                   f'$\omega={omega.initial_value:.3f}$',
         )
         plt.plot(
             oscillator_solution.t, oscillator_fake_position_data,
             'ro', label='Noisy fake data'
         )
         plt.plot(
             oscillator_optimal_solution.t, oscillator_optimal_solution.y[0],
             'k-*',
             label='Optimized solution from noisy data\n'
                   f'using {minimization_algorithm} algorithm\n'
                   f'$\omega={oscillator_parameters_optimization.x[0]:.3f}$',
         )
         plt.xlabel('t')
         plt.ylabel('q(t)')
         plt.legend()
         plt.grid()
         plt.tight_layout()
         plt.show()
         plt.close()

   The plot you get should be similar to the following

      .. image:: ../_static/ho_example.png



The example is complete, should look like the following and should run as it is
given that you have previously installed ``dinjo``, ``numpy`` and ``matplotlib``

.. code-block:: python

   from dinjo.model import ModelIVP, StateVariable, Parameter
   from dinjo.optimizer import Optimizer

   import numpy as np
   import matplotlib.pyplot as plt


   # Define the IVP
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


   # Define State Variables
   q = StateVariable(
       name='position', representation='q', initial_value=1.0
   )
   p = StateVariable(
       name='momentum', representation='p', initial_value=0.0
   )

   # Define Paramters
   omega = Parameter(
       name='frequency', representation='w',
       initial_value=2 * np.pi, bounds=[4, 8]
   )

   state_vars = [q, p]
   params = [omega]

   t_span = [0, 1]
   t_steps = 50

   # Instantiate the IVP class with appropiate State Variables and Parameters
   oscillator_model = ModelOscillator(
       state_variables=state_vars,
       parameters=params,
       t_span=t_span,
       t_steps=t_steps
   )

   # Run the model
   oscillator_solution = oscillator_model.run_model()

   fake_data_noise_factor = 0.3

   # Build fake observation data from the solution
   oscillator_fake_position_data = (
       oscillator_solution.y[0]
       + (2 * np.random.random(t_steps) - 1) * fake_data_noise_factor
   )

   # Instantiate Optimizer using your data
   oscillator_optimizer = Optimizer(
       model=oscillator_model,
       reference_state_variable=q,
       reference_values=oscillator_fake_position_data,
       reference_t_values=oscillator_solution.t
   )

   minimization_algorithm = 'differential_evolution'

   # Optimize parameters
   oscillator_parameters_optimization = \
       oscillator_optimizer.optimize(algorithm=minimization_algorithm)

   # The attribute oscillator_parameters_optimization.x contains the
   # optimal parameters
   print(f'Optimal value of $\omega$ = {oscillator_parameters_optimization.x[0]}')

   # Plot solution
   oscillator_optimal_solution = oscillator_model.run_model(
       parameters=oscillator_parameters_optimization.x
   )

   plt.figure()
   plt.plot(
       oscillator_solution.t, oscillator_solution.y[0],
       'k-',
       label='Exact Solution using '
             f'$\omega={omega.initial_value:.3f}$',
   )
   plt.plot(
       oscillator_solution.t, oscillator_fake_position_data,
       'ro', label='Noisy fake data'
   )
   plt.plot(
       oscillator_optimal_solution.t, oscillator_optimal_solution.y[0],
       'k-*',
       label='Optimized solution from noisy data\n'
             f'using {minimization_algorithm} algorithm\n'
             f'$\omega={oscillator_parameters_optimization.x[0]:.3f}$',
   )
   plt.xlabel('t')
   plt.ylabel('q(t)')
   plt.legend()
   plt.grid()
   plt.tight_layout()
   plt.show()
   plt.close()

.. _scipy.integrate.solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html