import numbers
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp


class Variable:
    """Represents a variable

    Attributes
    ----------
    name : str
        name of the variable.
    representation : str
        string representing the state variable (could be the same as
        name.)
    initial_value : float
        reference value of the variable being represented.
    """
    def __init__(
        self, name: str, representation: str, initial_value: float = 0,
        *args, **kwargs
    ) -> None:
        self.name = name
        self.representation = representation
        self.initial_value = initial_value


class StateVariable(Variable):
    """Represents a State Variable of an initial value problem."""
    pass


class Parameter(Variable):
    """Represents a parameter of the differential equations definining
    an initial value problem.

    In addition to the attributes defined in
    :py:class:`dinjo.model.Variable`

    Attributes
    ----------
    bounds : 2-tuple of floats.
        list containing the minimum and maximum values that the
        parameter can take ``(min, max)``.
    """
    def __init__(
        self, name: str, representation: str, initial_value: float = 0,
        bounds: Optional[List[float]] = None, *args, **kwargs
    ) -> None:
        super().__init__(name, representation, initial_value)
        self.bounds = bounds if bounds else [initial_value, initial_value]

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds_input):
        attr_err_message = "bounds must be a list of non decreasing numbers."
        init_val_not_in_bounds_range = "initial_value must be in the range defined by bounds."

        type_check: bool = (
            isinstance(bounds_input, list)
            and len(bounds_input) == 2
            and isinstance(bounds_input[0], numbers.Number)
            and isinstance(bounds_input[1], numbers.Number)
        )

        if not type_check:
            raise ValueError(attr_err_message)

        order_check: bool = bounds_input[0] <= bounds_input[1]

        if not order_check:
            raise ValueError(attr_err_message)

        if not (
            bounds_input[0] <= self.initial_value
            and bounds_input[1] >= self.initial_value
        ):
            raise ValueError(init_val_not_in_bounds_range)

        self._bounds = bounds_input


class ModelIVP:
    """Defines and integrates an initial value problem.

    Attributes
    ----------
    state_variables : list[:class:`StateVariable`]
    parameters : list[:class:`Parameter`]
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0
        and integrates until it reaches t=tf.
    t_steps : int
        The solver will get the solution for ``t_steps`` equally
        separated times from ``t0`` to ``tf``.
    t_span : list[float]
        List containing the time values in which the user wants to
        evaluate the solution. All the values must be within the
        interval defined by ``t_span``.
    """
    def __init__(
        self,
        state_variables: List[StateVariable],
        parameters: List[Parameter],
        t_span: Optional[List[float]] = None,
        t_steps: int = 50,
        t_eval: Optional[List[float]] = None
    ) -> None:
        self.state_variables = state_variables
        self.parameters = parameters
        self.t_span = t_span if t_span else [0, 10]
        self.t_steps = t_steps
        self.t_eval = t_eval if t_eval else list(np.linspace(*t_span, t_steps))

    def _get_variable_init_vals(
        self, variables: List[Variable]
    ) -> List[float]:
        return [var.initial_value for var in variables]

    @property
    def state_variables_init_vals(self) -> List[float]:
        """Get the values of the model's state variables initial values
        in the order they are currently stored in ``self.state_variables``.
        """
        return self._get_variable_init_vals(self.state_variables)

    @property
    def parameters_init_vals(self):
        """Get the values of the model's parameters initial values in
        theorder they are currently stored in ``self.parameters``.
        """
        return self._get_variable_init_vals(self.parameters)

    def build_model(self, t, y, *args):
        """Defines the differntial equations of the model.

        Override this method so that it contains the differential
        equations of your IVP. The signature of the method must be
        ``build_model(self, t, y, *args)`` where ``t`` is the time,
        ``y`` is the state vector, and ``args`` are other parameters of
        the system.

        Parameters
        ----------
        t : float
            time at which the differential equation must be evaluated.
        y : list[float]
            state vector at which the differential must be evaluated.
        *args : any
            other parameters of the differential equation

        Returns
        -------
        The method must return the time derivative of the
        state vector evaluated at a given time.

        Note
        ----
        The parameters must be defined in the same order in which the
        parameters are stored in :attr:`ModelIVP.parameters`.

        Note
        ----
        The state variable vector must be defined in the same order
        in which the state variables are stored in
        :attr:`ModelIVP.state_variables`.

        Example
        -------
        For example if you want to simulate a harmonic
        oscillator of frequency :math:`\omega` and mass :math:`m`
        this method must be implemented as follows:

        .. code-block:: python

            def build_model(self, t, y, w, m):
                q, p = y

                # Hamilton's equations
                dydt = [
                    p,                      # dq/dt
                    - (w ** 2) * q          # dp/dt
                ]

                return dydt
        """
        raise NotImplementedError

    def run_model(
        self,
        parameters: List[float] = None,
        method: str = 'RK45'
    ):
        """Integrate model using ``scipy.integrate.solve_ivp``

        Parameters
        ----------
        parameters : list[float]
            List of the values of the paramters of the initial value
            problem.
        method : srt
            Integration method. Must be one of the methods accepted
            by ``scipy.integrate.solve_ivp``

        Returns
        -------
        Bunch object with the following fields defined (same return type
        as scipy.integrate.solve_ivp):
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Values of the solution at t.
        sol : OdeSolution or None
            Found solution as OdeSolution instance; None if dense_output
            was set to False.
        t_events : list of ndarray or None
            Contains for each event type a list of arrays at which an
            event of that type event was detected. None if events was
            None.
        y_events : list of ndarray or None
            For each value of t_events, the corresponding value of the
            solution. None if events was None.
        nfev : int
            Number of evaluations of the right-hand side.
        njev : int
            Number of evaluations of the Jacobian.
        nlu : int
            Number of LU decompositions.
        status : int
            Reason for algorithm termination:
        """

        parameters = (
            parameters if parameters is not None else self.parameters_init_vals
        )

        parameters_permitted_types = (list, tuple, np.ndarray)
        parameters_type_is_permitted = False

        for permitted_type in parameters_permitted_types:
            parameters_type_is_permitted += isinstance(parameters, permitted_type)

        if not parameters_type_is_permitted:
            raise TypeError(
                "parameters must be a list, tuple or numpy.ndarray"
            )

        solution = solve_ivp(
            fun=self.build_model,
            t_span=self.t_span,
            y0=self.state_variables_init_vals,
            method=method,
            t_eval=self.t_eval,
            args=parameters
        )

        return solution
