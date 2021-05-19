.. DINJO documentation master file, created by
   sphinx-quickstart on Tue May 18 12:06:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _index:

===============
DINJO |version|
===============

*DINJO Is Not Just An Optimizer* is a Python framework designed for the optimization of initial value problems' parameters.

Lets say you have some 'experimental' data of a state variable :math:`S` of the initial value problem :math:`dS/dt = f(t, S; p), S(t_0) = f_0` where :math:`p` is a list of parameters.

If you want to know the optimal value of :math:`p` so that the solution of the initial value problem fits your experimental data, you can use DINJO to get an approximate value of the optimal :math:`p`.


Getting Started
===============

.. toctree::
   :maxdepth: 6
   :caption: Getting Started
   :hidden:

   start/overview
   start/examples

- :ref:`Overview <start-overview>`
- :ref:`Examples <start-examples>`

Install DINJO using PyPI:

.. code-block:: bash

   pip install dinjo

Or directly from the latest dev versioN, using source code:

.. code-block:: bash

   git clone https://github.com/fenfisdi/dinjo
   cd dinjo
   python setup.py install

Start using DINJO!


The Source Code
===============

.. toctree::
   :caption: THE SOURCE CODE
   :hidden:

   dinjo/dinjo

:mod:`dinjo`
   Package source code

   :mod:`dinjo.model`
      Define your own initial value problems (IVPs) and solve them using this module.

   :mod:`dinjo.optimizer`
      Define your IVP optimization problem and solve it using this module.

   :mod:`dinjo.predefined`
      Some predefined models

      :mod:`dinjo.predefined.epidemiology._seir_model`
         SEIR initial value problem.

      :mod:`dinjo.predefined.epidemiology._seirv_model`
         SEIRV initial value problem.
      
      :mod:`dinjo.predefined.epidemiology._seirv_fixed`
         SEIR initial value problem.
      
      :mod:`dinjo.predefined.epidemiology._sir_model`
         SIR initial value problem.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
