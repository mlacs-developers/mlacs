State sampling methods
######################

.. module:: mlacs.state

.. index::
   single: Class reference; StateManager

The State classes are object managing the state being simulated by the MLMD simulations, and hence, the state being approximated by MLACS using Machine Learning potential.
All the State classes are structured the same way with `run_dynamics` and `initialize_momenta` functions.

StateManager
************

.. autoclass:: StateManager
   :members: run_dynamics, initialize_momenta

Thermodynamic states
********************

These States are used to sample via MLMD simulation specific thermodyamic states/ensembles (NVT, NPT, ...).

LangevinState
~~~~~~~~~~~~~

.. autoclass:: LangevinState

LammpsState
~~~~~~~~~~~

.. autoclass:: LammpsState

PafiLammpsState
~~~~~~~~~~~~~~~

.. autoclass:: PafiLammpsState

Ground states
*************

These States are used to determine/relax atomic positions at 0K.

OptimizeLammpsState
~~~~~~~~~~~~~~~~~~~

.. autoclass:: OptimizeLammpsState

OptimizeAseState
~~~~~~~~~~~~~~~~

.. autoclass:: OptimizeAseState

Minimum Energy Path
*******************

These States are used to sample Minimum energy path.

BaseMepState
~~~~~~~~~~~~

Main class for ASE method.

.. autoclass:: mlacs.state.mep_ase_state.BaseMepState

LinearInterpolation
~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearInterpolation

NebAseState
~~~~~~~~~~~

.. autoclass:: NebAseState

CiNebAseState
~~~~~~~~~~~~~

.. autoclass:: CiNebAseState

StringMethodAseState
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StringMethodAseState

NebLammpsState
~~~~~~~~~~~~~~

.. autoclass:: NebLammpsState
