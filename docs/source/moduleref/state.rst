State sampling methods
######################

.. module:: mlacs.state

.. index::
   single: Class reference; StateManager

The State Classes are object managing the state being simulated by the MLMD simulations, and hence, the state being approximated by MLACS.

StateManager
************

.. autoclass:: mlacs.state.state.StateManager

Thermodynamic states
********************

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

OptimizeLammpsState
~~~~~~~~~~~~~~~~~~~

.. autoclass:: NebLammpsState

NebLammpsState
~~~~~~~~~~~~~~

.. autoclass:: NebLammpsState
