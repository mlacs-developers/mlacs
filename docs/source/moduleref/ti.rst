Thermodynamic Integration
#########################

.. module:: mlacs.ti

Modules to handle thermodynamic integration

ThermodynamicIntegration
************************

.. autoclass:: ThermodynamicIntegration
   :members: run

.. module:: mlacs.ti.thermostate

ThermoState
~~~~~~~~~~~

.. autoclass:: ThermoState


.. module:: mlacs.ti.solids

EinsteinSolidState
~~~~~~~~~~~~~~~~~~

.. autoclass:: EinsteinSolidState


.. module:: mlacs.ti.liquids

UFLiquidState
~~~~~~~~~~~~~

.. autoclass:: UFLiquidState


.. module:: mlacs.ti.reversible_scaling

ReversibleScalingState
**********************

.. autoclass:: ReversibleScalingState


.. module:: mlacs.ti.gpthermoint

Gaussian Process
################

GpThermoIntT
************

.. autoclass:: GpThermoIntT
   :members: add_new_data, get_helmholtz_free_energy

GpThermoIntVT
*************

.. autoclass:: GpThermoIntVT
   :members: add_new_data, get_helmholtz_free_energy, get_gibbs_free_energy, get_volume_from_press_temp, get_thermal_expansion
