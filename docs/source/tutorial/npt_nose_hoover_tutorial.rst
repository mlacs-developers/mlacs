.. _npt_nose_hoover_tutorial:


Running a NPT sampling using a Nosé-Hoover thermostat and barostat.
===================================================================

In this tutorial, we will see how to setup a NPT simulation using a Nosé-Hoover thermostat and barostat.
The system is an Al FCC crystal at 300 K and 10 GPa of pressure.
For simplicity and rapidity, the potential will be the effective medium theory calculator.


Setting the simulation
----------------------

We start by importing the packages. In this case, we are using the LammpsState to use the LAMMPS implementation of the thermostat and barostat. 

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 1-9

Parameters for MLACS simulation:

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 18-23

The LammpsState is an interface beetween LAMMPS and the MLACS package. In this case, we wanted to use a Nosé-Hoover thermostat and barostat. This is possible by setting the parameter ``langevin`` to ``False`` which by default is set to ``True``.
To run a NPT sampling, we need to define the target pressure. If the target is set to ``None``, we switch to a NVT sampling.
It is possible to define how to apply the pressure: 

    - ``ptype=isotropic``: all 3 diagonal components together are coupled when pressure is computed (use by default).
    - ``ptype=anisotropic``:  x, y, and z dimensions are controlled independently 

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 24-29

NPT sampling imposes to compute stresses. This is possible by setting a weight on the MLIP stress coefficient.

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 30-36

Supercell creation using ASE atoms objects. In this case, a 2x2x2 cubic FCC aluminum supercell.

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 38-40

In this example, LAMMPS is used for the NPT MD simulation and the descriptors calculation. As in the previous example, we need to set an ASE env variable to run LAMMPS.

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 42-44

Now, we can create the three main objects of a MLACS simulation:

    - the Mlip object to compute descriptors and to fit the potential (mlip).
    - the State object to define the sampling method (state).
    - the Calculator object to compute the energies, forces and stresses (calc). In this case, we use an effective medium theory calculator for simplicity and rapidity. 

.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 48-63

Now, we can run the simulation:
 
.. literalinclude:: ../../../examples/LammpsState_NPT_Al300K.py
    :lines: 65-69
