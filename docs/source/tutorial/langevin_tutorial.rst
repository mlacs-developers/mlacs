.. _langevin_tutorial:


On-The-Fly MLACS simulation of the canonical distribution of Copper at 300K
===========================================================================

In this tutorial, we will see how to setup on a fcc Cu, a MD simulation at a 300K using a langevin integration.
For simplicity and rapidity, the potential will be the effective medium theory calculator.


Setting the simulation
----------------------

We start by importing the packages

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 1-8

Parameters for MLACS simulation:

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 16-21

Parameters for the MD sampling in a NVT ensemble and using the ASE implementation of a langevin integrator:

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 22-25

Settings for the MLIP. Here, we use a SNAP potential has implemented in LAMMPS with bispectrum descriptors.

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 26-28

Supercell creation using ASE atoms objects. In this case, a 2x2x2 cubic FCC Copper supercell.

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 30-32

The descriptors calculation is done by LAMMPS in the present case, so we need to set an ASE env variable to run this type of calculation using LAMMPS. 

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 34-36

Now, we can create the three main objects of a MLACS simulation:

    - the Mlip object to compute descriptors and to fit the potential (mlip).
    - the State object to define the sampling method (state).
    - the Calculator object to compute the energies, forces and stresses (calc). In this case, we use an effective medium theory calculator for simplicity and rapidity. 

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 38-47

Now, we can run the simulation:
 
.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 49-53


