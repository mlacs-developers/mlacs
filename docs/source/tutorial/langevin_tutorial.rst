.. _langevin_tutorial:


On-The-Fly MLACS simulation of the canonical distribution of Copper at 300K
===========================================================================

In this tutorial, we will see how to setup on a fcc Cu, a MD simulation at a 300K using a langevin integration.
For simplicity and rapidity, the potential will be the effective medium theory calculator.


Setting the simulation
----------------------


We start by importing the packages

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
<<<<<<< HEAD
    :lines: 1-8

Parameters for MLACS simulation:

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 16-21

Parameters for the MD sampling in a NVT ensemble and using the ASE implementation of a langevin integrator:

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 22-25

Settings for the MLIP. Here, we use a SNAP potential has implemented in LAMMPS with bispectrum descriptors.
||||||| merged common ancestors
    :lines: 1-6
=======
    :lines: 1-8
>>>>>>> 07414622dd097532f2d9640a5e2f5b2502bc37fe

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
<<<<<<< HEAD
    :lines: 26-28

Supercell creation using ASE atoms objects. In this case, a 2x2x2 cubic FCC Copper supercell.

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 30-32

The descriptors calculation is done by LAMMPS in the present case, so we need to set an ASE env variable to run this type of calculation using LAMMPS. 

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 34-36

Now, we can create the three main objects of a MLACS simulation the Mlip object for the calculation of descriptors and the potential fitting (mlip), the State object for the sampling method (state) and the Calculator defined as the reference (calc). In this case, we use an effective medium theory calculator for simplicity and rapidity. 
||||||| merged common ancestors
    :lines: 14-24
=======
    :lines: 16-26
>>>>>>> 07414622dd097532f2d9640a5e2f5b2502bc37fe

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
<<<<<<< HEAD
    :lines: 38-47
||||||| merged common ancestors
    :lines: 27-30
=======
    :lines: 29-35
>>>>>>> 07414622dd097532f2d9640a5e2f5b2502bc37fe

Now, we can run the simulation:
 
.. literalinclude:: ../../../examples/Langevin_Cu300K.py
<<<<<<< HEAD
    :lines: 49-53
||||||| merged common ancestors
    :lines: 33-44
=======
    :lines: 39-40

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 42-43

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 45-46

.. literalinclude:: ../../../examples/Langevin_Cu300K.py
    :lines: 48-49
>>>>>>> 07414622dd097532f2d9640a5e2f5b2502bc37fe
