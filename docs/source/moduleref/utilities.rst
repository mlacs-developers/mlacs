Utilities
#########

.. module:: mlacs.utilities.io_lammps

Lammps IO
*********

.. autoclass:: LammpsInput
   :members: add_block, to_string

.. autoclass:: LammpsBlockInput
   :members: add_variable, to_string, pop, extend

.. module:: mlacs.utilities.io_abinit

Abinit IO
*********

.. autoclass:: AbinitNC

.. autofunction:: set_aseAtoms 

.. module:: mlacs.utilities.thermo

Thermodynamic
*************

Functions to compute some thermodynamic properties

free_energy_harmonic_oscillator
===============================

.. autofunction:: free_energy_harmonic_oscillator

free_energy_com_harmonic_oscillator
===================================

.. autofunction:: free_energy_com_harmonic_oscillator

free_energy_ideal_gas
=====================

.. autofunction:: free_energy_ideal_gas

free_energy_uhlenbeck_ford
==========================

.. autofunction:: free_energy_uhlenbeck_ford

Path Integral
*************

compute_centroid_atoms
======================

.. autofunction:: mlacs.utilities.path_integral.compute_centroid_atoms

Minimum Energy Path
*******************

.. module:: mlacs.core
.. autoclass:: PathAtoms
   :members: images, initial, final, masses, xi, update, splined, get_splined_atoms, set_splined_matrices

Miscellanous
************

.. module:: mlacs.utilities.miscellanous
.. autofunction:: get_elements_Z_and_masses

.. autofunction:: create_random_structures

.. autofunction:: compute_correlation

.. autofunction:: compute_averaged

.. autofunction:: interpolate_points

.. autofunction:: integrate_points
