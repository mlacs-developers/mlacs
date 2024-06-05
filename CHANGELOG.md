CHANGE LOG
==========

********************************************************************************************

# Todo

## Tests and examples
* Add the missing potentials in the package
* Remove the MPI feature from tests
* Write a file listing all the tests
* Reduce the computational time of various tests
* Create a workdir to run the tests

## Output files
* Update some printing to the log (mlip, calculator, ...)
* **Francois** : Add informations in the header and footer of the log file **(papers, documentation, license...)**
* Create a NetCDF file with all the configurations (useful for qagate)
    * or concatenate all the GSR.nc files
    * or print and concatenate all the HIST.nc files
    * or using ASE?

## Documentation
* Verify and update the comments at the start of each object
* Finish the documentation
* Document the **mlacs** commands (correlation...)
* Create a logo
* [ReadTheDocs](https://about.readthedocs.com/?ref=readthedocs.com)

## Architecture
* Create two branches for development and production
* Separate the CHANGELOG.md (and move the Todo list in the developement branch)
* Create a command **mlacs clean** 

## Features
* **Aloïs** : Check the MTP implementation using MLIP-3
* Give the possibility to run DFT calculation in parallel
* **Francois** : Stop the SCF loop after *nconf*
* Merge the weights and configurations in *atoms* ?
* **Pauline** and **Gabriel** : Merge the NETI
* **Francois** Compute the phonon spectrum using aTDEP (thanks to the script done by **Olivier**)
* **Francois** Print the thermodynamic quantities in single file: P, V, T... but also F, G... (using NETI)
* Give the possibility to the user to set equal masses for all the elements (to accelerate the dynamic).
* Go continuously from nstep_eq to nstep

## MLIP
* Use relative paths rather than absolute
* Write a function to create a pair_style/pair_coeff starting from a json

## Troubleshot issues and bugs
* Sometimes the RMSE oscillates between two values (add a mixing?)
* **Francois** Is there any kT prefactor in front of the LS(forces) coming from the cost function (DKL vs. DF)? Rewrite the equations.
* The number of weights is different from the number of configurations
* After a restart, during trajectory reading, if the calculation stop, the symbolic link for MLIP does not point toward the last MLIP.
* NETI: Clean the directory (ThermoInt) before launching the calculation otherwise the free energy calculation will be false.

********************************************************************************************
# Unreleased

## Added

* ACE and MTP potential
* New weights (uniform and non-uniform)
* New examples, tests (with pytest) and tutorials
* Database calculators
* CalcProperty
* File management system for files and logs (*workdir*, *folder* and *subfolder*)
* ...

## Changed

* MBAR
* NETI
* states
* documentation
* ...

## Removed

* RdfLammpsState
* ...

********************************************************************************************
# 0.0.13

## Added

* Thermodynamic integration 
* Reversible scaling
* Tests *Pytest* (list?)
* Tutorials *Jupyter Notebook* (list?)
* Plots *correlation*, *error*, *weights* using **CLI**
* DeltaLearningPotential
* PAFI
* ...

## Changed

* MBAR
* Modularisation of MLACS
* ...

## Removed

* ...
