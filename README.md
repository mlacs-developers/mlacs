Machine-Learning Assisted Canonical Sampling (MLACS)
====================================================

A python package to do Machine-Learning assisted simulation of atomic systems at finite temperature, focusing on thermodynamic properties.

Table of Contents
=================

- [Capabilities](#capabilities)
- [Background](#background)
- [Interfaces](#interfaces)
- [Install](#install)
- [Getting started](#gettingstarted)
- [License](#license)


Capabilities
============

This package provides a set of tools to perform various simulations bringing together machine-learning methods with quantum simulations of atomic systems.

Here are some features implemented in the package
- Sampling of the canonical distribution of a system through variational inference
- Training of machine-learning potential
- Free energy calculation with non-equilibrium thermodynamic integration
- Free energy fitting with gaussian process regression
- Free energy path with nudge elastic band or constrained sampling

Background
==========

Interfaces
==========

The code can serve as a link between many quantum simulations code and machine-learning models.

Any code having a ASE `Calculator` interface can be used as a reference.


The machine-learning potential¹ interfaced with MLACS at the moment are
- SNAP model with linear or quadratic fit
- ML-IAP model with linear or quadratic fit
- Moment Tensor Potential of the mlip-2 package

¹These packages have to be installed on their own.

Install
=======


Python Environment
------------------
It is recommended to use a relatively recent version of python (>3.8) for optimal package operation. It is also a good idea to use python virtual environments, which makes it easier to manage python packages and their versions.
The creation of the environment is done with the python `virtualenv` module (or for recent versions of python, `venv`).

    $ mkdir pyenv
    $ python3 -m virtualenv /path_to/pyenv

Loading environment²:

    $ source /path_to/pyenv/bin/activate

Installing packages using pip:

    $ pip install -r requirements.txt 
    $ pip install -r requirements.txt --find-links /packages_directory --no-index #No internet

On calculator, the `--find-links` allows to specify the directory where the packages are, for that it is necessary to download them beforehand via a `pip download` or directly on `https://pypi.org/!`. The `--no-index` prevents pip from fetching packages from the online repository.
Finally, you can install MLACS:

	$ pip install --upgrade . # In main directory /path_to/mlacs/

At the end, we can check that the package is loaded:

    $ python
    >>> from mlacs import OtfMlacs

²The environment name in parentheses should appear on the terminal.

You can also run the tests to verify that the package is running properly. Tests are located in the test/ repertory. You can then simply execute the pytest commmand: 

	$ pytest

> [!WARNING]
> You need to define the ASE_LAMMPSRUN_COMMAND environment variable to specify where MLACS can find LAMMPS before running the tests (see below).
> Some tests are passed if lammps has not been compile with the REPLICA package and if you haven't installed the mlp executable for Moment Tensor Potential (see below).


LAMMPS
------
It is recommended to use the latest version of [LAMMPS](https://docs.lammps.org/Manual.html). The current version of MLACS works with the latest 'release' version of LAMMPS, which can be downloaded from the site or via git:

    $ git clone -b release https://github.com/lammps/lammps.git lammps

To compile LAMMPS, you have the choice between two options `cmake` or the classic `make`.

    $ make purge             # remove any deprecated src files
    $ make package-update    # sync package files with src files

To limit the size of the executable, it is best to install only the packages you need. To do this, go to the source directory (`/src`) of LAMMPS, then:

    $ make no-all            # remove all packages
    $ make yes-nameofpackage # Add manually the package into the src directory
    $ make mpi               # re-build for your machine (mpi, serial, etc)

Several packages are necessary for the proper functioning of MLACS, here is a non-exhaustive list of recommended packages:

    ml-snap, ml-iap, manybody, meam, molecule, class2, kspace, replica,
    extra-fix, extra-pair, extra-compute, extra-dump, qtb
    
> [!WARNING]
> Some versions of LAMMPS are not compatible with certain versions of ASE. Versions prior to 03Aug22 are compatible with ASE versions prior to 3.22. For LAMMPS versions 03Aug22 and beyond, we hardly recommend to use ASE versions up to 3.23.

MLACS will then call LAMMPS through ASE, which relies on environment variables.
They can be set before running the simulation or by modifying environment variables directly in the python script.

    $ export ASE_LAMMPSRUN_COMMAND='lmp_serial'                              # Serial
    $ export ASE_LAMMPSRUN_COMMAND='mpirun -n 4 lmp_mpi'                     # MPI

ABINIT
------
MLACS provides interfaces with different codes through the ASE python package. But it is recommended to use [Abinit](https://www.abinit.org/), since we design an ``AbinitManager`` to handle specific workflows with it. The Abinit package also provides several codes like `atdep` a useful tool to compute temperature dependent properties from MLACS trajectories.

[aTDEP](https://docs.abinit.org/guide/atdep/) is based on the Temperature Dependent Effective Potential (TDEP) developped by O. Hellman et al. in 2011 and implemented in Abinit by J.Bouchet and F. Bottin in 2015.

It is also recommended to use the latest versions of Abinit, at least up to version 9, for an easier files management and to benefit of the newest `atdep` developement. 

To compile Abinit, we highly recommend you to follow the instructions provided on the [website](https://docs.abinit.org/installation/).


Python Packages
===============
MLACS uses very few external packages (and that is a choice), only ASE and its dependencies in its standard version. The necessary packages are included in the `requirement.txt` file located in the main directory `/mlacs`. They can be downloaded in advance with the pip module.

    $ pip download -r /path_to/mlacs/requirements.txt

Required Packages
-----------------
ASE:

ASE is an atomic simulation environment, interfaced with several codes and written in order to set up, control and analyze atomic simulations. As mentioned previously, the correct version must be used for LAMMPS.

    $ git clone -b 3.23.0 https://gitlab.com/ase/ase.git

Then in the package directory

    $ python setup.py install

pymbar:

Python implementation of the multistate Bennett acceptance ratio (MBAR) method for estimating expectations and free energy differences from equilibrium samples from multiple probability densities.

    $ git clone https://github.com/choderalab/pymbar.git

scikit-learn:

Advanced fitting method provided by the Scikit Learn package can be used instead of an Ordinary Least Squares method. From experience, a simple ``np.linalg.lstsq`` often suffice for fitting a simple linear MLIP. It is only recommended to use these advanced methods when you are using a quadratic MLIP. In this case, the number of coefficients increases exponentially and a simple Least Square method could fail. This package is also used for Gaussian Process. 

netCDF4:

Python package to read netCDF binary format. This package can be really usefull when you are using Abinit as Calculator, since it output a lot of usefull informations in the netCDF outputs. 
MLACS also outputs thermodynamics properties, trajectories and results of an applied weighting policy using this file format. The files can be visualized using the [qAgate](https://github.com/piti-diablotin/qAgate) visualization software or [AbiPy](http://abinit.github.io/abipy/) an open-source library for analyzing the results produced by ABINIT.

Highly Recomended Packages
--------------------------
mlip-3 (or mlip-2):

The ``mlp`` software is used by MLACS to fit Moment Tensor Potentials (MTP). It has been developed at Skoltech (Moscow) by Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev, and Ivan Novikov.

    $ git clone https://gitlab.com/ashapeev/mlip-3.git

To use it you also need to recompile LAMMPS with the specific interface:

    $ git clone https://gitlab.com/ivannovikov/interface-lammps-mlip-3.git

pyace:

The [pyace](https://pacemaker.readthedocs.io/en/latest/) (aka python-ace) package is used within MLACS to fit interatomic potentials in a general nonlinear Atomic Cluster Expansion (ACE) form. It contains the ``pacemaker`` tools and other Python wrappers and utilities.

    $ git clone https://github.com/ICAMS/python-ace

To use it you also need to recompile LAMMPS with the specific [interface](https://github.com/ICAMS/lammps-user-pace), which can be obtained from the LAMMPS source directory:

	$ cd lammps/src 
	$ make lib-pace args="-b"
	$ make yes-ml-pace
	$ make mpi # or make serial


Optional Packages
-----------------
icet:

MLACS uses icet for Disorder Local Moment simulation and the Special Quasirandom Structures generator. DLM is a method to simulate an antiferromagnetic (colinear case) material by imposing periodically a random spin configuration. 

Getting Started
===============

You can find some jupyter-notebooks in the tutorials folder

*NOTE* notebooks have to be created. It's simpler and more complete than script tutorials. Also it's much easier for new user apparently.
I will start making some for what we have currently to give examples


License
=======
MLACS is released under the GNU GPL license. For more details see the LICENSE file.
