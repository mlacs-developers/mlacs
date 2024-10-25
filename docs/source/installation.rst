.. _installation:
.. index:: Installation

Installation
##############

Interfaces
==========

The code can serve as a link between many quantum simulations code and machine-learning models.

Any code having a ASE ``Calculator`` interface can be used as a reference.

The machine-learning potential¹ interfaced with MLACS at the moment are:

- SNAP model with linear or quadratic fit
- ML-IAP model with linear or quadratic fit
- Moment Tensor Potential of the mlip-2 package

¹These packages have to be installed on their own.

Install
=======

Python Environment
------------------

It is recommended to use a relatively recent version of python (>3.8) for optimal package operation. It is also a good idea to use python virtual environments, which makes it easier to manage python packages and their versions.
The creation of the environment is done with the python ``virtualenv`` module (or for recent versions of python, ``venv``).

.. code:: bash

    $ mkdir pyenv
    $ python3 -m virtualenv /path_to/pyenv

Loading environment²:

.. code:: bash

    $ source /path_to/pyenv/bin/activate

Installing packages using pip:

.. code:: bash

    $ pip install -r requirements.txt
    $ pip install -r requirements.txt --find-links /packages_directory --no-index #No internet

On calculator, the ``--find-links`` allows to specify the directory where the packages are, for that it is necessary to download them beforehand via a ``pip download`` or directly on `https://pypi.org/! <https://pypi.org/!>`__. The ``--no-index`` prevents pip from fetching packages from the online repository.
Finally, you can install MLACS:

.. code:: bash

    $ pip install --upgrade . # In main directory /path_to/mlacs/

At the end, we can check that the package is loaded:

.. code:: bash

    $ python
    >>> from mlacs import OtfMlacs

²The environment name in parentheses should appear on the terminal.

You can also run the tests to verify that the package is running properly. Tests are located in the `test/` repertory. You can then simply execute the pytest commmand: 

.. code:: bash

	$ pytest

.. admonition:: Warning

    - You need to define the ASE_LAMMPSRUN_COMMAND environment variable to specify where MLACS can find LAMMPS before running the tests (see below).
    - The pymbar and netCDF4 are needed to pass the tests. 
    - Some tests are passed if lammps has not been compile with the REPLICA package and if you haven't installed the mlp executable for Moment Tensor Potential (see below).

LAMMPS
------

It is recommended to use the latest version of `LAMMPS <https://docs.lammps.org/Manual.html>`__. The current version of MLACS works with the latest 'release' version of LAMMPS, which can be downloaded from the site or via git:

.. code:: bash

    $ git clone -b release https://github.com/lammps/lammps.git lammps

To compile LAMMPS, you have the choice between two options ``cmake`` or the classic ``make``.

.. code:: bash

    $ make purge             # remove any deprecated src files
    $ make package-update    # sync package files with src files

To limit the size of the executable, it is best to install only the packages you need. To do this, go to the source directory (``/src``) of LAMMPS, then:

.. code:: bash

    $ make no-all            # remove all packages
    $ make yes-nameofpackage # Add manually the package into the src directory
    $ make mpi               # re-build for your machine (mpi, serial, etc)

Several packages are necessary for the proper functioning of MLACS, here is a non-exhaustive list of recommended packages:

``ml-snap, ml-iap, manybody, meam, molecule, class2, kspace, replica,
extra-fix, extra-pair, extra-compute, extra-dump, qtb``

.. admonition:: Warning

    Some versions of LAMMPS are not compatible with certain versions of ASE. Versions prior to 03Aug22 are compatible with ASE versions prior to 3.22. For LAMMPS versions 03Aug22 and beyond, we hardly recommand to use ASE versions up to 3.23.

MLACS will then call LAMMPS through ASE, which relies on environment variables.
They can be set before running the simulation or by modifying environment variables directly in the python script.

.. code:: bash

    $ export ASE_LAMMPSRUN_COMMAND='lmp_serial'                              # Serial
    $ export ASE_LAMMPSRUN_COMMAND='mpirun -n 4 lmp_mpi'                     # MPI

ABINIT
------

MLACS provides interfaces with different codes through the ASE python package. But it is recommended to use `Abinit <https://www.abinit.org/>`__, since we design an ``AbinitManager`` to handle specific workflows with it. The Abinit package also provides several codes like ``atdep`` a useful tool to compute temperature dependent properties from MLACS trajectories.

`aTDEP <https://docs.abinit.org/guide/atdep/>`__ is based on the Temperature Dependent Effective Potential (TDEP) developed by O. Hellman et al. in 2011 and implemented in Abinit by J.Bouchet and F. Bottin in 2015.

It is also recommended to use the latest versions of Abinit, at least up to version 9, for an easier files management and to benefit of the newest ``atdep`` developments.
To compile Abinit, we highly recommend you to follow the instructions provided on the `website <https://docs.abinit.org/installation/>`__.

Python Packages
===============

MLACS uses very few external packages (and that is a choice), only ASE and its dependencies in its standard version. The necessary packages are included in the ``requirement.txt`` file located in the main directory ``/mlacs``. They can be downloaded in advance with the pip module.

.. code:: bash

    $ pip download -r /path_to/mlacs/requirements.txt

Required Packages
-----------------

ASE:

ASE is an atomic simulation environment, interfaced with several codes and written in order to set up, control and analyze atomic simulations. As mentioned previously, the correct version must be used for LAMMPS.

.. code:: bash

    $ git clone -b 3.23.1 https://gitlab.com/ase/ase.git

Then in the package directory:

.. code:: bash

    $ python setup.py install

pymbar:

Python implementation of the multistate Bennett acceptance ratio (MBAR) method for estimating expectations and free energy differences from equilibrium samples from multiple probability densities.

.. code:: bash

    $ git clone https://github.com/choderalab/pymbar.git

scikit-learn:

Advanced fitting method provided by the Scikit Learn package can be used instead of an Ordinary Least Squares method. From experience, a simple ``np.linalg.lstsq`` often suffices for fitting a simple linear MLIP. It is only recommended to use these advanced methods when you are using a quadratic MLIP. In this case, the number of coefficients increases exponentially and a simple Least Square method could fail. This package is also used for Gaussian Process.

netCDF4:

Python package to read netCDF binary format. This package can be really useful when you are using Abinit as Calculator, since it outputs a lot of useful information in the netCDF outputs.
MLACS also outputs thermodynamics properties, trajectories and results of an applied weighting policy using this file format. The files can be visualized using the `qAgate <https://github.com/piti-diablotin/qAgate>`__ visualization software or `AbiPy <http://abinit.github.io/abipy/>`__ an open-source library for analyzing the results produced by ABINIT.

Highly Recommended Packages
---------------------------

mlip-3 (or mlip-2):

The ``mlp`` software is used by MLACS to fit Moment Tensor Potentials (MTP). It has been developed at Skoltech (Moscow) by Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev, and Ivan Novikov.

.. code:: bash

    $ git clone https://gitlab.com/ashapeev/mlip-3.git

To use it you also need to recompile LAMMPS with the specific interface:

.. code:: bash

    $ git clone https://gitlab.com/ivannovikov/interface-lammps-mlip-3.git

pyace:

The `pyace <https://pacemaker.readthedocs.io/en/latest/>`__ (aka python-ace) package is used within MLACS to fit interatomic potentials in a general nonlinear Atomic Cluster Expansion (ACE) form. It contains the ``pacemaker`` tools and other Python wrappers and utilities.

.. code:: bash

    $ git clone https://github.com/ICAMS/python-ace

To use it you also need to recompile LAMMPS with the specific `interface <https://github.com/ICAMS/lammps-user-pace>`__ , which can be obtained from the LAMMPS source directory:

.. code:: bash

	$ cd lammps/src
	$ make lib-pace args="-b"
	$ make yes-ml-pace
	$ make mpi # or make serial

Optional Packages
-----------------

icet:

MLACS uses icet for Disorder Local Moment simulation and the Special Quasirandom Structures generator. DLM is a method to simulate an antiferromagnetic (colinear case) material by imposing periodically a random spin configuration.
