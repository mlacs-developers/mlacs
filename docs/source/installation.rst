.. _installation:
.. index:: Installation

Installation
============

LAMMPS
------
It is recommended to use the latest version of LAMMPS. The current version of MLACS (0.0.10) works with the latest 'release' version of LAMMPS (03Aug22), which can be downloaded from the site or via git:

.. code-block:: console

    $ git clone -b release https://github.com/lammps/lammps.git lammps

To compile LAMMPS, you have the choice between two options `cmake` or the classic `make`.

.. code-block:: console

    $ make purge             # remove any deprecated src files
    $ make package-update    # sync package files with src files

To limit the size of the executable, it is best to install only the packages you need. To do this, go to the source directory (`/src`) of LAMMPS, then:

.. code-block:: console

    $ make no-all            # remove all packages
    $ make yes-nameofpackage # Add manually the package into the src directory
    $ make mpi               # re-build for your machine (mpi, serial, etc)

Several packages are necessary for the proper functioning of MLACS, here is a non-exhaustive list of recommended packages:

.. code-block:: console

    ml-snap, ml-iap, manybody, molecule, class2, kspace, replica,
    extra-fix, extra-pair, extra-compute, extra-dump, user-rpmd,
    misc

**Warning!**
- There may be compilation problems with the `misc` package depending on the compiler used. The source of the problem often comes from the file `pair_list.cpp` in this case it is enough to edit it and delete the `_noalias` line 91 and 92.¹
- Some versions of LAMMPS are not compatible with certain versions of ASE. Versions prior to 03Aug22 are compatible with ASE versions prior to 3.23. For LAMMPS versions 03Aug22 and beyond, development versions of ASE must be used.

### Python Packages
MLACS uses very few external packages (and that is a choice), only ASE and its dependencies in its standard version. The necessary packages are included in the `requirement.txt` file located in the main directory `/otf_mlacs`. They can be downloaded in advance with the pip module.

.. code-block:: console

    $ pip download -r /path_to/otf_mlacs/requirements.txt

ASE
---
ASE is an atomic simulation environment, interfaced with several codes and written in order to set up, control and analyze atomic simulations. As mentioned previously, the correct version must be used for LAMMPS.

.. code-block:: console

    $ git clone -b 3.22.1 https://gitlab.com/ase/ase.git # If LAMMPS < 03Aug22 
    $ git clone -b 3.23.0b1 https://gitlab.com/ase/ase.git # If LAMMPS < 03Aug22

Then in the package directory

.. code-block:: console

    $ python setup.py install

i-Pi
----

Hiphive + Phonopy
-----------------

Pymbar
------

Python Environment
------------------
It is recommended to use a relatively recent version of python ($>$3.7) for optimal package operation. It is also a good idea to use python virtual environments, which makes it easier to manage python packages and their versions.
The creation of the environment is done with the python `virtualenv` (or for recent versions of python, `venv`) module.

.. code-block:: console

    $ mkdir pyenv
    $ python3 -m virtualenv /path_to/pyenv

Loading environment²:

.. code-block:: console

    $ source /path_to/pyenv/bin/activate

Installing packages using pip:

.. code-block:: console

    $ pip install packages
    $ pip install packages --find-links /packages_directory --no-index #No internet

On calculator, the `--find-links` allows to specify the directory where the packages are, for that it is necessary to download them beforehand via a `pip download` or directly on `https://pypi.org/!`. The `--no-index` prevents pip from fetching packages from the online repository.
Finally, you can install MLACS in editable version:

.. code-block:: console

    $ pip install -e /path_to/otf_mlacs/ # Path to setup.py

At the end, we can check that the package is loaded:

.. code-block:: console

    $ python
    >>> from mlacs import OtfMlacs

¹This modification has no impact on the operation of the code, the option only intervenes for compilation.
²The environment name in parentheses should appear on the terminal.
