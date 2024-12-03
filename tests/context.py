"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
import os
import sys
import shlex
import shutil
import importlib.util
from subprocess import Popen, PIPE
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa
import mlacs  # noqa


def has_lammps_nompi():
    """
    Returns True if Lammps doesn't have mpi.
    """
    envvar = "ASE_LAMMPSRUN_COMMAND"
    exe = os.environ.get(envvar)
    from mlacs.utilities.io_lammps import get_lammps_command
    exe = get_lammps_command()
    error = b'ERROR: Processor partitions do not match number of allocated'
    if exe is None:
        exe = "lmp_mpi"
    cmd = f"{exe} -h"
    lmp_info = Popen(shlex.split(cmd), stdout=PIPE).communicate()[0]
    if b'REPLICA' not in lmp_info:
        return True
    cmd = f"mpirun -n 2 {exe} -partition 2x1"

    if shutil.which("mpirun") is None or shutil.which(exe) is None:
        return True

    lmp_info = Popen(shlex.split(cmd), stdout=PIPE).communicate()[0]
    if error in lmp_info:
        return True
    return False


def has_mlp():
    """
    Returns True if there is no mlp executable.
    """
    return shutil.which("mlp") is None

def has_pyace():
    """
    Returns True if there is no pyace module
    """
    try:
        import pyace
        return False
    except ImportError:
        return True

def has_pyace():
    """
    Returns True if there is no pyace module
    """
    try:
        import pyace  # noqa
        return False
    except ImportError:
        return True


def has_netcdf():
    """
    Returns True if there is no netCDF4 package.
    """
    if importlib.util.find_spec('netCDF4') is None:
        return True
    return False
