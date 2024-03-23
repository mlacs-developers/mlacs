import os
import sys
import shlex
import shutil
from subprocess import run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa
import mlacs  # noqa


def has_lammps_nompi():
    """
    Returns True if Lammps doesn't have mpi.
    """
    envvar = "ASE_LAMMPSRUN_COMMAND"
    exe = os.environ.get(envvar)
    if exe is None:
        exe = "lmp_mpi"
    cmd = f"mpirun -n 2 {exe} -h -partition 1x2"
    lmp_handle = run(shlex.split(cmd))
    return lmp_handle.returncode != 0


def has_mlp():
    """
    Returns True if there is no mlp executable.
    """
    return shutil.which("mlp") is None
