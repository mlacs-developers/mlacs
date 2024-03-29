import os
import sys
import shlex
import shutil
from subprocess import Popen, PIPE
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa
import mlacs  # noqa


def has_lammps_nompi():
    """
    Returns True if Lammps doesn't have mpi.
    """
    envvar = "ASE_LAMMPSRUN_COMMAND"
    exe = os.environ.get(envvar)
    error = b'ERROR: Processor partitions do not match number of allocated'
    if exe is None:
        exe = "lmp_mpi"
    cmd = f"{exe} -h"
    lmp_info = Popen(shlex.split(cmd), stdout=PIPE).communicate()[0]
    if b'REPLICA' not in lmp_info:
        return True
    cmd = f"mpirun -2 {exe} -partition 2x1"
    lmp_info = Popen(shlex.split(cmd), stdout=PIPE).communicate()[0]
    if error in lmp_info:
        return True
    return False


def has_mlp():
    """
    Returns True if there is no mlp executable.
    """
    return shutil.which("mlp") is None
