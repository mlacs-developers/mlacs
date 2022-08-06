"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import call

import numpy as np
from ase.io.lammpsdata import write_lammps_data

from mlacs.utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class ThermoState:
    """
    Parent class for the thermodynamic state used in thermodynamic integration

    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        ASE atoms object on which the simulation will be performed
    pair_style: :class:`str`
        pair_style for the LAMMPS input
    pair_coeff: :class:`str` or :class:`list` of :class:`str`
        pair_coeff for the LAMMPS input
    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1.5``
    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.
    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.
    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    interval : :class:`int` (optional)
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default ``50``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    trajinterval : :class:`int` (optional)
        Number of steps between MLMD traj writing. Default ``50``.
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 dt=1.5,
                 nsteps=10000,
                 nsteps_eq=5000,
                 rng=None,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50,
                 trajinterval=50):

        self.atoms = atoms
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.dt = dt
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self._get_lammps_command()

        self.logfile = logfile
        self.trajfile = trajfile
        self.loginterval = loginterval
        self.trajinterval = trajinterval
        if interval is not None:
            self.loginterval = interval
            self.trajinterval = interval

        self.elem, self.Z, self.masses, self.charges = \
            get_elements_Z_and_masses(self.atoms)

# ========================================================================== #
    def run_dynamics(self, wdir):
        """
        """
        atomsfname = wdir + "atoms.in"
        lammpsfname = wdir + "lammps_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input(wdir)
        call(lammps_command, shell=True, cwd=wdir)

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd

# ========================================================================== #
    def write_lammps_input(self, atoms):
        """
        Write the LAMMPS input for the MD simulation
        """
        raise NotImplementedError

# ========================================================================== #
    def get_workdir(self):
        """
        """
        return self.suffixdir

# ========================================================================== #
    def post_process(self):
        """
        """
        pass

# ========================================================================== #
    def get_log_input(self, suffix=None):
        """
        """
        input_string = "#####################################\n"
        input_string += "#          Logging\n"
        input_string += "#####################################\n"
        input_string += "variable    t equal step\n"
        input_string += "variable    mytemp equal temp\n"
        input_string += "variable    mype equal pe\n"
        input_string += "variable    myke equal ke\n"
        input_string += "variable    myetot equal etotal\n"
        input_string += "variable    mypress equal press/10000\n"
        input_string += "variable    mylx  equal lx\n"
        input_string += "variable    myly  equal ly\n"
        input_string += "variable    mylz  equal lz\n"
        input_string += "variable    vol   equal (lx*ly*lz)\n"
        input_string += "variable    mypxx equal pxx/(vol*10000)\n"
        input_string += "variable    mypyy equal pyy/(vol*10000)\n"
        input_string += "variable    mypzz equal pzz/(vol*10000)\n"
        input_string += "variable    mypxy equal pxy/(vol*10000)\n"
        input_string += "variable    mypxz equal pxz/(vol*10000)\n"
        input_string += "variable    mypyz equal pyz/(vol*10000)\n"

        if suffix is None:
            input_string += 'fix mythermofile all print ' + \
                            f'{self.loginterval} "$t ' + \
                            '${myetot}  ${mype} ${myke} ${mytemp} ' + \
                            '${mypress} ${mypxx} ${mypyy} ' + \
                            '${mypzz} ${mypxy} ${mypxz} ${mypyz}" ' + \
                            'append mlmd.log title "# Step  Etot  Epot  ' + \
                            'Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'
        else:
            input_string += 'fix mythermofile all print ' + \
                            f'{self.loginterval} "$t ' + \
                            '${myetot}  ${mype} ${myke} ${mytemp} ' + \
                            '${mypress} ${mypxx} ${mypyy} ' + \
                            '${mypzz} ${mypxy} ${mypxz} ${mypyz}" ' + \
                            f'append mlmd_{suffix}.log ' + \
                            'title "# Step  Etot  Epot  ' + \
                            'Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_traj_input(self, suffix=None):
        """
        """
        input_string = "#####################################\n"
        input_string += "#           Dumping\n"
        input_string += "#####################################\n"
        if suffix is None:
            input_string += f"dump dum1 all custom {self.trajinterval} " + \
                            f"{self.trajfile} id type xu yu zu " + \
                            "vx vy vz fx fy fz element \n"
        else:
            input_string += f"dump dum1 all custom {self.trajinterval} " + \
                            f"{self.trajfile}_{suffix} id type xu yu zu " + \
                            "vx vy vz fx fy fz element \n"
        input_string += "dump_modify dum1 append yes\n"
        input_string += "dump_modify dum1 element "  # Add element type
        input_string += " ".join([p for p in self.elem])
        input_string += "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_general_input(self):
        """
        """
        input_string = "# LAMMPS input file to run a MLMD simulation " + \
                       "for thermodynamic integration\n"
        input_string += "#####################################\n"
        input_string += "#           General parameters\n"
        input_string += "#####################################\n"
        input_string += "units        metal\n"

        pbc = self.atoms.get_pbc()
        input_string += "boundary     " + \
                        "{0} {1} {2}\n".format(*tuple("sp"[int(x)]
                                               for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    atoms.in\n"
        for i, mass in enumerate(self.masses):
            input_string += "mass         " + str(i + 1) + \
                            "  " + str(mass) + "\n"
        for iel, el in enumerate(self.elem):
            input_string += f"group        {el} type {iel+1}\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_interaction_input(self):
        """
        """
        input_string = "#####################################\n"
        input_string += "#           Interactions\n"
        input_string += "#####################################\n"
        input_string += "pair_style    " + self.pair_style + "\n"
        if isinstance(self.pair_coeff, list):
            for coeff in self.pair_coeff:
                input_string += "pair_coeff    " + coeff + "\n"
        else:
            input_string += "pair_coeff    " + self.pair_coeff + "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string
