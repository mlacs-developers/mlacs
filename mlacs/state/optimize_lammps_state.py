"""
// (c) 2023 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import run, PIPE

from ase.io import read
from ase.io.lammpsdata import write_lammps_data

from .lammps_state import LammpsState
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import (get_general_input,
                                   get_log_input,
                                   get_minimize_input,
                                   get_traj_input,
                                   get_interaction_input,
                                   get_last_dump_input)


# ========================================================================== #
# ========================================================================== #
class OptimizeLammpsState(LammpsState):
    """
    Class to manage geometry optimizations with LAMMPS


    Parameters
    ----------
    min_style: :class:`str`
        Choose a minimization algorithm to use when a minimize command is
        performed.
        Default `cg`.

    etol: :class:`float`
        Stopping tolerance for energy
        Default ``0.0``

    ftol: :class:`float`
        Stopping tolerance for energy
        Default ``1.0e-6``

    pressure: :class:`float` or ``None`` (optional)
        Target pressure for the optimization, in GPa.
        If ``None``, no cell relaxation is applied.
        Default ``None``

    ptype: ``iso`` or ``aniso`` (optional)
        Handle the type of pressure applied. Default ``iso``

    vmax: ``iso`` or ``aniso`` (optional)
        The vmax keyword can be used to limit the fractional change in the
        volume of the simulation box that can occur in one iteration of
        the minimizer.
        Default ``1.0e-3``

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``0.5`` fs.

    nsteps : :class:`int` (optional)
        Maximum number of minimizer iterations during production phase.
        Also sets up the max number of force/energy evaluations.
        Default ``10000`` steps.

    nsteps_eq : :class:`int` (optional)
        Maximum number of minimizer iterations during equilibration phase.
        Also sets up the max number of force/energy evaluations.
        Default ``1000`` steps.

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
    """
    def __init__(self,
                 min_style='cg',
                 etol=0.0,
                 ftol=1.0e-6,
                 dt=0.5,
                 pressure=None,
                 ptype="iso",
                 vmax=1.0e-3,
                 nsteps=10000,
                 nsteps_eq=1000,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 workdir=None):
        LammpsState.__init__(self,
                             temperature=None,
                             pressure=None,
                             ptype="iso",
                             dt=dt,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq,
                             logfile=logfile,
                             trajfile=trajfile,
                             loginterval=loginterval,
                             workdir=workdir)

        self.style = min_style
        self.criterions = (etol, ftol)
        self.vmax = vmax
        self.langevin = False

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False):
        """
        Function to run the dynamics
        """

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        atoms = supercell.copy()

        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        write_lammps_data(self.workdir + self.atomsfname,
                          atoms,
                          atom_style=atom_style)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        self.write_lammps_input(atoms,
                                atom_style,
                                pair_style,
                                pair_coeff,
                                model_post,
                                nsteps)

        lammps_command = f"{self.cmd} -partition {self.nbeads}x1 -in " + \
            f"{self.lammpsfname} -sc out.lmp"
        lmp_handle = run(lammps_command,
                         shell=True,
                         cwd=self.workdir,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

        if charges is not None:
            init_charges = atoms.get_initial_charges()
        fname = "configurations.out"
        atoms = read(f"{self.workdir}{fname}")
        if charges is not None:
            atoms.set_initial_charges(init_charges)

        return atoms.copy()

# ========================================================================== #
    def write_lammps_input(self,
                           atoms,
                           atom_style,
                           pair_style,
                           pair_coeff,
                           model_post,
                           nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms)
        pbc = atoms.get_pbc()

        input_string = ""
        input_string += get_general_input(pbc,
                                          masses,
                                          charges,
                                          atom_style,
                                          nbeads=self.nbeads,
                                          ispimd=self.ispimd)
        input_string += get_interaction_input(pair_style,
                                              pair_coeff,
                                              model_post)
        if self.logfile is not None:
            input_string += get_log_input(self.loginterval, self.logfile)
        if self.trajfile is not None:
            input_string += get_traj_input(self.loginterval,
                                           self.trajfile,
                                           elem)

        input_string += get_last_dump_input(self.workdir,
                                            elem,
                                            1,
                                            self.nbeads)

        input_string += get_minimize_input(self.style,
                                           self.criterions,
                                           nsteps,
                                           self.pressure,
                                           self.ptype,
                                           self.vmax)

        with open(self.workdir + "lammps_input.in", "w") as f:
            f.write(input_string)
