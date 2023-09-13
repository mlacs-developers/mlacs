"""
// (c) 2023 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import run, PIPE

from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .lammps_state import LammpsState
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import write_atoms_lammps_spin_style


# ========================================================================== #
# ========================================================================== #
class PimdLammpsState(LammpsState):
    """
    Class to manage PIMD simulations with LAMMPS
    """
    def __init__(self,
                 temperature,
                 temperature_spin,
                 pressure=None,
                 t_stop=None,
                 p_stop=None,
                 damp=None,
                 pdamp=None,
                 ptype="iso",
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 nbeads=1,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 msdfile=None,
                 rdffile=None,
                 rng=None,
                 init_momenta=None,
                 workdir=None):
        LammpsState.__init__(self,
                             temperature=temperature,
                             pressure=pressure,
                             t_stop=t_stop,
                             p_stop=p_stop,
                             damp=damp,
                             pdamp=pdamp,
                             ptype=ptype,
                             dt=dt,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq,
                             fixcm=fixcm,
                             logfile=logfile,
                             trajfile=trajfile,
                             loginterval=loginterval,
                             rng=rng,
                             init_momenta=init_momenta,
                             workdir=workdir)

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     nbeads_return=1):

        atoms = supercell.copy()
        spins = atoms.get_array("spins")

        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        if self.t_stop is None:
            temp = self.temperature
        else:
            if eq:
                temp = self.t_stop
            else:
                temp = self.rng.uniform(self.temperature, self.t_stop)

        if self.p_stop is None:
            press = self.pressure
        else:
            if eq:
                press = self.pressure
            else:
                press = self.rng.uniform(self.pressure, self.p_stop)

        if self.t_stop is not None:
            MaxwellBoltzmannDistribution(atoms,
                                         temperature_K=temp,
                                         rng=self.rng)
        write_lammps_data(self.workdir + self.atomsfname,
                          atoms,
                          velocities=True,
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
                                nsteps,
                                temp,
                                press)

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
        if self.nbeads > 1:
            ndigit = len(str(self.nbeads))
            # Will be changed for the full PIMD simulations
            fname += f"_{1:0{ndigit}d}"
        atoms = read(f"{self.workdir}{fname}")
        if charges is not None:
            atoms.set_initial_charges(init_charges)

        return atoms.copy()
