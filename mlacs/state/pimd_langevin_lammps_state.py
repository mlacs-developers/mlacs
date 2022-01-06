"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from ase.units import fs

from mlacs.state.pimd_lammps_state import PIMDLammpsState
from mlacs.utilities import get_elements_Z_and_masses


#========================================================================================================================#
#========================================================================================================================#
class PIMDLangevinLammpsState(PIMDLammpsState):
    """
    """
    def __init__(self,
                 nbeads,
                 temperature,
                 damp=None,
                 dt=1.0,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 neighbourlist=100,
                 nprocperbead=1,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 workdir=None
                ):
        
        PIMDLammpsState.__init__(self,
                                 nbeads,
                                 temperature,
                                 dt,
                                 nsteps,
                                 nsteps_eq,
                                 fixcm,
                                 neighbourlist,
                                 nprocperbead,
                                 logfile,
                                 trajfile,
                                 interval,
                                 loginterval,
                                 rng,
                                 init_momenta,
                                 workdir
                                )
        self.damp = damp


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms[0])
        pbc             = atoms[0].get_pbc()

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"

        input_string  = ""
        input_string += self.get_general_input(pbc, masses)

        input_string += self.get_interaction_input(pair_style, pair_coeff)

        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "\n"

        input_string += "fix           f1 all nve\n"
        input_string += "fix           f2 all rpmd {0}\n".format(self.temperature)
        input_string += "fix           f3 all langevin {0} {0} {1} {2}${{ibead}}\n".format(self.temperature, damp, self.rng.integers(9999999))
        if self.fixcm:
            input_string += "fix           f4  all recenter INIT INIT INIT\n"
        input_string += "\n\n\n"

        if self.logfile is not None:
            input_string += self.get_log_in()
        if self.trajfile is not None:
            input_string += self.get_traj_in(elem)

        input_string += self.get_last_dump_input(elem, nsteps)
        input_string += "run           {0}".format(nsteps)

        with open(self.lammpsfname, "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = None
        if damp is None:
            damp = 1 / fs

        msg  = "PIMD NVT Langevin dynamics as implemented in LAMMPS\n"
        msg += "Number of beads                          {0}\n".format(self.nbeads)
        msg += "Temperature (in Kelvin)                  {0}\n".format(self.temperature)
        msg += "Number of MLPIMD equilibration steps :   {0}\n".format(self.nsteps_eq)
        msg += "Number of MLPiMD production steps :      {0}\n".format(self.nsteps)
        msg += "Timestep (in fs) :                       {0}\n".format(self.dt)
        msg += "Damping parameter (in fs) :              {0}\n".format(damp)
        msg += "\n"
        return msg
