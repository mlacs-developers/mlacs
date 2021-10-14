import numpy as np

from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import LammpsState
from mlacs.utilities import get_elements_Z_and_masses



#========================================================================================================================#
#========================================================================================================================#
class LangevinLammpsState(LammpsState):
    """
    State Class for running a Langevin simulation as implemented in LAMMPS

    Parameters
    ----------

    temperature : float
        Temperature of the simulation, in Kelvin
    gjf : bool (optional)
        If true, the 2half GJF integrator is used.
        Else, the standard Velocity-Verlet Langevin integrator is used.
        Default True.
    damp : float (optional)
        Damping parameter. If None, a damping parameter of a hundred time the timestep is used.
    dt : float (optional)
        Timestep, in fs. Default 1.5 fs.
    nsteps : int (optional)
        Number of MLMD steps for production runs. Default 1000 steps.
    nsteps_eq : int (optional)
        Number of MLMD steps for equilibration runs. Default 100 steps.
    fixcm : bool (optional)
        Fix position and momentum center of mass. Default True.
    logfile : str (optional)
        Name of the file for logging the MLMD trajectory.
        If none, no log file is created. Default None.
    trajfile : str (optional)
        Name of the file for saving the MLMD trajectory.
        If none, no traj file is created. Default None.
    interval : int (optional)
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default 50
    loginterval : int (optional)
        Number of steps between MLMD logging. Default 50.
    trajinterval : int (optional)
        Number of steps between MLMD traj writing. Default 50.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default correspond to numpy.random.default_rng()
    init_momenta : array (optional)
        If None, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    workdir : str (optional)
        Working directory for the LAMMPS MLMD simulations. If none, a LammpsMLMD
        directory is created
    """
    def __init__(self,
                 temperature,
                 gjf=True,
                 damp=None,
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 workdir=None
                ):

        LammpsState.__init__(self,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval,
                             rng,
                             init_momenta,
                             workdir
                            )

        self.temperature = temperature
        self.damp        = damp
        self.gjf         = gjf


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"

        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + self.atomsfname
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"

        #input_string += "velocity  all create {0}  {1}\n".format(self.temperature, self.rng.integers(999999))


        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "\n"

        if self.gjf:
            input_string += "fix    1  all langevin {0} {0}  {1} {2}  gjf vhalf\n".format(self.temperature, damp, self.rng.integers(99999))
        else:
            input_string += "fix    1  all langevin {0} {0}  {1}  {2}\n".format(self.temperature, damp, self.rng.integers(99999))
        input_string += "fix    2  all nve\n"

        if self.fixcm:
            input_string += "fix    3  all recenter INIT INIT INIT"

        if self.logfile is not None:
            input_string += self.get_log_in()
        if self.trajfile is not None:
            input_string += self.get_traj_in(elem)


        input_string += "# Dump last step\n"
        input_string += "dump last all custom {0} ".format(nsteps) + self.workdir + "configurations.out  id type xu yu zu vx vy vz fx fy fz element\n"
        input_string += "dump_modify last element "
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "dump_modify last delay {0}\n".format(nsteps)
        input_string += "\n"

        input_string += "run  {0}".format(nsteps)


        with open(self.lammpsfname, "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta is None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature, rng=self.rng)
        else:
            atoms.set_momenta(self.init_momenta)


#========================================================================================================================#
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = None
        if damp is None:
            damp = 1 / fs
        if self.gjf == True:
            integrator = "2 half GJF"
        else:
            integraotr = "Velocity-verlet"

        msg  = "Simulated state :\n"
        msg += "NVT Langevin dynamics as implemented in LAMMPS\n"
        msg += "Temperature (in Kelvin)                  {0}\n".format(self.temperature)
        msg += "Number of MLMD equilibration steps :     {0}\n".format(self.nsteps_eq)
        msg += "Number of MLMD production steps :        {0}\n".format(self.nsteps)
        msg += "Timestep (in fs) :                       {0}\n".format(self.dt)
        msg += "Damping parameter (in fs) :              {0}\n".format(damp)
        if self.gjf:
            msg += "A 2 half GJF integrator is used\n"
        else:
            msg += "A velocity-Verlet integrator is used\n"
        msg += "\n"
        return msg
