import numpy as np

from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import LammpsState
from mlacs.utilities import get_elements_Z_and_masses



#========================================================================================================================#
#========================================================================================================================#
class PIMDLammpsState(LammpsState):
    """
    State Class for running a PIMD simulation as implemented in LAMMPS

    Parameters
    ----------

    temperature : float
        Temperature of the simulation, in Kelvin
    nbead: int
        Number of bead for the quantum polymer. Default 8
    method : str
        Method for the ML-PIMD can be 'pimd', 'nmpimd' or 'cmd'. Default 'npimd'
    sp : float
        Scaling factor on Planck constant. Default None
    fmass : float
        Scaling factor on masses. Default None
    nhc : int
        Number of chain in Nose-Hoover thermostat. Default None
    dt : float
        Timestep, in fs. Default 1.5 fs
    nsteps : int
        Number of MLMD steps for production runs. Default 1000
    nsteps_eq : int
        Number of MLMD steps for equilibration runs. Default 100
    fixcm : bool
        Fix position and momentum center of mass. Default True
    logfile : str
        Name of the file for logging the MLMD trajectory. Default None
    trajfile : str
        Name of the file for saving the MLMD trajectory. Default None
    interval : int
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default 50
    loginterval : int
        Number of steps between MLMD logging. Default 50
    trajinterval : int
        Number of steps between MLMD traj writing. Default 50
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default None, corresponding to numpy.random.default_rng()
    init_momenta : array (optional)
        If None, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    workdir : str (optional)
        Working directory for the LAMMPS MLMD simulations. If none, a LammpsMLMD
        directory is created
    """
    def __init__(self,
                 temperature,
                 nbead=8,
                 method='nmpimd',
                 sp=None,
                 fmass=None,
                 nhc=None,
                 dt=1.5*fs,
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
        self.nbead       = nbead
        self.method      = method
        self.sp          = sp
        self.fmass       = fmass
        self.nhc         = nhc


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)

        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "atom_modify  path yes\n"
        input_string += "read_data    " + self.atomsfname
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"
        input_string += "\n"

        input_string += "variable ibead uloop {0}\n".format(self.nbead)
        input_string += "\n"

        input_string += "velocity  all create {0}  {1}${{ibead}} dist gaussian\n".format(self.temperature, self.rng.integers(999999))
        input_string += "\n"
        input_string += "\n"


        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"
        input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt/ (fs * 1000))
        input_string += "\n"

        input_string += "fix 1 all pimd method {0} temp {1} ".format(self.method, self.temperature)
        if self.fmass:
            input_string += "fmass {0} ".format(self.fmass)
        if self.sp:
            input_string += "sp {0} ".format(self.sp)
        if self.nhc:
            input_string += "nhc {0}".format(self.nhc)
        input_string += "\n"

        if self.fixcm:
            input_string += "fix    2  all recenter INIT INIT INIT\n"
        input_string += "\n"
        input_string += "\n"

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
