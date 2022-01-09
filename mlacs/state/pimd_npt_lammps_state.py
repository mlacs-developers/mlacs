"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state.pimd_lammps_state import PimdLammpsState
from mlacs.utilities import get_elements_Z_and_masses


#========================================================================================================================#
#========================================================================================================================#
class PimdNptLammpsState(PimdLammpsState):
    """
    State Class for running a PIMD NPT simulation as implemented in LAMMPS

    Parameters
    ----------

    nbeads : :class:`int`
        Number of replicas for the path integral molecular dynamics. 
    temperature : :class:`float`
        Temperature of the simulation, in Kelvin
    pressure : float
        Pressure for the simulation, in GPa
    gjf : :class:`Bool` (optional)
        If true, the 2half GJF integrator is used.
        Else, the standard Velocity-Verlet Langevin integrator is used.
        Default ``True``.
    ptype : ``\"iso\"`` or ``\"aniso\"`` (optional)
        Type of external strain tensor to manage the
        deformation of the cell. Default 'iso'.
    damp : :class:`float` (optional)
        Damping parameter. If None a damping parameter of ``100`` times dt is used.
        Default ``None``.
    pdamp : :class:`float` (optional)
        Damping parameter for the barostat. Default ``1000`` times ``dt`` is used.
        Default ``None``.
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    neighbourlist : :class:`int` (optional)
        Frequency of the neighbour list update during the MLPIMD. Default ``100``.
    nprocperbead : :class:`int` (optional)
        Number of process per replica. default ``1``.
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
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default correspond to :class:`numpy.random.default_rng()`
    init_momenta : :class:`numpy.ndarray` (optional)
        If ``None``, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations. If ``None``, a LammpsMLMD
        directory is created
    """
    def __init__(self,
                 nbeads,
                 temperature,
                 pressure,
                 ptype="iso",
                 gjf=False,
                 damp=None,
                 pdamp=None,
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
        
        PimdLammpsState.__init__(self,
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
        self.pressure = pressure
        self.ptype    = ptype
        self.gjf      = gjf
        self.damp     = damp
        self.pdamp    = pdamp


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

        pdamp = self.pdamp
        if self.pdamp is None:
            pdamp = "$(1000*dt)"

        input_string  = ""
        input_string += self.get_general_input(pbc, masses)

        input_string += self.get_interaction_input(pair_style, pair_coeff)

        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "\n"

        input_string += "fix           f1 all rpmd {0}\n".format(self.temperature)
        if self.gjf:
            input_string += "fix           f3 all langevin {0} {0} {1} {2}${{ibead}} zero yes gjf vhalf\n".format(self.temperature, damp, self.rng.integers(9999999))
        else:
            input_string += "fix           f3 all langevin {0} {0} {1} {2}${{ibead}} zero yes\n".format(self.temperature, damp, self.rng.integers(9999999))
        input_string += "fix           f2 all nph  {0} {1} {1} {2}\n".format(self.ptype, self.pressure * 10000, pdamp)
        if self.fixcm:
            input_string += "fix           f4 all recenter INIT INIT INIT\n"
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
        pdamp = self.pdamp
        if pdamp is None:
            pdamp = 1000 * self.dt

        msg  = "PIMD NPT Langevin dynamics as implemented in LAMMPS\n"
        msg += "Number of beads                          {0}\n".format(self.nbeads)
        msg += "Temperature (in Kelvin)                  {0}\n".format(self.temperature)
        msg += "Number of MLPIMD equilibration steps :   {0}\n".format(self.nsteps_eq)
        msg += "Number of MLPiMD production steps :      {0}\n".format(self.nsteps)
        msg += "Timestep (in fs) :                       {0}\n".format(self.dt)
        msg += "Damping parameter (in fs) :              {0}\n".format(damp)
        msg += "Barostat damping parameter (in fs) :     {0}\n".format(pdamp)
        msg += "\n"
        return msg
