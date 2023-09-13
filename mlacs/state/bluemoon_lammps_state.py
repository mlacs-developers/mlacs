from .pafi_lammps_state import PafiLammpsState


# ========================================================================== #
# ========================================================================== #
class BlueMoonLammpsState(PafiLammpsState):
    """
    Class to manage constrained MD along a linear reaction coordinate using
    the fix Pafi with LAMMPS. This is similar to a Blue Moon sampling.

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.
    reaction_coordinate: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        if ``None``, automatic search of the saddle point.
        Default ``None``
    maxjump: :class:`float`
        Maximum atomic jump authorized for the free energy calculations.
        Configurations with an high `maxjump` will be removed.
        Default ``0.4``
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    damp: :class:`float` or ``None``
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    langevin: :class:`Bool`
        If ``True``, a Langevin thermostat is used.
        Else, a Brownian dynamic is used.
        Default ``True``
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    prt : :class:`Bool` (optional)
        Printing options. Default ``True``
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
    """
    def __init__(self,
                 temperature,
                 configurations,
                 reaction_coordinate=None,
                 maxjump=0.4,
                 dt=1.5,
                 damp=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 langevin=True,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=49,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 prt=True,
                 workdir=None):
        PafiLammpsState.__init__(self,
                                 temperature,
                                 configurations,
                                 reaction_coordinate,
                                 1.0,
                                 maxjump,
                                 dt,
                                 damp,
                                 nsteps,
                                 nsteps_eq,
                                 langevin,
                                 fixcm,
                                 logfile,
                                 trajfile,
                                 interval,
                                 loginterval,
                                 trajinterval,
                                 rng,
                                 init_momenta,
                                 prt,
                                 workdir)
        self.xilinear = True
