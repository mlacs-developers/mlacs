"""
// Copyright (C) 2022-2024 MLACS group (PR, AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.mdMana
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
try:  # Scipy >= 1.6.0
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:  # Scipy < 1.6.0
    from scipy.integrate import cumtrapz

from ase.units import kB

from ..core.manager import Manager
from ..utilities.io_lammps import LammpsBlockInput

from .thermostate import ThermoState
from .solids import EinsteinSolidState
from .liquids import UFLiquidState
from .thermoint import ThermodynamicIntegration


# ========================================================================== #
# ========================================================================== #
class ReversibleScalingState(ThermoState):
    """
    Class for performing thermodynamic integration for a
    range of temperature using reversible scaling.

    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        ASE atoms object on which the simulation will be performed

    pair_style: :class:`str`
        pair_style for the LAMMPS input

    pair_coeff: :class:`str` or :class:`list` of :class:`str`
        pair_coeff for the LAMMPS input

    fcorr1: :class:`float` or ``None``
        First order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.

    fcorr2: :class:`float` or ``None``
        Second order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.

    t_start: :class:`float` (optional)
        Initial temperature of the simulation, in Kelvin. Default ``300``.

    t_end: :class:`float` (optional)
        Final temperature of the simulation, in Kelvin. Default ``1200``.

    fe_init: :class:`float` (optional)
        Free energy of the initial temperature, in eV/at. Default ``None``.

    phase: :class:`str`
        The phase of the system for which the free energy is computed.
        This input is used to compute the reference free energy at the
        starting pressure using a EinsteinSolidState object for solids
        and a UFLiquidState object for liquids. Can be either 'solid'
        or 'liquid'

    ninstance: :class:`int` (optional)
        If Free energy calculation has to be done before temperature sweep,
        settles the number of forward and backward runs. Default ``1``.

    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1.5``

    damp : :class:`float` (optional)
        Damping parameter. If ``None``, a damping parameter of a
        hundred time the timestep is used.

    pressure: :class:`float` or ``None``
        Pressure of the simulation.
        If ``None``, simulations are performed in the NVT ensemble.
        Default ``None``.

    pdamp : :class:`float` (optional)
        Damping parameter for the barostat. Default 1000 times ``dt`` is used.
        Default ``None``.

    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.

    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.

    gjf: :class:`bool`
        Whether to use the GJF integrator, if the Langevin thermostat is used.
        Default `True`

    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    langevin: :class:`bool`
        Whether to use a langevin thermostat. Default `True`

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.

    interval : :class:`int` (optional)
        Number of steps between log and traj writing. Override
        loginterval. Default ``50``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 fcorr1=None,
                 fcorr2=None,
                 t_start=300,
                 t_end=1200,
                 fe_init=None,
                 phase=None,
                 ninstance=1,
                 dt=1,
                 damp=None,
                 pressure=None,
                 pdamp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 gjf=True,
                 rng=None,
                 langevin=True,
                 logfile=None,
                 trajfile=None,
                 interval=500,
                 loginterval=50,
                 **kwargs):

        self.atoms = atoms
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2
        self.t_start = t_start
        self.t_end = t_end
        self.fe_init = fe_init
        self.phase = phase
        self.ninstance = ninstance
        self.damp = damp
        self.pressure = pressure
        self.langevin = langevin
        self.pdamp = pdamp
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq
        self.gjf = gjf
        self.dt = dt

        if kwargs.get('folder') is None:
            folder = f"ReversibleScaling_T{self.t_start}K_T{self.t_end}K"
            if self.pressure is None:
                folder += "_NVT"
            else:
                folder += f"_{self.pressure}GPa"
            kwargs['folder'] = folder

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        # reversible scaling
        ThermoState.__init__(self,
                             atoms,
                             pair_style,
                             pair_coeff,
                             dt,
                             nsteps,
                             nsteps_eq,
                             rng=rng,
                             logfile=logfile,
                             trajfile=trajfile,
                             interval=interval,
                             loginterval=loginterval,
                             **kwargs)

# ========================================================================== #
    @Manager.exec_from_path
    def run(self):
        """
        """

        if self.fe_init is None and self.phase is not None:
            self.run_single_ti()
        else:
            self.fe_init = 0.0

        self.run_dynamics(self.atoms, self.pair_style, self.pair_coeff)

        with open("MLMD.done", "w") as f:
            f.write("Done")

# ========================================================================== #
    def run_single_ti(self):
        """
        Free energy calculation before sweep
        """
        folder = self.folder + "/Fe_ref"
        if self.phase == 'solid':
            self.state = EinsteinSolidState(self.atoms,
                                            self.pair_style,
                                            self.pair_coeff,
                                            self.t_start,
                                            pressure=self.pressure,
                                            fcorr1=self.fcorr1,
                                            fcorr2=self.fcorr2,
                                            k=None,
                                            dt=self.dt,
                                            workdir=self.workdir,
                                            folder=folder)
        elif self.phase == 'liquid':
            self.state = UFLiquidState(self.atoms,
                                       self.pair_style,
                                       self.pair_coeff,
                                       self.t_start,
                                       pressure=self.pressure,
                                       fcorr1=self.fcorr1,
                                       fcorr2=self.fcorr2,
                                       dt=self.dt,
                                       workdir=self.workdir,
                                       folder=folder)

        self.ti = ThermodynamicIntegration(self.state,
                                           ninstance=self.ninstance,
                                           workdir=self.workdir,
                                           folder=self.folder,
                                           subfolder=self.subfolder,
                                           logfile='FreeEnergy.log')
        self.fe_init = self.ti.run().mean(axis=1)[0]
        return self.fe_init

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        """
        """
        if self.damp is None:
            self.damp = "$(100*dt)"

        if self.pdamp is None:
            self.pdamp = "$(1000*dt)"

        temp = self.t_start
        self.info_dynamics["temperature"] = temp
        if self.langevin:
            langevinseed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("thermostat", "Integrators")
        block("timestep", f"timestep {self.dt / 1000}")
        block("momenta", "velocity all create " +
              f"{temp} {langevinseed} dist gaussian")
        # If we are using Langevin, we want to remove the random part
        # of the forces
        if self.langevin:
            block("rmv_langevin", "fix ff all store/force")
            txt = f"fix f1 all langevin {temp} {temp} {self.damp} " + \
                  f"{langevinseed} zero yes"
            block("langevin", txt)
            if self.pressure is None:
                block("nve", "fix f2 all nve")
            if self.pressure is not None:
                # Fix center of mass for barostat
                block("xcm", "variable xcm equal xcm(all,x)")
                block("ycm", "variable ycm equal xcm(all,y)")
                block("zcm", "variable zcm equal xcm(all,z)")
                block("nph", f"fix f2 all nph iso {self.pressure*10000} " +
                             f"{self.pressure*10000} {self.pdamp} " +
                             "fixedpoint ${xcm} ${ycm} ${zcm}")
            block("compute temp without cm", "compute c1 all temp/com")
            block("fix modify cm", "fix_modify f1 temp c1")
            if self.pressure is not None:
                block("fix modify cm", "fix_modify f2 temp c1")
        else:
            if self.pressure is None:
                block("nvt", f"fix f1 all nvt temp {temp} {temp} {self.damp}")
            if self.pressure is not None:
                block("npt", f"fix f1 all npt temp {temp} {temp} " +
                             f"{self.damp} iso {self.pressure*10000} " +
                             f"{self.pressure*10000} {self.pdamp} " +
                             "fixedpoint ${xcm} ${ycm} ${zcm}")
            block("compute temp without cm", "compute c1 all temp/com")
            block("fix modify cm", "fix_modify f1 temp c1")
        return block

# ========================================================================== #
    def _get_block_traj(self, el):
        """
        """
        if self.trajfile:
            block = LammpsBlockInput("dump", "Dumping")
            txt = f"dump dum1 all custom {self.loginterval} {self.trajfile} "
            txt += "id type xu yu zu vx vy vz fx fy fz "
            txt += "element"
            block("dump", txt)
            block("dump_modify1", "dump_modify dum1 append yes")
            txt = "dump_modify dum1 element " + " ".join([p for p in el])
            block("dump_modify2", txt)
            return block
        else:
            pass

# ========================================================================== #
    def _get_neti(self):
        """
        """
        tstart = self.t_start
        tend = self.t_end

        blocks = []
        pair_style = self.pair_style.split()
        if len(self.pair_coeff) == 1:
            pair_coeff = self.pair_coeff[0].split()
            hybrid_pair_coeff = " ".join([*pair_coeff[:2],
                                          pair_style[0],
                                          *pair_coeff[2:]])
        else:
            hybrid_pair_coeff = []
            for pc in self.pair_coeff:
                pc_ = pc.split()
                hpc_ = " ".join([*pc_[:2], *pc_[2:]])
                hybrid_pair_coeff.append(hpc_)

        block0 = LammpsBlockInput("eq fwd", "Equilibration before fwd rs")
        block0("run eq fwd", f"run {self.nsteps_eq}")
        blocks.append(block0)

        block1 = LammpsBlockInput("fwd", "Forward Integration")
        block1("lambda fwd", "variable lambda equal " +
               f"1/(1+(elapsed/{self.nsteps})*({tend}/{tstart}-1))")
        if len(self.pair_coeff) == 1:
            block1("adapt pair_style", "fix f3 all adapt 1 pair " +
                   f"{self.pair_style} scale * * v_lambda")
        else:
            # pair_style comd compatible only with one zbl, To be fixed
            block1("scaling pair_style", "pair_style hybrid/scaled v_lamda " +
                   f"{pair_style[1]} {pair_style[2]} " +
                   f"{pair_style[3]} v_lambda {pair_style[4]}")
            block1("pair_coeff_1", "pair_coeff " + hybrid_pair_coeff[0])
            block1("pair_coeff_2", "pair_coeff " + hybrid_pair_coeff[1])
        block1("write fwd", "fix f4 all print 1 " +
               "\"$(pe/atoms) ${mypress} ${vol} ${lambda}\" screen no " +
               "append forward.dat title " +
               "\"# pe    pressure    vol    lambda\"\n")
        block1("run fwd", f"run {self.nsteps}")
        if len(self.pair_coeff) == 1:
            block1("unfix f3", "unfix f3 ")
        block1("unfix f4", "unfix f4 ")
        blocks.append(block1)

        block2 = LammpsBlockInput("eq bwd", "Equilibration before bwd rs")
        block2("run eq bwd", f"run {self.nsteps_eq}")
        blocks.append(block2)

        block3 = LammpsBlockInput("bwd", "Backward Integration")
        block3("lambda bwd", "variable lambda equal " +
               f"1/(1+(1-elapsed/{self.nsteps})*({tend}/{tstart}-1))")
        if len(self.pair_coeff) == 1:
            block3("adapt pair_style", "fix f3 all adapt 1 pair " +
                   f"{self.pair_style} scale * * v_lambda")
        else:
            # pair_style comd compatible only with one zbl, To be fixed
            block3("scaling pair_style", "pair_style hybrid/scaled v_lamda " +
                   f"{pair_style[1]} {pair_style[2]} " +
                   f"{pair_style[3]} v_lambda {pair_style[4]}")
            block3("pair_coeff_1", "pair_coeff " + hybrid_pair_coeff[0])
            block3("pair_coeff_2", "pair_coeff " + hybrid_pair_coeff[1])
        block3("write bwd", "fix f4 all print 1 " +
               "\"$(pe/atoms) ${mypress} ${vol} ${lambda}\" screen no " +
               "append backward.dat title " +
               "\"# pe    pressure    vol    lambda\"\n")
        blocks.append(block3)
        return blocks

# ========================================================================== #
    @Manager.exec_from_path
    def postprocess(self):
        """
        Compute the free energy from the simulation
        """
        natoms = len(self.atoms)
        if self.pressure is not None:
            p = self.pressure/160.21766208  # already divided by 100000
        else:
            p = 0.0
        # Get data
        v_f, fp, fvol, lambda_f = np.loadtxt("forward.dat", unpack=True)
        v_b, bp, bvol, lambda_b = np.loadtxt("backward.dat", unpack=True)

        v_f /= lambda_f
        v_b /= lambda_b

        # add pressure contribution
        fvol = fvol / natoms
        bvol = bvol / natoms
        v_f = v_f + p * fvol
        v_b = v_b + p * bvol

        # Integrate the forward and backward data
        int_f = cumtrapz(v_f, lambda_f, initial=0)
        int_b = cumtrapz(v_b[::-1], lambda_b[::-1], initial=0)
        # Compute the total work
        work = (int_f + int_b) / (2 * lambda_f)

        temperature = self.t_start / lambda_f
        free_energy = self.fe_init / lambda_f + \
            1.5 * kB * temperature * np.log(lambda_f) + work

        results = np.array([temperature, free_energy]).T
        header = "   T [K]    F [eV/at]"
        fmt = "%10.3f  %10.6f"
        np.savetxt("free_energy.dat", results, header=header, fmt=fmt)
        return

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        npt = False
        if self.pressure is not None:
            npt = True
        if self.damp is None:
            self.damp = 100 * self.dt

        msg = "Thermodynamic Integration using Reversible Scaling\n"
        msg += f"Starting temperature :          {self.t_start}\n"
        msg += f"Stopping temperature :          {self.t_end}\n"
        msg += f"Langevin damping :              {self.damp} fs\n"
        msg += f"Timestep :                      {self.dt} fs\n"
        msg += f"Number of steps :               {self.nsteps}\n"
        msg += f"Number of equilibration steps : {self.nsteps_eq}\n"
        if not npt:
            msg += "Constant volume simulation\n"
        else:
            if self.pdamp is None:
                pdamp = 1000 * self.dt
            msg += "Constant pressure simulation\n"
            msg += f"Pressure :                      {self.pressure} GPa\n"
            msg += f"Pressure damping :              {pdamp} fs\n"
        return msg
