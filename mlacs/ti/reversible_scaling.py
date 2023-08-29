"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import numpy as np
from scipy.integrate import cumtrapz
from ase.units import kB

from ..utilities.miscellanous import get_elements_Z_and_masses
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
    t_start: :class:`float` (optional)
        Initial temperature of the simulation, in Kelvin. Default ``300``.
    t_end: :class:`float` (optional)
        Final temperature of the simulation, in Kelvin. Default ``1200``.
    fe_init: :class:`float` (optional)
        Free energy of the initial temperature, in eV/at. Default ``None``.
    ninstance: :class:`int` (optional)
        If Free energy calculation has to be done before temperature sweep
        Settles the number of forward abackward runs. Default ``1``.
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
    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    suffixdir: :class:`str`
        Suffix for the directory in which the computation will be run.
        If ``None``, a directory ``\"Solid_TXK\"`` is created,
        where X is the temperature. Default ``None``.
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
                 t_start=300,
                 t_end=1200,
                 fe_init=None,
                 phase=None,
                 ninstance=1,
                 dt=1.5,
                 damp=None,
                 pressure=None,
                 pdamp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 gjf=True,
                 rng=None,
                 suffixdir=None,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50,
                 trajinterval=50):

        self.t_start = t_start
        self.t_end = t_end
        self.fe_init = fe_init
        self.ninstance = ninstance
        self.damp = damp
        self.pressure = pressure
        self.pdamp = pdamp
        self.gjf = gjf

        # Free energy calculation before sweep
        if self.fe_init is None:
            if phase=='solid':
                self.state = EinsteinSolidState(atoms,          
                                                pair_style,     
                                                pair_coeff,     
                                                t_start,    
                                                fcorr1=None,    
                                                fcorr2=None,    
                                                k=None,         
                                                dt=dt,           
                                                damp=None,      
                                                nsteps=10000,   
                                                nsteps_eq=5000, 
                                                nsteps_msd=25000,
                                                rng=None,       
                                                suffixdir=None, 
                                                logfile=True,   
                                                trajfile=True,  
                                                interval=500,   
                                                loginterval=50, 
                                                trajinterval=50)
            elif phase=='liquid':
                self.state = UFLiquidState(atoms,         
                                           pair_style,    
                                           pair_coeff,    
                                           t_start,   
                                           fcorr1=None,   
                                           fcorr2=None,   
                                           p=50,          
                                           sigma=2.0,     
                                           dt=dt,          
                                           damp=None,     
                                           nsteps=10000,  
                                           nsteps_eq=5000,
                                           rng=None,      
                                           suffixdir=None,
                                           logfile=True,  
                                           trajfile=True, 
                                           interval=500,  
                                           loginterval=50,
                                           trajinterval=50)
            self.ti=ThermodynamicIntegration(self.state,
                                             ninstance,
                                             logfile='FreeEnergy.log')
            self.ti.run()
            # Get Fe
            if self.ninstance==1:
                _, self.fe_init = self.state.postprocess(self.ti.get_fedir())
            elif self.ninstance > 1:
                tmp = []
                for i in range(self.ninstance):
                    _, tmp_fe_init = self.state.postprocess(self.ti.get_fedir() \
                                                            + f"for_back_{i+1}/")
                    tmp.append(tmp_fe_init)
                self.fe_init = np.mean(tmp)
        # reversible scaling
        ThermoState.__init__(self,
                             atoms,
                             pair_style,
                             pair_coeff,
                             dt,
                             nsteps,
                             nsteps_eq,
                             rng,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval)

        self.suffixdir = f"ReversibleScaling_T{self.t_start}K_T{self.t_end}K"
        if self.pressure is None:
            self.suffixdir += "_NVT"
        else:
            self.suffixdir += "_{0}GPa".format(self.pressure)
        self.suffixdir += "/"
        if suffixdir is not None:
            self.suffixdir = suffixdir
        if self.suffixdir[-1] != "/":
            self.suffixdir += "/"
 
# ========================================================================== #
    def run(self, wdir):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

            self.run_dynamics(wdir)

        with open(wdir + "MLMD.done", "w") as f:
            f.write("Done")

# ========================================================================== #
    def write_lammps_input(self, wdir):
        """
        Write the LAMMPS input for the MLMD simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(self.atoms)

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"

        pdamp = self.pdamp
        if pdamp is None:
            pdamp = "$(1000*dt)"

        input_string = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += f"variable      nsteps equal {self.nsteps}\n"
        input_string += f"variable      nstepseq equal {self.nsteps_eq}\n"
        input_string += f"variable      tstart equal {self.t_start}\n"
        input_string += f"variable      tend  equal {self.t_end}\n"
        input_string += f"timestep      {self.dt/ 1000}\n"
        input_string += "#####################################\n"
        input_string += "\n\n"

        input_string += self.get_interaction_input()

        input_string += "\n\n"

        input_string += "#####################################\n"
        input_string += "#          Integrators\n"
        input_string += "#####################################\n"
        input_string += "velocity      all create ${tstart} " + \
            f"{self.rng.integers(99999)} dist gaussian\n"
        if self.pressure is not None:
            input_string += "# Fix center of mass for barostat\n"
            input_string += "variable      xcm equal xcm(all,x)\n"
            input_string += "variable      ycm equal xcm(all,y)\n"
            input_string += "variable      zcm equal xcm(all,z)\n"
        if self.pressure is None:
            input_string += "fix           f2  all nve\n"
        else:
            input_string += "fix           f2  all nph iso " + \
                f"{self.pressure} {self.pressure} {pdamp} " + \
                "fixedpoint ${xcm} ${ycm} ${zcm}\n"
        input_string += "fix           f1  all langevin ${tstart} " + \
            f"${{tstart}}  {damp}  {self.rng.integers(99999)} zero yes\n"
        input_string += "\n"
        input_string += "# Fix center of mass\n"
        input_string += "compute       c1 all temp/com\n"
        input_string += "fix_modify    f1 temp c1\n"
        if self.pressure is not None:
            input_string += "fix_modify    f2 temp c1\n"
        input_string += "#####################################\n"

        input_string += "\n\n"

        if self.logfile is not None:
            input_string += self.get_log_input()
        if self.trajfile is not None:
            input_string += self.get_traj_input()

        input_string += "\n"

        input_string += "#####################################\n"
        input_string += "# Forward integration\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "variable     lambda equal " + \
            "1/(1+(elapsed/${nsteps})*(${tend}/${tstart}-1))\n"
        input_string += "fix          f3 all adapt 1 pair " + \
            f"{self.pair_style} scale * * v_lambda\n"
        input_string += "fix          f4 all print 1 " + \
            "\"$(pe/atoms) ${lambda}\" screen no " + \
            "append forward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "unfix        f3\n"
        input_string += "unfix        f4\n"
        input_string += "#####################################\n"

        input_string += "\n\n"

        input_string += "#####################################\n"
        input_string += "# Backward integration\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "variable     lambda equal " + \
            "1/(1+(1-elapsed/${nsteps})*(${tend}/${tstart}-1))\n"
        input_string += "fix          f3 all adapt 1 pair " + \
            f"{self.pair_style} scale * * v_lambda\n"
        input_string += "fix          f4 all print 1 " + \
            "\"$(pe/atoms) ${lambda}\" screen no " + \
            "append backward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "#####################################\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)

# ========================================================================== #
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """

        # Get data
        v_f, lambda_f = np.loadtxt(wdir+"forward.dat", unpack=True)
        v_b, lambda_b = np.loadtxt(wdir+"backward.dat", unpack=True)

        v_f /= lambda_f
        v_b /= lambda_b

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
        np.savetxt(wdir + "free_energy.dat", results, header=header, fmt=fmt)
        return ""

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        npt = False
        if self.pressure is not None:
            npt = True
        if self.damp is None:
            damp = 100 * self.dt

        msg = "Thermodynamic Integration using Reversible Scaling\n"
        msg += f"Starting temperature :          {self.t_start}\n"
        msg += f"Stopping temperature :          {self.t_end}\n"
        msg += f"Langevin damping :              {damp} fs\n"
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
