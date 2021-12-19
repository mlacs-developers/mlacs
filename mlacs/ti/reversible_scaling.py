"""
"""
import os

import numpy as np
from scipy.integrate import cumtrapz

from ase.units import kB

from mlacs.utilities.miscellanous import get_elements_Z_and_masses
from mlacs.ti.thermostate import ThermoState




#========================================================================================================================#
#========================================================================================================================#
class ReversibleScalingState(ThermoState):
    """
    """
    def __init__(self,
                 atoms,
                 dt=1,
                 t_start=300,
                 t_end=1200,
                 damp=None,
                 pressure=None,
                 pdamp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 gjf=True,
                 rng=None,
                ):

        self.t_start  = t_start
        self.t_end    = t_end  
        self.damp     = damp
        self.pressure = pressure
        self.pdamp    = pdamp
        self.gjf      = gjf

        ThermoState.__init__(self,
                             atoms,
                             dt,
                             nsteps,
                             nsteps_eq,
                             rng,
                            )

        self.suffixdir = "ReversibleScaling_T{0}K_T{1}K".format(self.t_start, self.t_end)
        if self.pressure is None:
            self.suffixdir += "_NVT"
        else:
            self.suffixdir += "_{0}GPa".format(self.pressure)
        self.suffixdir += "/"


#========================================================================================================================#
    def run(self, wdir, pair_style, pair_coeff, mlip_style):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

        self.run_dynamics(wdir, pair_style, pair_coeff, mlip_style)

        with open(wdir + "MLMD.done", "w") as f:
            f.write("Done")


#========================================================================================================================#
    def write_lammps_input(self, wdir, pair_style, pair_coeff, mlip_style):
        """
        Write the LAMMPS input for the MLMD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(self.atoms)

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"

        pdamp = self.pdamp
        if pdamp is None:
            pdamp = "$(1000*dt)"
        

        input_string  = "# LAMMPS input file to run a MLMD simulation for thermodynamic integration\n"
        input_string += "units        metal\n"

        pbc = self.atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + wdir + "atoms.in"
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"

        input_string += "variable      nsteps equal {0}\n".format(self.nsteps)
        input_string += "variable      nstepseq equal {0}\n".format(self.nsteps_eq)
        input_string += "variable      tstart equal {0}\n".format(self.t_start)
        input_string += "variable      tend  equal {0}\n".format(self.t_end)

        input_string += "\n\n"

        input_string += "# Interactions\n"
        input_string += pair_style + "\n"
        input_string += pair_coeff
        if mlip_style == "snap" or mlip_style == "mliap":
            input_string += " " + " ".join(elem)
        input_string += "\n\n\n"

        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "\n"
        input_string += "velocity      all create ${{tstart}} {0} dist gaussian\n".format(self.rng.integers(99999))

        input_string += "\n\n"

        
        if self.pressure is not None:
            input_string += "# Fix center of mass for barostat\n"
            input_string += "variable      xcm equal xcm(all,x)\n"
            input_string += "variable      ycm equal xcm(all,y)\n"
            input_string += "variable      zcm equal xcm(all,z)\n"
            input_string += "\n\n"

        input_string += "# Integrators\n"
        if self.gjf:
            input_string += "fix    f1  all langevin ${{tstart}} ${{tstart}}  {0} {1}  zero yes gjf vhalf\n".format(damp, self.rng.integers(99999))
        else:
            input_string += "fix    f1  all langevin ${{tstart}} ${{tstart}}  {0}  {1} zero yes\n".format(damp, self.rng.integers(99999))
        if self.pressure is None:
            input_string += "fix    f2  all nve\n"
        else:
            input_string += "fix    f2  all nph iso {0} {0} {1} fixedpoint ${{xcm}} ${{ycm}} ${{zcm}}\n".format(self.pressure, pdamp)
        input_string += "\n"

        input_string += "# Fix center of mass\n"
        input_string += "compute       c1 all temp/com\n"
        input_string += "fix_modify    f1 temp c1\n"
        input_string += "fix_modify    f2 temp c1\n"

        input_string += "\n"

        """
        if self.logfile is not None:
            input_string += self.get_log_input()
        if self.trajfile is not None:
            input_string += self.get_traj_input()
        """

        input_string += "\n"

        input_string += "#######################\n"
        input_string += "# Forward integration\n"
        input_string += "#######################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        #input_string += "print        \"$(pe/atoms) 1\" file forward.dat\n"
        input_string += "variable     lambda equal 1/(1+(elapsed/${nsteps})*(${tend}/${tstart}-1))\n"
        input_string += "fix          f3 all adapt 1 pair snap scale * * v_lambda\n"
        input_string += "fix          f4 all print 1 \"$(pe/atoms) ${lambda}\" screen no " + \
                        "append forward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "unfix        f3\n"
        input_string += "unfix        f4\n"
        
        input_string += "\n\n"


        input_string += "#######################\n"
        input_string += "# Backward integration\n"
        input_string += "#######################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        #input_string += "print        \"$(pe/atoms) $(v_tstart/v_tend)\" file backward.dat\n"
        input_string += "variable     lambda equal 1/(1+(1-elapsed/${nsteps})*(${tend}/${tstart}-1))\n"
        input_string += "fix          f3 all adapt 1 pair snap scale * * v_lambda\n"
        input_string += "fix          f4 all print 1 \"$(pe/atoms) ${lambda}\" screen no " + \
                        "append backward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)


#========================================================================================================================#
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
        work  = (int_f + int_b) / (2 * lambda_f)

        temperature = self.t_start / lambda_f
        free_energy = 1.5 * kB * np.log(lambda_f) + work
        
        results = np.array([temperature, free_energy]).T
        header  = "   T [K]    F [eV/at]"
        fmt     = "%10.3f  %10.6f"
        np.savetxt(wdir + "free_energy.dat", results, header=header, fmt=fmt)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        npt = False
        if self.pressure is not None:
            npt = True
            pressure = self.pressure
        if self.damp is None:
            damp = 100 * self.dt
        
        msg  = "Thermodynamic Integration using Reversible Scaling\n"
        msg += "Starting temperature :          {0}\n".format(self.t_start)
        msg += "Stopping temperature :          {0}\n".format(self.t_end)
        msg += "Langevin damping :              {0} fs\n".format(damp)
        msg += "Timestep :                      {0} fs\n".format(self.dt)
        if not npt:
            msg += "Constant volume simulation\n"
        else:
            if self.pdamp is None:
                pdamp = 1000 * self.dt
            msg += "Constant pressure simulation\n"
            msg += "Pressure :                      {0} GPa\n".format(self.pressure)
            msg += "Pressure damping :              {0} fs\n".format(pdamp)
        return msg


#========================================================================================================================#
    def get_workdir(self):
        """
        """
        return self.suffixdir
