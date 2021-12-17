"""
"""
import numpy as np

from mlacs.ti.thermostate import ThermoState




#========================================================================================================================#
#========================================================================================================================#
class ReversibleScalingState(ThermoState):
    """
    """
    def __init__(self,
                 atoms,
                 t_start=300,
                 t_end=1200,
                 ntemp=4,
                 temps=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 fixcm=True,
                 rng=None,
                 workdir=None
                )
        
        Thermostate.__init__(self,
                             atoms,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             rng,
                             workdir
                            )


#========================================================================================================================#
    def write_lammps_input(self, pair_style, pair_coeff):
        """
        Write the LAMMPS input for the MLMD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(self.atoms)

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)$"

        input_string  = "# LAMMPS input file to run a MLMD simulation for thermodynamic integration\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + self.atomsfname
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"


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


        input_string += "#######################\n"
        input_string += "# Formard integration\n"
        input_string += "#######################\n"
        input_string += "# Equilibration\n"
        input_string += "run    {0}\n".format(self.nsteps_eq)
        input_string += "print   \"$(pe/atoms) $(v_T/1600)\" file forward.dat\n"
        input_string += "variable     lambda equal 1/(1+(elapsed/${t})*(1600/$T-1))\n"
        input_string += "fix    f3 all adapt 1 pair mliap scale * * v_lambda\n"
        input_string += "fix    f4 all print 1 \"$(pe/atoms) ${lambda}\" screen no " + \
                        "append formard.dat title \"# pe    lambda\"\n"
        input_string += "run    {0}\n".format(self.nsteps)
        input_string += "unfix  f3\n"
        input_string += "unfix  f4\n"
        
        input_string += "\n\n"


        input_string += "#######################\n"
        input_string += "# Backward integration\n"
        input_string += "#######################\n"
        input_string += "# Equilibration\n"
        input_string += "run    {0}\n".format(self.nsteps_eq)
        input_string += "print   \"$(pe/atoms) $(v_T/1600)\" file backward.dat\n"
        input_string += "variable     lambda equal 1/(1-(elapsed/${t})*(1600/$T-1))\n"
        input_string += "fix    f3 all adapt 1 pair mliap scale * * v_lambda\n"
        input_string += "fix    f4 all print 1 \"$pe/atoms) ${lambda}\" screen no " + \
                        "append backward.dat title \"# pe    lambda\"\n"
        input_string += "run    {0}\n".format(self.nsteps)

