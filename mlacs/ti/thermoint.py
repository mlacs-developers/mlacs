"""
"""
import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read

from mlacs.utilities.thermolog import ThermoLog
from mlacs.ti.thermostate import ThermoState


#========================================================================================================================#
#========================================================================================================================#
class ThermodynamicIntegration:
    """
    """
    def __init__(self,
                 thermostate,
                 mlip_style="snap"
                ):

        self.mlip_style = mlip_style
        self.log        = ThermoLog()
 
        # Construct the working directory to run the thermodynamic integrations
        self.workdir = os.getcwd() + "/ThermoInt/"
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
 
        # Create list of thermostate
        if isinstance(thermostate, ThermoState):
            self.state = [thermostate]
        elif isinstance(thermostate, list):
            self.state = thermostate
        else:
            msg = "state should be a ThermoState object or a list of ThermoState objects"
            raise TypeError(msg)
        self.nstate = len(self.state)
 
        self.checkmlip()
 
 
        self.recap_state()


#========================================================================================================================#
    def run(self):
        """
        """
        msg  = "Running the simulation\n"
        self.log.logger_log.info(msg)
        for istate in range(self.nstate):
            stateworkdir = self.workdir + self.state[istate].get_workdir()
            msg  = "State {0}/{1} :\n".format(istate+1, self.nstate)
            msg += "Working directory : {0}\n".format(stateworkdir)
            self.log.logger_log.info(msg)
            self.state[istate].run(stateworkdir, self.pair_style, self.pair_coeff, self.mlip_style)
            msg  = "MLMD imulation done\n"
            msg += "Running post-process\n"
            self.log.logger_log.info(msg)
            self.state[istate].postprocess(stateworkdir)
            msg  = "Post-process Done\n"
            #msg += "Results can be found in {0}\n\n".format(stateworkdir)
            self.log.logger_log.info(msg)


#========================================================================================================================#
    def checkmlip(self):
        """
        """
        if self.mlip_style == "snap":
            msg  = "The MLIP is a SNAP potential\n"
            check = [False, False]
            modelfile = os.getcwd() + "/MLIP.snap.model"
            if os.path.isfile(modelfile):
                check[0] = True
                msg += "Model file {0}\n".format(modelfile)
            descriptorfile = os.getcwd() + "/MLIP.snap.descriptor"
            if os.path.isfile(modelfile):
                check[1] = True
                msg += "Descriptor file {0}\n".format(modelfile)
            self.pair_style = "pair_style     snap"
            self.pair_coeff = "pair_coeff     * * {0} {1}".format(modelfile, descriptorfile)
        elif self.mlip_style == "mliap":
            msg  = "The MLIP is a MLIAP potential\n"
            check = [False, False]
            modelfile = os.getcwd() + "MLIP.mliap.model"
            if os.path.isfile(modelfile):
                check[0] = True
                msg += "Model file {0}\n".format(modelfile)
            descriptorfile = os.getcwd() + "MLIP.mliap.descriptor"
            if os.path.isfile(modelfile):
                check[1] = True
                msg += "Descriptor file {0}\n".format(modelfile)
            self.pair_style = "pair_style     snap"
            self.pair_coeff = "pair_coeff     * * {0} {1}".format(modelfile, descriptorfile)

        msg += "\n"
        self.log.logger_log.info(msg)

        if not np.all(check):
            msg = "Can't find the MLIP files."
            raise InputError(msg)


#========================================================================================================================#
    def recap_state(self):
        """
        """
        msg = ""
        for istate in range(self.nstate):
            msg += "State {0}/{1} :\n".format(istate+1, self.nstate)
            msg += self.state[istate].log_recap_state()
            msg += "\n\n"
        self.log.logger_log.info(msg)
        pass


#========================================================================================================================#
