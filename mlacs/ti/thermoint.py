"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
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
    Class to handle a series of thermodynamic integration

    Parameters
    ----------
    thermostate: :class:`thermostate` or :class:`list` of :class:`thermostate`
        State for which the thermodynamic integration should be performed
    logfile: :class:`str` (optional)
        Name of the logfile. Default ``\"ThermoInt.log\"``
    """
    def __init__(self,
                 thermostate,
                 logfile=None
                ):

        self.log = ThermoLog(logfile)
 
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
 
        self.recap_state()


#========================================================================================================================#
    def run(self):
        """
        Launch the simulation
        """
        msg  = "Running the simulation\n"
        self.log.logger_log.info(msg)
        for istate in range(self.nstate):
            stateworkdir = self.workdir + self.state[istate].get_workdir()
            msg  = '===============================================================\n' 
            msg += "State {0}/{1} :\n".format(istate+1, self.nstate)
            msg += '===============================================================\n' 
            msg += "Working directory : {0}\n".format(stateworkdir)
            self.log.logger_log.info(msg)
            self.state[istate].run(stateworkdir)
            msg  = "MLMD simulation done\n"
            msg += "Running post-process"
            self.log.logger_log.info(msg)
            msg  = "Post-process Done\n"
            self.log.logger_log.info(msg)
            msg  = self.state[istate].postprocess(stateworkdir)
            self.log.logger_log.info(msg)
            self.log.logger_log.info("\n")


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
