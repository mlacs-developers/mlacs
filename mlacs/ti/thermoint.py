"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os

from ..utilities.thermolog import ThermoLog
from .thermostate import ThermoState
from concurrent.futures import ThreadPoolExecutor


# ========================================================================== #
# ========================================================================== #
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
                 ninstance=10,
                 logfile=None):

        self.log = ThermoLog(logfile)
        self.ninstance = ninstance

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
            msg = "state should be a ThermoState object or " + \
                  "a list of ThermoState objects"
            raise TypeError(msg)
        self.nstate = len(self.state)
        self.recap_state()

# ========================================================================== #
    def run(self):
        """
        Launch the simulation
        """
        with ThreadPoolExecutor(max_workers=self.ninstance) as executor:
            for istate in range(self.nstate):
                executor.submit(self._run_one_state, istate)
                msg = f"State {istate+1}/{self.nstate} launched"
                stateworkdir = self.workdir + self.state[istate].get_workdir()
                msg += f"Working directory for this state : \n{stateworkdir}\n"

# ========================================================================== #
    def _run_one_state(self, istate):
        """
        """
        stateworkdir = self.workdir + self.state[istate].get_workdir()
        self.state[istate].run(stateworkdir)
        msg = f"State {istate+1} : Molecular Dynamics Done\n"
        msg += "Starting post-process\n"
        self.log.logger_log.info(msg)
        msg = '============================================================\n'
        msg += f"State {istate+1} : Post-process Done\n"
        msg += self.state[istate].postprocess(stateworkdir)
        msg += '============================================================\n'
        self.log.logger_log.info(msg)

# ========================================================================== #
    def recap_state(self):
        """
        """
        msg = ""
        for istate in range(self.nstate):
            msg += "State {0}/{1} :\n".format(istate+1, self.nstate)
            msg += self.state[istate].log_recap_state()
            msg += "\n\n"
        self.log.logger_log.info(msg)
