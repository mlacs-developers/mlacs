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
    Class to handle a series of thermodynamic integration on sampled states

    Parameters
    ----------
    thermostate: :class:`thermostate` or :class:`list` of :class:`thermostate`
        State for which the thermodynamic integration should be performed
    ninstance: : class:`int` 
        Numer of forward and backward to be performed, default 1
    logfile: :class:`str` (optional)
        Name of the logfile. Default ``\"ThermoInt.log\"``
    """
    def __init__(self,
                 thermostate,
                 ninstance=1,
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
            # Create ninstance state
            if self.ninstance > 1:
                state_replica = self.state
                self.state.extend(state_replica * (self.ninstance-1))
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
                msg = f"State {istate+1}/{self.nstate} launched\n"
                stateworkdir = self.workdir + self.state[istate].get_workdir()
                msg += f"Working directory for this state : \n{stateworkdir}\n"
                self.log.logger_log.info(msg)

# ========================================================================== #
    def _run_one_state(self, istate):
        """
        Run the simulation for one state
        """
        if self.ninstance > 1:
            for i in range(self.ninstance):
                stateworkdir = self.workdir + self.state[istate].get_workdir() + f"for_back_{i+1}/"
                self.state[istate].run(stateworkdir)
                msg = f"State {istate+1} instance_{i+1} : Molecular Dynamics Done\n"
                msg += "Starting post-process\n"
                self.log.logger_log.info(msg)
                msg = '============================================================\n'
                msg += f"State {istate+1} instance_{i+1}: Post-process Done\n"
                msg += self.state[istate].postprocess(stateworkdir)
                msg += '============================================================\n'
                self.log.logger_log.info(msg)
            self.error(istate)
        elif self.ninstance == 1: 
            stateworkdir = self.workdir + self.state[istate].get_workdir()
            self.state[istate].run(stateworkdir)
            msg = f"State {istate+1}: Molecular Dynamics Done\n"
            msg += "Starting post-process\n"
            self.log.logger_log.info(msg)
            msg = '============================================================\n'
            msg += f"State {istate+1}: Post-process Done\n"
            msg += self.state[istate].postprocess(stateworkdir)
            msg += '============================================================\n'
            self.log.logger_log.info(msg)

# ========================================================================== #
    def recap_state(self):
        """
        """
        msg = "Total number of state : {0}. One state is equivalent to ninstance f/b\n".format(self.nstate)
        for istate in range(self.nstate):
            msg += "State {0}/{1} :\n".format(istate+1, self.nstate)
            msg += self.state[istate].log_recap_state()
            msg += "\n\n"
        self.log.logger_log.info(msg)

# ========================================================================== #
    def error(self, istate):
        """
        Error and average in free energy instances for one state
        Computed is ninstance > 1
        """
        import numpy as np
        
        stateworkdir = self.workdir + self.state[istate].get_workdir()
        fe = []
        for j in range(self.ninstance):
            tmp_fe = np.loadtxt(stateworkdir + f"for_back_{j+1}/" + f"free_energy.dat")
            fe.append(tmp_fe[1])
        ferr = np.std(fe, axis = 0)
        femean = np.mean(fe, axis = 0)
        msg = f"Free Energy mean and error for state {istate+1}:\n"
        msg += f"- Mean: {femean:10.6f}\n"
        msg += f"- Error: {ferr:10.6f}\n"
        self.log.logger_log.info(msg)

        

