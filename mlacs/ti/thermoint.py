"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os

import numpy as np
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
    wdir: :class:`str`
    ninstance: : class:`int`
        Numer of forward and backward to be performed, default 1
    logfile: :class:`str` (optional)
        Name of the logfile. Default ``\"ThermoInt.log\"``
    """
    def __init__(self,
                 thermostate,
                 ninstance=1,
                 wdir=None,
                 logfile=None):

        self.log = ThermoLog(logfile)
        self.ninstance = ninstance
        self.logfile = logfile

        # Construct the working directory to run the thermodynamic integrations
        if wdir is None:
            self.workdir = os.path.join(os.getcwd(), "ThermoInt/")
        elif wdir is not None:
            self.workdir = os.path.join(wdir, "ThermoInt/")
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        # Create list of thermostate
        if isinstance(thermostate, ThermoState):
            self.state = [thermostate]
            # Create ninstance state
        #    if self.ninstance > 1:
        #        state_replica = self.state
        #        self.state.extend(state_replica * (self.ninstance-1))
        elif isinstance(thermostate, list):
            self.state = thermostate
        else:
            msg = "state should be a ThermoState object or " + \
                  "a list of ThermoState objects"
            raise TypeError(msg)
        self.nstate = len(self.state)
        self.recap_state()
#        self.log.logger_log.removeHandler(
#            logging.FileHandler(self.logfile, 'a'))
#        logging.FileHandler(self.logfile, 'a').close()

# ========================================================================== #
    def run(self):
        """
        Launch the simulation
        """
        tasks = (self.ninstance * self.nstate)
        with ThreadPoolExecutor(max_workers=tasks) as executor:
            for istate in range(self.nstate):
                if self.ninstance > 1:
                    for i in range(self.ninstance):
                        executor.submit(self._run_one_state, istate, i)
                        msg = f"State {istate+1}/{self.nstate} " + \
                              f"instance_{i+1} launched\n"
                        stateworkdir = os.path.join(self.workdir,
                            self.state[istate].get_workdir(),
                            f"for_back_{i+1}/")
                        msg += "Working directory for this instance " + \
                               f"of state : \n{stateworkdir}\n"
                        self.log.logger_log.info(msg)
                elif self.ninstance == 1:
                    executor.submit(self._run_one_state, istate, i=1)
                    msg = f"State {istate+1}/{self.nstate} launched\n"
                    stateworkdir = os.path.join(self.workdir,
                        self.state[istate].get_workdir())
                    msg += "Working directory for this state " + \
                           f": \n{stateworkdir}\n"
                    self.log.logger_log.info(msg)

        if self.ninstance > 1:
            for istate in range(self.nstate):
                self.error(istate)

# ========================================================================== #
    def _run_one_state(self, istate, i):
        """
        Run the simulation for one state
        """
        ii = istate + 1
        if self.ninstance > 1:
            stateworkdir = os.path.join(self.workdir,
                self.state[istate].get_workdir(),
                f"for_back_{i+1}/")
            self.state[istate].run(stateworkdir)
            msg = f"State {ii} instance_{i+1} : Molecular Dynamics Done\n"
            msg += "Starting post-process\n"
            self.log.logger_log.info(msg)
            msg, _ = self.state[istate].postprocess(stateworkdir)
            self.log.logger_log.info(msg)
            msg = '=' * 59 + "\n"
            msg += f"State {ii} instance_{i+1}: Post-process Done\n"
            msg += "=" * 59 + "\n"
            self.log.logger_log.info(msg)
        elif self.ninstance == 1:
            stateworkdir = os.path.join(self.workdir, self.state[istate].get_workdir())
            self.state[istate].run(stateworkdir)
            msg = f"State {ii}: Molecular Dynamics Done\n"
            msg += "Starting post-process\n"
            self.log.logger_log.info(msg)
            msg, _ = self.state[istate].postprocess(stateworkdir)
            self.log.logger_log.info(msg)
            msg = "=" * 59 + "\n"
            msg += f"State {istate+1}: Post-process Done\n"
            msg += "=" * 59 + "\n"
            self.log.logger_log.info(msg)

# ========================================================================== #
    def recap_state(self):
        """
        """
        msg = f"Total number of state : {self.nstate}. "
        msg += "One state is equivalent to ninstance f/b\n"
        for istate in range(self.nstate):
            msg += f"State {istate+1}/{self.nstate} :\n"
            msg += self.state[istate].log_recap_state()
            msg += "\n\n"
        self.log.logger_log.info(msg)

# ========================================================================== #
    def error(self, istate):
        """
        Error and average in free energy instances for one state
        Computed if ninstance > 1
        """
        stateworkdir = os.path.join(self.workdir, self.state[istate].get_workdir())
        fe = []
        for i in range(self.ninstance):
            # tmp_fe = np.loadtxt(stateworkdir +
            #                     f"for_back_{i+1}/" +
            #                     "free_energy.dat")
            # fe.append(tmp_fe[1]+tmp_fe[len(tmp_fe)-1])
            _, tmp_fe = self.state[istate].postprocess(stateworkdir +
                                                       f"for_back_{i+1}/")
            fe.append(tmp_fe)
        ferr = np.std(fe, axis=0)
        femean = np.mean(fe, axis=0)
        msg = f"Free Energy mean and error for state {istate+1}:\n"
        msg += f"- Mean: {femean:10.6f}\n"
        msg += f"- Error: {ferr:10.6f}\n"
        self.log.logger_log.info(msg)

# ========================================================================== #
    def get_fedir(self):
        """
        Get the directory where free energy is.
        Useful to get free energy for property convergence during sampling
        """
        # if self.ninstance > 1:
        #     for i in range(self.ninstance):
        #         stateworkdir = self.workdir + \
        #                        self.state.get_workdir() + \
        #                        f"for_back_{i+1}/"
        # elif self.ninstance == 1:
        for istate in range(self.nstate):
            stateworkdir = os.path.join(self.workdir,
                           self.state[istate].get_workdir())
        return stateworkdir
