"""
// Copyright (C) 2022-2024 MLACS group (PR, GA, AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core import Manager
from ..utilities import save_cwd
from ..utilities.thermolog import ThermoLog
from .thermostate import ThermoState


# ========================================================================== #
# ========================================================================== #
class ThermodynamicIntegration(Manager):
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
                 logfile="ThermoInt.log",
                 workdir='ThermoInt',
                 **kwargs):

        Manager.__init__(self, workdir=workdir, **kwargs)

        self.workdir.mkdir(exist_ok=True, parents=True)

        self.log = ThermoLog(str(self.workdir / logfile))
        self.ninstance = ninstance
        self.logfile = logfile

        # Create list of thermostate
        if isinstance(thermostate, ThermoState):
            thermostate.workdir = self.workdir
            self.state = [thermostate]
        elif isinstance(thermostate, list):
            self.state = thermostate
            for st in self.state:
                st.workdir = self.workdir
        else:
            msg = "state should be a ThermoState object or " + \
                  "a list of ThermoState objects"
            raise TypeError(msg)
        self.nstate = len(self.state)
        self.recap_state()

# ========================================================================== #
    @Manager.exec_from_path
    def run(self):
        """
        Launch the simulation
        """
        tasks = (self.ninstance * self.nstate)
        with save_cwd(), ThreadPoolExecutor(max_workers=tasks) as executor:
            for istate in range(self.nstate):
                for i in range(self.ninstance):
                    if self.ninstance > 1:
                        stateworkdir = os.path.join(self.workdir,
                                                    self.state[istate].get_workdir(),
                                                    f"for_back_{i+1}/")
                    elif self.ninstance == 1:
                        stateworkdir = os.path.join(self.workdir,
                                                    self.state[istate].get_workdir())
                    os.makedirs(stateworkdir, exist_ok=True)
                    future = executor.submit(self._run_one_state, istate, i)
                    msg = f"State {istate + 1}/{self.nstate} instance_{i} launched\n"
                    msg += f"Working directory for this instance: \n{stateworkdir}\n"
                    self.log.logger_log.info(msg)

                    future.result()
            # executor.shutdown(wait=True)

        if self.ninstance > 1:
            for istate in range(self.nstate):
                self.error(istate)

# ========================================================================== #
    @Manager.exec_from_subdir
    def _run_one_state(self, istate, i):
        """
        Run the simulation for one state
        """
        ii = istate + 1
        if self.ninstance > 1:
            self.state[istate].subfolder = f"for_back_{i+1}"
            self.state[istate].run()
            msg = f"State {ii} instance_{i+1} : Molecular Dynamics Done\n"
            msg += "Starting post-process\n"
            self.log.logger_log.info(msg)
            msg, _ = self.state[istate].postprocess()
            self.log.logger_log.info(msg)
            msg = '=' * 59 + "\n"
            msg += f"State {ii} instance_{i+1}: Post-process Done\n"
            msg += "=" * 59 + "\n"
            self.log.logger_log.info(msg)
        elif self.ninstance == 1:
            self.state[istate].subfolder = ''
            self.state[istate].run()
            msg = f"State {ii}: Molecular Dynamics Done\n"
            msg += "Starting post-process\n"
            self.log.logger_log.info(msg)
            msg, _ = self.state[istate].postprocess()
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
        fe = []
        for i in range(self.ninstance):
            self.state[istate].subfolder = f"for_back_{i+1}"
            _, tmp_fe = self.state[istate].postprocess()
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
        for istate in range(self.nstate):
            stateworkdir = self.state[istate].path
        return stateworkdir
