"""
// Copyright (C) 2022-2024 MLACS group (PR, GA, AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
import copy


import numpy as np

from ..core import Manager
from ..utilities import save_cwd
from ..utilities.log import ThermoLog
from .thermostate import ThermoState


# ========================================================================== #
# ========================================================================== #
class ThermodynamicIntegration(Manager):
    """
    Class to handle a series of thermodynamic integration on sampled states

    Parameters
    ----------
    thermostate: :class:`thermostate` or :class:`list` of :class:`thermostate`
        State(s) for which the thermodynamic integration should be performed
    ninstance: : class:`int`
        Numer of forward and backward to be performed per state, default 1
    logfile: :class:`str` (optional)
        Name of the logfile. Default ``\"ThermoInt.log\"``
    workdir: :class:`str`(optional)
        Name of the root folder in which the simulations will be performed.
        Default `ThermoInt`
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

        Returns
        -------

        fe: :class:`np.ndarray`
            The free energy of each instance of each state, in an array of
            shape (nstate, ninstance). In eV/at
        """
        values = np.zeros((self.nstate, self.ninstance))
        with save_cwd():
            for istate in range(self.nstate):
                for i in range(self.ninstance):
                    worker = copy.deepcopy(self.state[istate])
                    worker.folder += f"/for_back{i+1}"
                    worker.rng = np.random.default_rng()
                    worker.run()
                    msg, fe = worker.postprocess()
                    values[istate, i] = fe
                    self.log.logger_log.info(msg)

        for istate in range(self.nstate):
            femean = values[istate].mean()
            ferr = values[istate].std()
            msg = f"Free Energy mean and error for state {istate+1}:\n"
            msg += f"- Mean: {femean:10.6f}\n"
            msg += f"- Error: {ferr:10.6f}\n"
            self.log.logger_log.info(msg)

        self.log.write_footer()
        return values

# ========================================================================== #
    def recap_state(self):
        """
        """
        msg = f"Total number of state : {self.nstate}, "
        msg += f"each state will be run {self.ninstance} times\n"
        for istate in range(self.nstate):
            msg += f"State {istate+1}/{self.nstate} :\n"
            msg += self.state[istate].log_recap_state()
            msg += "\n\n"
        self.log.logger_log.info(msg)
