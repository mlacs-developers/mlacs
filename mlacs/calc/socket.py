"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""


from ase.calculators.socketio import SocketIOCalculator

from mlacs.calc.calc_manager import CalcManager


# =========================================================================== #
# =========================================================================== #
class SocketCalcManager(CalcManager):
    """
    Class for managing the true potential through the SocketIO calculator
    Work in progress.

    Parameters
    ----------
    """
    def __init__(self,
                 calc=None,
                 magmoms=None,
                 socketlog=None,
                 unixsocket=None,
                 port=None,
                 **kwargs):
        CalcManager.__init__(self, calc=calc, magmoms=magmoms, **kwargs)

        # This will launch the server
        SocketIOCalculator(unixsocket=unixsocket, port=port, log=socketlog)

# =========================================================================== #
    def compute_true_potential(self, atoms):
        """
        """
        atoms.set_initial_magnetic_moments(self.magmoms)
