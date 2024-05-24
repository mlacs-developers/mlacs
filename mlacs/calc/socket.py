"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

from ase.calculators.socketio import SocketIOCalculator

from mlacs.calc.calc_manager import CalcManager


# =========================================================================== #
# =========================================================================== #
class SocketCalcManager(CalcManager):
    """
    Class for managing the true potential through the SocketIO calculator

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
        CalcManager.__init__(self, calc=calc, magmoms=magmoms **kwargs) 

        # This will launch the server
        SocketIOCalculator(unixsocket=unixsocket, port=port, log=socketlog)

# =========================================================================== #
    def compute_true_potential(self, atoms):
        """
        """
        atoms.set_initial_magnetic_moments(self.magmoms)
