"""
// Copyright (C) 2022-2024 MLACS group (AC, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path
import ase
from .calc_manager import CalcManager


# ========================================================================== #
# ========================================================================== #
class DatabaseCalc(CalcManager):
    """
    Calculators that sequentially reads a previously calculated traj files.
    Normal utilisator want to set OtfMlacs.nstep to len(traj).
    Can be used with restart to append trajfile to the current traj.

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object

    trajfile: :class:`str` or :class:`pathlib.Path`
        The trajectory file from which DatabaseCalc will read

    trainfile: :class:`str`, :class:`pathlib.Path`,
        The training.traj file, configuration used for fitting but
        not counted for thermodynamic properties

    magmoms: :class:`np.ndarray` (optional)
        An array for the initial magnetic moments for each computation
        If ``None``, no initial magnetization. (Non magnetic calculation)
        Default ``None``.
    """
    def __init__(self,
                 trajfile,
                 trainfile,
                 magmoms=None,
                 **kwargs):
        CalcManager.__init__(self, "dummy", magmoms, **kwargs)
        if isinstance(trajfile, str) or isinstance(trajfile, Path):
            self.traj = [ase.io.read(trajfile, index=":")]
        else:
            self.traj = [ase.io.read(t, index=":") for t in trajfile]
        self.training = ase.io.read(trainfile, index=":")
        self.current_conf = 0

# ========================================================================== #
    def compute_true_potential(self,
                               mlip_confs,
                               state=None,
                               step=None):
        """
        Return the energy of given configurations contained in the Trajectory
        file. This is a way to replay a previously done MLACS simulations.
        """
        # 1. Create a copy of the next atoms in traj as true_atoms
        # 2. Modify mlip_atoms positions to match what we have in the traj
        # 3. Change the Parent MLIP that generated true_confs
        length_trajs = sum([len(t) for t in self.traj])
        assert len(mlip_confs) + self.current_conf <= length_trajs, \
            "You cannot do more step than there is in the Trajectory file" +\
            "\nNumber of conf in the given Trajectory. If multiple state" +\
            " you need to give a list of Traj file."

        true_confs = []
        i = 0
        for mlip_conf, s in zip(mlip_confs, state):
            if s == "Training":
                true_confs.append(self.training.pop())
                continue

            true_confs.append(self.traj[i][self.current_conf])
            mlip_conf.set_positions(true_confs[-1].get_positions())
            true_confs[-1].info = mlip_conf.info
            i += 1
        if state[0] != "Training":
            self.current_conf += 1

        return true_confs
