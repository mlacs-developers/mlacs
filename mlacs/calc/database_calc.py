"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import ase
from .calc_manager import CalcManager


# ========================================================================== #
# ========================================================================== #
class DatabaseCalc(CalcManager):
    """
    Calculators that sequentially reads a previously calculated traj files.
    Normal utilisator want to set OtfMlacs.nstep to len(traj)
    Can be used with restart to append trajfile to the current traj.

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object

    trajfile :class:`str` or :class:`pathlib.Path`
        The trajectory file from which DatabaseCalc will read

    trainfile :class:`str`, :class:`pathlib.Path`,
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
                 magmoms=None):
        CalcManager.__init__(self, "dummy", magmoms)
        self.traj = ase.io.read(trajfile, index=":")
        self.training = ase.io.read(trainfile, index=":")
        self.current_conf = 0

# ========================================================================== #
    def compute_true_potential(self,
                               mlip_confs,
                               state=None,
                               step=None):
        """
        1. Create a copy of the next atoms in traj as true_atoms
        2. Modify mlip_atoms positions to match what we have in the traj
        3. Change the Parent MLIP that generated true_confs
        """
        assert len(mlip_confs) + self.current_conf <= len(self.traj), \
            "You cannot do more step than there is in the Trajectory file" +\
            f"\nNumber of conf in the given Trajectory: {len(self.traj)}"

        true_confs = []
        for mlip_conf, s in zip(mlip_confs, state):
            if s == "Training":
                true_confs.append(self.training.pop())
                continue

            true_confs.append(self.traj[self.current_conf])
            mlip_conf.set_positions(true_confs[-1].get_positions())

            if 'parent_mlip' in mlip_conf.info:
                true_confs[-1].info['parent_mlip'] = \
                        mlip_conf.info['parent_mlip']
            self.current_conf += 1
        return true_confs