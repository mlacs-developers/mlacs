"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path
import logging
import numpy as np
from ase.units import GPa
try:
    # With the annoying mandatory warning from mbar, we have to initialize
    # the log here otherwise the log doesn't work
    # I have to see how to handle this in a better way.
    # This might be an indication of needing to redo the logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("pymbar")
    logger.setLevel(logging.ERROR)
    from pymbar import MBAR
    logger.setLevel(logging.INFO)
    ispymbar = True
except ModuleNotFoundError:
    ispymbar = False

from ase.atoms import Atoms
from ase.units import kB
from ..core.manager import Manager
from .weighting_policy import WeightingPolicy


default_parameters = {"solver": "L-BFGS-B",
                      "scale": 1.0,
                      "start": 2,
                      }


# ========================================================================== #
# ========================================================================== #
class MbarManager(WeightingPolicy):
    """
    Computation of weight according to the multistate Bennett acceptance
    ratio (MBAR) method for the analysis of equilibrium samples from multiple
    arbitrary thermodynamic states.

    Parameters
    ----------
    solver: :class:`str`
        Define type of solver for pymbar
        Default L-BFGS-B

    scale: :class:`float`
        Imposes weights for the new configurations.
        Default 1.0

    start: :class:`int`
        Step to start weight computation.
        At least 2 since you need two potentials to compare them.
        Default 2

    database: :class:`ase.Trajectory`
        Initial database (optional)
        Default :class:`None`

    weight: :class:`list` or :class:`str`
        If you use an initial database, it needs weight.
        Can a list or an np.array of values or a file.
        Default :class:`None`
    """

    def __init__(self, parameters=dict(),  energy_coefficient=1.0,
                 forces_coefficient=1.0, stress_coefficient=1.0,
                 **kwargs):

        if not ispymbar:
            msg = "You need pymbar installed to use the MBAR manager"
            raise ModuleNotFoundError(msg)

        WeightingPolicy.__init__(self,
                                 energy_coefficient=energy_coefficient,
                                 forces_coefficient=forces_coefficient,
                                 stress_coefficient=stress_coefficient,
                                 **kwargs)

        if self.database is None:
            self.database = []
        self.parameters = default_parameters
        self.parameters.update(parameters)
        self.Nk = []
        self.train_mlip = False
        self.mlip_coef = []
        self.ukn = []

        self._newddb = []
        self._nstart = self.parameters['start']
        if self._nstart <= 1:
            msg = 'The "start" variable has to be higher than 2.\n'
            msg += 'You need at least two potentials to compare them.'
            raise ValueError(msg)

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def compute_weight(self, coef, predict, docalc=True):
        """
        Save the MLIP coefficients and compute the Weight
        Compute the matrice Ukn of partition fonctions of shape [ndesc, nconf]
        according to the given predict(desc, coef)
        """
        # Update the variables
        if coef is not None:
            self.mlip_coef.append(coef)

        self.train_mlip = True
        self.database.extend(self._newddb)
        self.nconfs = len(self.database)
        if 0 == len(self.mlip_coef):
            self.Nk = np.r_[self.nconfs]
        else:
            self.Nk = np.append(self.Nk, [len(self._newddb)])
        self._newddb = []

        if not docalc:
            return "_", "_"

        # Calculate ukn
        ukn = np.zeros([len(self.mlip_coef), len(self.database)])
        for idx, mlip_coef in enumerate(self.mlip_coef):
            # Calculate the MLIP Energy and Pressure according to the new coef
            mlip_E, mlip_F, mlip_S = predict(self.database, mlip_coef)
            mlip_P = []
            for s in mlip_S:
                mlip_P.append(-np.sum(s[:3])/3)

            ukn[idx] = self._get_ukn(energy=mlip_E,
                                     pressure=mlip_P)
        self.ukn = ukn
        squared_ukn = np.zeros([len(self.ukn), len(self.ukn[-1])])
        for i in range(len(self.ukn)):
            for j in range(len(self.ukn[i])):
                squared_ukn[i, j] = self.ukn[i][j]

        # Finally, calculate weight
        fname = "MLIP.weight"
        header = ''
        if self._nstart <= len(self.mlip_coef):
            weight = self._compute_weight(squared_ukn)
            self.weight = weight
            neff = self.get_effective_conf()

            header += "Using MBAR weighting\n"
            header += f"Effective number of configurations: {neff:10.5f}\n"

            filepath = Path(fname)
            if filepath.exists():
                filepath.unlink()
            np.savetxt(fname, self.weight,
                       header=header, fmt="%25.20f")

            header += "Number of uncorrelated snapshots for each k state:\n"
            header += np.array2string(np.array(self.Nk, 'int')) + "\n"

        else:  # If there isn't enough coef, use UniformWeight
            self.weight = np.ones(len(self.database))/len(self.database)
        return header, "MLIP.weight"

# ========================================================================== #
    def init_weight(self):
        """
        Initialize the weight matrice with W = scale * 1/N.
        """
        return super().init_weight(scale=self.parameters['scale'])

# ========================================================================== #
    def update_database(self, atoms):
        """
        Update the database.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self._newddb.extend(atoms)
        if self.matsize is None:
            self.matsize = []
        self.matsize.extend([len(a) for a in atoms])

        # Sanity Check
        for at in atoms:
            if 'info_state' not in at.info:
                msg = "Atoms don't have 'info_state' for MBAR\n"
                msg += "To use mbar, look at the new traj file with 2 confs."
                msg += "Copy its info['info_state'] and add it to all the "
                msg += "atoms in traj."
                raise ValueError(msg)

# ========================================================================== #
    def _get_ukn(self, energy, pressure):
        """
        Compute Ukn matrices. u[k,n] -> k = mlip, n=conf
        """
        P, V, T = self._get_ensemble_info(pressure)
        PV = P*V / [len(at) for at in self.database]
        ukn = (energy + PV) / (kB * T)
        return ukn

# ========================================================================== #
    def _get_ensemble_info(self, pressure):
        """
        Read the ddb info state and returns arrays of P, dV, T.

        For now, only NVT and NPT are implemented.
        NVT : Aimed T, Instantaneous P, Constant V
        NPT : Aimed T, Constant P from the MLIP, Instantaneous V
        -----------------------------------------------
        NVE : Instantaneous T, No P, No V
        uVT/uPT : NVT/NPT + Constant u, Instantaneous N
        """
        P, V, T = [], [], []
        for idx, at in enumerate(self.database):
            info = at.info['info_state']
            ens = info['ensemble']
            if ens == "NVT":
                T = np.append(T, at.info['info_state']['temperature'])
                P = np.append(P, pressure[idx])
                V = np.append(V, at.get_volume())
            elif ens == "NPT":
                T = np.append(T, at.info['info_state']['temperature'])
                tmp_P = at.info['info_state']['pressure']
                P = np.append(P, tmp_P * GPa)  # From GPa to eV/ang**3
                V = np.append(V, at.get_volume())
            else:
                msg = "Only NVT and NPT are implemented in MLACS for now"
                raise NotImplementedError(msg)
        return P, V, T

# ========================================================================== #
    def _compute_weight(self, sq_ukn):
        """
        Uses pymbar.MAR() class.

        [1] Shirts MR and Chodera JD. Statistically optimal analysis of
        samples from multiple equilibrium states.
        J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177
        """
        mbar = MBAR(sq_ukn, self.Nk,
                    solver_protocol=[{'method': self.parameters['solver']}])
        weight = mbar.weights()[:, -1]
        return weight
