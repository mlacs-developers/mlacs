"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pathlib import Path
import logging
import numpy as np
from ..utilities import subfolder

try:
    # With the annoying mandatory warning from mbar, we have to initialize
    # the log here otherwise the log doesn't work
    # I have to see how to handle this in a better way.
    # This might be an indication of needing to redo the logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    from pymbar import MBAR
    logger.setLevel(logging.INFO)
    ispymbar = True
except ModuleNotFoundError:
    ispymbar = False

from ase.atoms import Atoms
from ase.units import kB
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
    mode: :class:`str`
        Define how to use MBAR.
            - compute: Compute weights.
            - train: Compute weights and use it for MLIP training.
        Default compute

    solver: :class:`str`
        Define type of solver for pymbar
        Default L-BFGS-B

    scale: :class:`float`
        Imposes weights for the new configurations.
        Only relevant in the train mode.
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
                 database=None, weight=None):
        if not ispymbar:
            msg = "You need pymbar installed to use the MBAR manager"
            raise ModuleNotFoundError(msg)

        WeightingPolicy.__init__(
                self,
                energy_coefficient=energy_coefficient,
                forces_coefficient=forces_coefficient,
                stress_coefficient=stress_coefficient,
                database=database, weight=weight)

        self.database = []
        self.parameters = default_parameters
        self.parameters.update(parameters)
        self.Nk = []
        self.train_mlip = False
        self.mlip_coef = []
        self.mlip_desc = []

        self._newddb = []
        self._nstart = self.parameters['start']
        if self._nstart <= 1:
            msg = 'The "start" variable has to be higher than 2.\n'
            msg += 'You need at least two potentials to compare them.'
            raise ValueError(msg)

# ========================================================================== #
    @subfolder
    def compute_weight(self, desc, coef, f_mlipE):
        """
        Save the descriptor and the MLIP coefficients.
        Compute the matrice Ukn of partition fonctions of shape [ndesc, nconf]
        according to the given f_mlipE(coef,desc)
        """
        if coef is not None:
            self.mlip_coef.append(coef)
            self.mlip_desc.append(desc)

        self.train_mlip = True
        self.database.extend(self._newddb)
        self.nconfs = len(self.database)

        if 0 == len(self.mlip_coef):
            self.Nk = np.r_[self.nconfs]
        else:
            self.Nk = np.append(self.Nk, [len(self._newddb)])
        self._newddb = []

        header = ''
        if self._nstart <= len(self.mlip_coef):

            shape = (len(self.mlip_coef), len(self.mlip_desc[-1]))
            ukn = np.zeros(shape)
            for istep, coeff in enumerate(self.mlip_coef):
                mlip_E = f_mlipE(coeff, self.mlip_desc[-1])
                ukn[istep] = self._get_ukn(mlip_E)

            weight = self._compute_weight(ukn)
            self.weight = weight
            neff = self.get_effective_conf()

            header += "Using MBAR weighting\n"
            header += f"Effective number of configurations: {neff:10.5f}\n"

            if Path("MLIP.weight").exists():
                Path("MLIP.weight").unlink()
            np.savetxt("MLIP.weight", self.weight,
                       header=header, fmt="%25.20f")
            header += "Number of uncorrelated snapshots for each k state:\n"
            header += np.array2string(np.array(self.Nk, 'int')) + "\n"

        return header, "MLIP.weight"

# ========================================================================== #
    def init_weight(self):
        """
        Initialize the weight matrice with W = scale * 1/N.
        """
        n_tot = len(self.matsize)
        weight = np.ones(n_tot) / n_tot
        if self._nstart < len(self.database):
            weight = self.parameters['scale'] * weight
            weight[:len(self.weight)] = self.weight
        return weight / np.sum(weight)

# ========================================================================== #
    def get_effective_conf(self):
        """
        Compute the number of effective configurations.
        Gives an idea on MLACS convergence.
        """
        neff = np.sum(self.weight)**2 / np.sum(self.weight**2)
        return neff

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

# ========================================================================== #
    def _get_ukn(self, ekn):
        """
        Compute Ukn matrices.
        """
        # Sanity Check
        for at in self.database:
            if 'info_state' not in at.info:
                msg = "Atoms don't have 'info_state' for the thermodynamic"
                raise ValueError(msg)
        assert len(ekn) == self.nconfs

        P, V, T = self._get_ensemble_info()
        ukn = (ekn + P * V) / (kB * T)
        return ukn

# ========================================================================== #
    def _get_ensemble_info(self):
        """
        Read the ddb info state and returns arrays of P, V, T.

        For now, only NVT and NPT are implemented.
        NVT : Aimed T, Constant P, Constant V
        NPT : Aimed T, Instantaneous P, Instantaneous V
        -----------------------------------------------
        NVE : Instantaneous T, No P, No V
        uVT/uPT : NVT/NPT + Constant u, Instantaneous N
        """
        P, V, T = [], [], []
        for at in self.database:
            info = at.info['info_state']
            ens = info['ensemble']
            if ens == "NVT":
                T = np.append(T, at.info['info_state']['temperature'])
                P = np.append(P, -np.sum(at.get_stress()[:3])/3)
                V = np.append(V, at.get_volume())
            elif ens == "NPT":
                raise NotImplementedError
            else:
                msg = "Only NVT and NPT are implemented in MLACS for now"
                raise NotImplementedError(msg)
        return P, V, T

# ========================================================================== #
    def _compute_weight(self, ukn):
        """
        Uses pymbar.MAR() class.

        [1] Shirts MR and Chodera JD. Statistically optimal analysis of
        samples from multiple equilibrium states.
        J. Chem. Phys. 129:124105, 2008.  http://dx.doi.org/10.1063/1.2978177
        """
        mbar = MBAR(ukn, self.Nk,
                    solver_protocol=[{'method': self.parameters['solver']}])
        weight = mbar.weights()[:, -1]
        return weight
