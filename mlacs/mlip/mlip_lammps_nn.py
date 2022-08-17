"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from mlacs.mlip.neural_network import NeuralNetworkMlip
from mlacs.mlip.mlip_lammps_interface import LammpsMlipInterface


# ========================================================================== #
# ========================================================================== #
class LammpsMlipNn(NeuralNetworkMlip):
    """
    MLIP Manager Class to interface with the Neural Network par of the
    ML-IAP package

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Should contains the same elements as the main simulation
    rcut : :class:`float` (optional)
        Cutoff radius, in angstrom. Default 5.0.
    style : \"snap\"`` or ``\"so3\"`` (optional)
        Style of the descriptor. 'snap' is based on the extension of
        the atomic environment on the 4-sphere.
        Default ``\"snap\"``.
    mlip_parameters: :class:`dict`
        Dictionnary containing the parameters for the MLIP
    radelems : :class:`list` (optional)
        Cutoff scaling factor for each elements.
        If ``None``, 0.5 for each elements. Default ``None``.
    welems : :class:`list` (optional)
        :class:`list` of the weight of each atomic type in the descriptor.
        If ``None``, the weight is given by
        :math: ̀\\frac{Z_i}{\\sum_i Z_i} ̀
    nthrow: :class:`int` (optional)
        Number of initial configuration to throw
        as the simulation runs (Counting the training configurations).
        Default ``10``.
    energy_coefficient : :class:`float` (optional)
        Parameter controlling the importance of energy
        in the fitting of the MLIP. Default ``1.0``.
    forces_coefficient : :class:`float` (optional)
        Parameter controlling the importance of forces
        in the fitting of the MLIP. Default ``1.0``.
    stress_coefficient : :class:`float` (optional)
        Parameter controlling the importance of stress
        in the fitting of the MLIP. Default ``0.0``.
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 style="snap",
                 descriptor_parameters=None,
                 nn_parameters=None,
                 radelems=None,
                 welems=None,
                 reference_potential=None,
                 fit_dielectric=False,
                 nthrow=10,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0):

        NeuralNetworkMlip.__init__(self,
                                   atoms,
                                   rcut,
                                   nthrow,
                                   nn_parameters,
                                   energy_coefficient,
                                   forces_coefficient,
                                   stress_coefficient)

        self.lammps_interface = LammpsMlipInterface(self.elements,
                                                    self.masses,
                                                    self.Z,
                                                    self.rcut,
                                                    "nn",
                                                    style,
                                                    descriptor_parameters,
                                                    radelems,
                                                    welems,
                                                    reference_potential,
                                                    fit_dielectric)
        self._initialize_nn()

        self.pair_style, self.pair_coeff, self.model_post = \
            self.lammps_interface.get_pair_coeff_and_style()
        self.atom_style, self.bond_style, self.bond_coeff, \
            self.angle_style, self.angle_coeff = \
            self.lammps_interface.get_bond_angle_coeff_and_style()
        self.bonds, self.angles = self.lammps_interface.get_bonds_angles()
        self.fit_dielectric = fit_dielectric

# ========================================================================== #
    def _get_nelem(self):
        return len(self.lammps_interface.elements)

# ========================================================================== #
    def _get_ncolumns(self):
        return self.lammps_interface.ncolumns - 1

# ========================================================================== #
    def compute_fit_matrix(self, atoms):
        """
        """
        return self.lammps_interface.compute_fit_matrix(atoms)

# ========================================================================== #
    def write_mlip(self, amin, amax, nn_weights, nnodes, activation):
        """
        """
        self.lammps_interface.write_mlip_model_nn(amin,
                                                  amax,
                                                  nn_weights,
                                                  nnodes,
                                                  activation)

# ========================================================================== #
    def init_calc(self):
        """
        """
        if self.lammps_interface.fit_dielectric:
            diel = self.coefficients[-1]
        else:
            diel = None
        self.calc = self.lammps_interface.load_mlip(diel)
        self.pair_style, self.pair_coeff, self.model_post = \
            self.lammps_interface.get_pair_coeff_and_style(diel)

# ========================================================================== #
    def get_mlip_dict(self):
        mlip_dict = self.lammps_interface.get_mlip_dict()
        mlip_dict['energy_coefficient'] = self.energy_coefficient
        mlip_dict['forces_coefficient'] = self.forces_coefficient
        mlip_dict['stress_coefficient'] = self.stress_coefficient
        mlip_dict['regularization'] = None
        return mlip_dict
