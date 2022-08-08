"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from mlacs.mlip.linear_mlip import LinearMlip
from mlacs.mlip.mlip_lammps_interface import LammpsMlipInterface


# ========================================================================== #
# ========================================================================== #
class LammpsMlip(LinearMlip):
    """
    MLIP Manager Class to interface with the ML-IAP package

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Should contains the same elements as the main simulation
    rcut : :class:`float` (optional)
        Cutoff radius, in angstrom. Default 5.0.
    model : ``\"linear\"`` or ``\"quadratic\"`` (optional)
        Model of the MLIP. Quadratic increase accuracy at the cost
        of an exponential augmentation of the number of coefficients.
        Default ``\"linear\"``.
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
    regularization: :class:`float`
    energy_coefficient : :class:`float` (optional)
        Parameter controlling the importance of energy
        in the fitting of the MLIP. Default ``1.0``.
    forces_coefficient : :class:`float` (optional)
        Parameter controlling the importance of forces
        in the fitting of the MLIP. Default ``1.0``.
    stress_coefficient : :class:`float` (optional)
        Parameter controlling the importance of stress
        in the fitting of the MLIP. Default ``0.0``.
    rescale_energy : :class:`Bool` (optional)
        If true, the energy data are divided by
        its standard deviation before the fit. Default ``True``.
    rescale_forces : :class:`Bool` (optional)
        If true, the forces data are divided by
        its standard deviation before the fit. Default ``True``.
    rescale_stress : :class:`Bool` (optional)
        If true, the stress data are divided by
        its standard deviation before the fit. Default ``True``.
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 model="linear",
                 style="snap",
                 mlip_parameters=None,
                 radelems=None,
                 welems=None,
                 reference_potential=None,
                 fit_dielectric=False,
                 nthrow=10,
                 regularization=None,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=0.0,
                 rescale_energy=True,
                 rescale_forces=True,
                 rescale_stress=True):
        LinearMlip.__init__(self,
                            atoms,
                            rcut,
                            nthrow,
                            regularization,
                            energy_coefficient,
                            forces_coefficient,
                            stress_coefficient,
                            rescale_energy=True,
                            rescale_forces=True,
                            rescale_stress=True)

        self.lammps_interface = LammpsMlipInterface(self.elements,
                                                    self.masses,
                                                    self.Z,
                                                    self.rcut,
                                                    model,
                                                    style,
                                                    mlip_parameters,
                                                    radelems,
                                                    welems,
                                                    reference_potential,
                                                    fit_dielectric)

        self.ncolumns = self.lammps_interface.ncolumns

        self.pair_style, self.pair_coeff, self.model_post = \
            self.lammps_interface.get_pair_coeff_and_style()
        self.atom_style, self.bond_style, self.bond_coeff, \
            self.angle_style, self.angle_coeff = \
            self.lammps_interface.get_bond_angle_coeff_and_style()
        self.bonds, self.angles = self.lammps_interface.get_bonds_angles()
        self.fit_dielectric = fit_dielectric

# ========================================================================== #
    def get_regularization_vector(self, lamb):
        """
        """
        nelem = len(self.lammps_interface.elements)
        regul = np.zeros(self.lammps_interface.ncolumns)
        regul[nelem:] = 1.0
        if self.fit_dielectric:
            regul[-1] = 0.0
        return regul

# ========================================================================== #
    def compute_fit_matrix(self, atoms):
        """
        """
        return self.lammps_interface.compute_fit_matrix(atoms)

# ========================================================================== #
    def write_mlip(self):
        """
        """
        self.lammps_interface.write_mlip_coeff(self.coefficients)

# ========================================================================== #
    def init_calc(self):
        """
        """
        if self.lammps_interface.fit_dielectric:
            self.calc = self.lammps_interface.load_mlip(self.coefficients[-1])
        else:
            self.calc = self.lammps_interface.load_mlip()
        coeffs = self.coefficients[-1]
        self.pair_style, self.pair_coeff, self.model_post = \
            self.lammps_interface.get_pair_coeff_and_style(coeffs)

# ========================================================================== #
    def get_mlip_dict(self):
        mlip_dict = self.lammps_interface.get_mlip_dict()
        mlip_dict['energy_coefficient'] = self.energy_coefficient
        mlip_dict['forces_coefficient'] = self.forces_coefficient
        mlip_dict['stress_coefficient'] = self.stress_coefficient
        return mlip_dict
