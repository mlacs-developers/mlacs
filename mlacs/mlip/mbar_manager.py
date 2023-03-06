"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pymbar.mbar import MBAR

# ========================================================================== #
# ========================================================================== #
class MbarManager:
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
    no_zstress: :class:`bool` (optional)
        If `True`, the Z components of the stress is not fitted.
        Can be useful to study 2D materials (Default False)
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
    def __init__(self):
        """
        Initialisation
        """


# ========================================================================== #
    def _compute_weights(self):
        """
        """
         
