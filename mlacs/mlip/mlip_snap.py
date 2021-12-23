"""
"""
import numpy as np

from ase.units import GPa

from mlacs.mlip.linear_mlip import LinearMlip
from mlacs.mlip.mlip_snap_interface import LammpsSnapInterface



#===================================================================================================================================================#
#===================================================================================================================================================#
class LammpsSnap(LinearMlip):
    """
    MLIP Manager Class to interface with the SNAP package

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Should contains the same elements as the main simulation
    rcut : float (optional)
        Cutoff radius, in angstrom. Default 5.0.
    twojmax : int (optiona)
        Parameter 2jmax of the snap descriptor. Default 8.
    chemflag : 0 or 1 (optional)
        If 0, the standard descriptor is used. If 1, the explicitely multi-elements is used. Default 0.
    radelems : list (optional)
        Cutoff scaling factor for each elements. If None, 0.5 for each elements. Default None.
    welems : list (optional)
        List of the weight of each atomic type in the descriptor. If None, the weight is given by
        :math: ̀\\frac{Z_i}{\\sum_i Z_i} ̀
    quadratic: bool (optional)
        Whether to use the quadratic implementation of SNAP. Default False.
    nthrow: int (optional)
        Number of initial configuration to throw as the simulation runs (Counting the training configurations).
        Default 10.
    energy_coefficient : float (optional)
        Parameter controlling the importance of energy in the fitting of the MLIP. Default 1.0.
    forces_coefficient : float (optional)
        Parameter controlling the importance of forces in the fitting of the MLIP. Default 1.0.
    stress_coefficient : float (optional)
        Parameter controlling the importance of stress in the fitting of the MLIP. Default 0.0.
    rescale_energy : Bool (optional)
        If true, the energy data are divided by its standard deviation before the fit. Default True.
    rescale_forces : Bool (optional)
        If true, the forces data are divided by its standard deviation before the fit. Default True.
    rescale_stress : Bool (optional)
        If true, the stress data are divided by its standard deviation before the fit. Default True.
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 twojmax=8,
                 chemflag=0,
                 radelems=None,
                 welems=None,
                 quadratic=False,
                 nthrow=10,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=0.0,
                 rescale_energy=True,
                 rescale_forces=True,
                 rescale_stress=True
                ):

        LinearMlip.__init__(self,
                            atoms,
                            rcut,
                            nthrow,
                            energy_coefficient,
                            forces_coefficient,
                            stress_coefficient,
                            rescale_energy=True,
                            rescale_forces=True,
                            rescale_stress=True
                           )

        self.lammps_interface = LammpsSnapInterface(self.elements,
                                                    self.masses,
                                                    self.Z,
                                                    self.rcut,
                                                    twojmax,
                                                    chemflag,
                                                    radelems,
                                                    welems,
                                                    quadratic
                                                   )

        self.ncolumns = self.lammps_interface.ncolumns

        self.pair_style, self.pair_coeff = self.lammps_interface.get_pair_coeff_and_style()


#===================================================================================================================================================#
    def compute_fit_matrix(self, atoms):
        """
        """
        return self.lammps_interface.compute_fit_matrix(atoms)


#===================================================================================================================================================#
    def write_mlip(self):
        """
        """
        self.lammps_interface.write_mlip_coeff(self.coefficients)


#===================================================================================================================================================#
    def init_calc(self):
        """
        """
        self.calc = self.lammps_interface.load_mlip()


#===================================================================================================================================================#
    def get_mlip_dict(self):
        mlip_dict = self.lammps_interface.get_mlip_dict()
        mlip_dict['energy_coefficient'] = self.energy_coefficient
        mlip_dict['forces_coefficient'] = self.forces_coefficient
        mlip_dict['stress_coefficient'] = self.stress_coefficient
        return mlip_dict
