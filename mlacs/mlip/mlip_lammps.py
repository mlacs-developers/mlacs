"""
"""
from mlacs.mlip.mlip_manager import MlipManager
from mlacs.mlip.mlip_lammps_interface import LammpsMlipInterface



#===================================================================================================================================================#
#===================================================================================================================================================#
class LammpsMlip(MlipManager):
    """
    MLIP Manager Class to interface with the ML-IAP package

    Parameters
    ----------

    atoms : ase.atoms object
        Should contains the same elements as the main simulation
    rcut : float
        Cutoff radius, in angstrom
    model : 'linear' or 'quadratic'
        Model of the MLIP. Quadratic increase accuracy at the cost of an exponential augmentation of the number of coefficients
    style : 'snap'
        Style of the descriptor. 'snap' is based on the extension of the atomic environment on the 4-sphere
    twojmax : int
        Parameter 2jmax of the snap descriptor
    lmax : int
        Parameter lmax of radial part of the so3 descriptor
    nmax : int
        Parameter nmax of the angular part of the so3 descriptor
    alpha : float
        Parameter of the gaussian smoothing of the so3 descriptor
    chemflag : 0 or 1
        If 0, the standard descriptor is used. If 1, the explicitely multi-elements is used
    radelems : list (optional)
        Cutoff scaling factor for each elements. If None, 0.5 for each elements.
    welems : list (optional)
        List of the weight of each atomic type in the descriptor. If None, the weight is given by
        :math: ̀\\frac{Z_i}{\\sum_i Z_i} ̀
    energy_coefficient : float
        Parameter controlling the importance of energy in the fitting of the MLIP
    forces_coefficient : float
        Parameter controlling the importance of forces in the fitting of the MLIP
    stress_coefficient : float
        Parameter controlling the importance of stress in the fitting of the MLIP
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 model="linear",
                 style="snap",
                 twojmax=8,
                 lmax=3,
                 nmax=5,
                 alpha=2.0,
                 chemflag=0,
                 radelems=None,
                 welems=None,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=0.0,
                ):

        MlipManager.__init__(self,
                             atoms,
                             rcut,
                             energy_coefficient,
                             forces_coefficient,
                             stress_coefficient,
                            )

        self.lammps_interface = LammpsMlipInterface(self.elements,
                                                    self.masses,
                                                    self.Z,
                                                    self.rcut,
                                                    model,
                                                    style,
                                                    twojmax,
                                                    lmax,
                                                    nmax,
                                                    alpha,
                                                    chemflag,
                                                    radelems,
                                                    welems)

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
