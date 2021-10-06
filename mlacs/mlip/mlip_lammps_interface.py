import os
import warnings
import numpy as np
from subprocess import call

from ase.io.lammpsdata import write_lammps_data
from ase.calculators.lammps import Prism, convert
from ase.calculators.lammpsrun import LAMMPS
from ase.units import GPa


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
class LammpsMlipInterface:
    """
    Class to interface the ML-IAP package of LAMMPS, in order to create linear or quadratic MLIP


    Parameters
    ----------

    elements: list
        list of elements in the fitting
    masses: list
        masses of the elements in the fitting
    rcut: float
        cutoff radius for the MLIP
    model: string
        linear or quadratic, denote the model used for the MLIP
    style: string
        snap or so3, denote the descriptor used for the MLIP
    twojmax: int
        twojmax parameters, used for the SNAP descriptor
    lmax: int
        angular momentum used for the SO3 descriptor
    nmax: int
        number of radial basis for the SO3 descriptor
    alpha: int
        gaussian factor for the radial basis in the SO3 descriptor
    chemflag: int
        1 to enable the explicitely multi-element variation of SNAP
    radelems: list of float
        factor to multiply the rcut params to compute interaction. One parameter per elements
        If None, is put to the default value of 0.5
    welems: list of float
        weights factor to enhance the sensibility of different species in the descriptors.
        if None, weights factor of element n is put to mass_n / sum(all_masses)
    """
    def __init__(self, 
                 elements, 
                 masses, 
                 Z,
                 rcut=5.0, 
                 model="linear", 
                 style="snap",
                 twojmax=8,
                 lmax=3,
                 nmax=5,
                 alpha=2.0,
                 chemflag=0,
                 radelems=None,
                 welems=None
                ):
        # Waiting for LAMMPS to finnish the implementation of SO3 potential
        if style != "snap":
            raise ValueError

        # Store parameters
        self.elements = np.array(elements)
        self.masses   = masses
        self.Z        = Z
        self.rcut     = rcut
        self.model    = model
        self.style    = style
        self.twojmax  = twojmax
        self.lmax     = lmax
        self.nmax     = nmax
        self.alpha    = alpha
        self.chemflag = chemflag
        if self.chemflag == 1:
            self.bnormflag = 1

        if radelems == None:
            self.radelems = np.array([0.5 for i in self.elements])
        else:
            self.radelems = radelems
        if welems == None:
            #self.welems = np.array(self.masses) / np.sum(self.masses)
            self.welems = np.array(self.Z) / np.sum(self.Z)
        else:
            self.welems = welems

        # Initialize the descriptor dimension, depending on the descriptor style and parameters
        if self.style == "so3":
            self.ndescriptors = int(self.nmax * (self.nmax + 1) * (self.lmax + 1) / 2)
        elif self.style == "snap":
            if self.twojmax %2 == 0:
                m                 = 0.5 * self.twojmax + 1
                self.ndescriptors = int(m * (m+1) * (2*m+1) / 6)
            else:
                m                 = 0.5 * (self.twojmax + 1)
                self.ndescriptors = int(m * (m+1) * (m+2) / 3)
            if self.chemflag == 1:
                self.ndescriptors *= len(self.elements)**3
        if self.model == "quadratic":
            self.ndescriptors += int(self.ndescriptors * (self.ndescriptors + 1) / 2)

        self.ncolumns = int(len(self.elements) * (self.ndescriptors + 1))
        self._get_lammps_command()

        # Check inputs and raise Error if needed
        if len(np.unique(self.elements)) != len(self.elements):
            raise ValueError
        if len(self.masses) != len(self.elements):
            raise ValueError
        if len(self.welems) != len(self.elements):
            raise ValueError
        if len(self.radelems) != len(self.elements):
            raise ValueError


#========================================================================================================================#
    def _write_lammps_input(self):
        '''
        Write the LAMMPS input to extract the descriptor and gradient value needed to fit
        '''
        input_string  = "# LAMMPS input file for extracting the MLIP descriptors\n"
        input_string += "clear\n"
        input_string += "boundary         p p p\n"
        input_string += "atom_style       atomic\n"
        input_string += "units            metal\n"
        input_string += "read_data        ${filename}\n"
        for n1 in range(len(self.elements)):
            input_string += "mass             {i} {mass}\n".format(i=n1+1, mass=self.masses[n1])

        input_string += "pair_style       zero {:}\n".format(self.rcut * 2)
        input_string += "pair_coeff       * *\n"

        input_string += "thermo           100\n"
        input_string += "timestep         0.005\n"
        input_string += "neighbor         1.0 bin\n"
        input_string += "neigh_modify     once no every 1 delay 0 check yes\n"

#       input_string += "compute          test all mliap model " + self.model + " descriptor " + self.style + " MLIP.mliap.descriptor\n"
#       input_string += "compute          mliap all mliap  descriptor " + self.style + " MLIP.mliap.descriptor  model " + self.model + "\n"
#       input_string += "compute          mliap all mliap  descriptor " + self.style + " MLIP.mliap.descriptor  model " + self.model + "\n"
        if self.style == "snap":
            input_string += "compute          mliap all mliap  descriptor sna MLIP.mliap.descriptor  model " + self.model + "\n"
        elif self.style == "so3":
            input_string += "compute          mliap all mliap  descriptor so3 MLIP.mliap.descriptor  model " + self.model + " gradgradflag 1\n"
#       input_string += "compute          snap all snap " + self.snap_cmd_line + "\n"
        input_string += "fix              mliap all ave/time 1 1 1 c_mliap[*] file descriptor.out mode vector\n"# format %.15f\n" 

        input_string += "run              0\n"

        f_lmp = open('base.in', "w")
        f_lmp.write(input_string)
        f_lmp.close()


#========================================================================================================================#
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the descriptor and gradient values
        '''
        lammps_command = self.cmd + " -var filename " + lmp_atoms_fname + " -in base.in -log log.lammps > lmp.out"
        call(lammps_command, shell=True)


#========================================================================================================================#
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd    = os.environ.get(envvar)
        if cmd is None:
            cmd    = "lmp"
        self.cmd = cmd


#========================================================================================================================#
    def cleanup(self):
        '''
        Function to cleanup the LAMMPS files used to extract the descriptor and gradient values
        '''
        call("rm lmp.out", shell=True)
        call("rm descriptor.out", shell=True)
        call("rm log.lammps", shell=True)
        call("rm base.in", shell=True)
        call("rm atoms.lmp", shell=True)


#========================================================================================================================#
    def _write_mlip_params(self):
        """
        Function to write the mliap.descriptor parameter files of the MLIP
        """
        with open("MLIP.mliap.descriptor", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("MLIP parameters\n")
            f.write("# Descriptor:  " + self.style + "\n")
            f.write("# Model:       " + self.model + "\n")
            f.write("\n")
            f.write("rcutfac     {:}\n".format(self.rcut))
            if self.style == "so3":
                f.write("nmax        {:}\n".format(self.nmax))
                f.write("lmax        {:}\n".format(self.lmax))
                f.write("alpha       {:}\n".format(self.alpha))

            elif self.style == "snap":
                f.write("twojmax     {:}\n".format(self.twojmax))
            f.write("\n\n\n")
            f.write("nelems      {:}\n".format(len(self.elements))) 
            f.write("elems       ")
            for n in range(len(self.elements)):
                f.write(self.elements[n] + " ")
            f.write("\n")
            f.write("radelems   ")
            for n in range(len(self.elements)):
                f.write(" {:}".format(self.radelems[n]))
            f.write("\n")
            f.write("welems    ")
            for n in range(len(self.elements)):
                f.write("  {:}".format(self.welems[n]))
            f.write("\n")

            if self.style == "snap" and self.chemflag == 1:
                f.write("\n\n")
                f.write("chemflag     1\n")
                f.write("bnormflag    1\n")


#========================================================================================================================#
    def write_mlip_coeff(self, coefficients):
        """
        Function to write the mliap.model parameter files of the MLIP
        """
        with open("MLIP.mliap.model", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("MLIP parameters\n")
            f.write("# Descriptor:  " + self.style + "\n")
            f.write("# Model:       " + self.model + "\n")
            f.write("\n")
            f.write("# nelems   ncoefs\n")
            f.write("{:}  {:}\n".format(len(self.elements), self.ndescriptors+1))
            for icoef in range(len(coefficients)):
                f.write("{:35.30f}\n".format(coefficients[icoef]))


#========================================================================================================================#
    def compute_fit_matrix(self, atoms):
        """
        Function to extract the descriptor and gradient values, as well as the true data
        Takes in input an atoms with a calculator attached
        """
        nrows = 3 * len(atoms) + 7

        lmp_atoms_fname = "atoms.lmp"
        self._write_lammps_input()
        self._write_mlip_params()

        amatrix  = np.zeros((nrows, self.ncolumns))
        ymatrix  = np.zeros((nrows))

        true_energy = atoms.get_potential_energy()
        true_forces = atoms.get_forces()
        true_stress = atoms.get_stress()
        
        # We need to reorganize the forces from ase to LAMMPS vector because of the weird non orthogonal LAMMPS input
        prism = Prism(atoms.get_cell())
        for iat in range(true_forces.shape[0]):
            true_forces[iat] = prism.vector_to_lammps(true_forces[iat])
        
        # Same thing for the stress
        xx, yy, zz, yz, xz, xy = true_stress
        str_ten = np.array([[xx, xy, xz],
                            [xy, yy, yz],
                            [xz, yz, zz]])
        
        rot_mat = prism.rot_mat
        str_ten = np.dot(rot_mat, str_ten)
        str_ten = np.dot(str_ten, rot_mat.T)
        stress  = str_ten[[0, 1, 2, 1, 0, 0],
                          [0, 1, 2, 2, 2, 1]]
        true_stress = -stress

        # Organize the data in the same order as in the LAMMPS output:
        # Energy, then 3*Nat forces, then 6 stress in form xx, yy, zz, yz, xz and xy
        data_true   = np.append(true_energy, true_forces.flatten(order="C")) # order C -> lammps is in C++, not FORTRAN
        data_true   = np.append(data_true,   true_stress)
        
        write_lammps_data(lmp_atoms_fname, atoms)
        self._run_lammps(lmp_atoms_fname)
        
        data_bispectrum = np.loadtxt("descriptor.out", skiprows=4)[:,1:-1]
        
        data_bispectrum[-6:]  /= atoms.get_volume()

        amatrix[:] = data_bispectrum

        self.cleanup()

        return amatrix, data_true


#========================================================================================================================#
    def load_mlip(self):
        '''
        Function to load a MLIP model
        Return a LAMMPSRUN calculator from the ASE package
        '''
        if self.style == "snap":
            style = "sna"
        elif self.style == "so3":
            style = "so3"
        pair_style = 'mliap  model ' + self.model + " MLIP.mliap.model  descriptor " + style + " MLIP.mliap.descriptor"
        pair_coeff = ' * * '
        for el in self.elements:
            pair_coeff += " " + el
        pair_coeff = [pair_coeff]
        pwd = os.getcwd()
        calc       = LAMMPS(keep_alive=False)
        calc.set(pair_style=pair_style, pair_coeff=pair_coeff)
        return calc


#========================================================================================================================#
    def get_pair_coeff_and_style(self):
        """
        """
        cwd = os.getcwd()

        if self.style == "snap":
            style = "sna"
        elif self.style == "so3":
            style = "so3"
        pair_style = "mliap model " + self.model + " " + cwd + "/MLIP.mliap.model  descriptor " + style + " " + cwd + "/MLIP.mliap.descriptor"

        pair_coeff = "* * " + " ".join(self.elements)
        return pair_style, pair_coeff


#========================================================================================================================#
    def get_mlip_dict(self):
        """
        Return a dictionnary with the parameters of the MLIP potential
        """
        mlip_dct = {"style": self.style,
                    "rcut": self.rcut,
                    "twojmax": self.twojmax,
                    "lmax": self.lmax,
                    "nmax": self.nmax,
                    "alpha": self.alpha,
                    "chemflag": self.chemflag,
                    "model": self.model,
                    "ncoef": self.ncolumns
                   }
        return mlip_dct
