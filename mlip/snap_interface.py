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
class LammpsSnapInterface:
    '''
    '''
    def __init__(self, elements, rcutfac=5.0, twojmax=8, chemflag=0, quadraticflag=0):
        self.elements = np.array(sorted([symb for symb in set(elements)]))
        # Get number of bispectrum component
        if twojmax %2 == 0:
            m = 0.5 * twojmax + 1
            if chemflag == 0:
                self.n_bispectrum = int(m * (m + 1) * (2*m + 1) /6)
                bnormflag           = 0
            else:
                self.n_bispectrum = int(m * (m + 1) * (2*m + 1) /6) * len(self.elements)**3
                bnormflag           = 1
        else:
            m = 0.5 * (twojmax + 1)
            if chemflag == 0:
                self.n_bispectrum   = int((m * (m + 1) * (m +2) / 3))
                bnormflag           = 0
            else:
                self.n_bispectrum   = int((m * (m + 1) * (m +2) / 3)) * len(self.elements)**3
                bnormflag           = 1
        # New modif
        if quadraticflag == 1:
            self.n_bispectrum += int(0.5 * self.n_bispectrum * (self.n_bispectrum + 1))

        self.rcutfac       = rcutfac
        self.twojmax       = twojmax
        self.chemflag      = chemflag
        self.quadraticflag = quadraticflag
        self.bnormflag     = bnormflag

        self.ncolumns = self.n_bispectrum * len(self.elements) + len(self.elements)
        self._get_lammps_command()

        self.snap_cmd_line = "{:} 0.99363  {:}".format(self.rcutfac, self.twojmax)
        for n in range(len(self.elements)):
            self.snap_cmd_line += " 0.5 "
        """
        if len(self.elements) == 2:
            self.snap_cmd_line += " 0.7  1.1 "
        else:
        """
        for n in range(len(self.elements)):
            self.snap_cmd_line += " 1.0 "
        if self.chemflag == 1:
            self.snap_cmd_line += "chem {:} ".format(len(self.elements))
            for n in range(len(self.elements)):
                self.snap_cmd_line += "{:} ".format(n)
        self.snap_cmd_line += "quadraticflag {:} bzeroflag 0 bnormflag {:} ".format(self.quadraticflag, self.bnormflag)


#========================================================================================================================#
    def _write_lammps_input(self):
        '''
        '''
        input_string  = "# LAMMPS input file for extracting the SNAP bispectrum\n"
        input_string += "clear\n"
        input_string += "boundary         p p p\n"
        input_string += "atom_style       atomic\n"
        input_string += "units            metal\n"
        input_string += "read_data        ${filename}\n"
        for n1 in range(len(self.elements)):
            input_string += "mass             {i} {mass}\n".format(i=n1+1, mass=1)

        input_string += "pair_style       zero {:}\n".format(self.rcutfac * 2)
        input_string += "pair_coeff       * *\n"

        input_string += "thermo           100\n"
        input_string += "timestep         0.005\n"
        input_string += "neighbor         1.0 bin\n"
        input_string += "neigh_modify     once no every 1 delay 0 check yes\n"

        input_string += "compute          snap all snap " + self.snap_cmd_line + "\n"
        input_string += "fix              snap all ave/time 1 1 1 c_snap[*] file snap.dat mode vector format %.15f\n" 

        input_string += "run              0\n"

        f_lmp = open('base.in', "w")
        f_lmp.write(input_string)
        f_lmp.close()


#========================================================================================================================#
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the bispectrum components and the reference data
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
        Function to cleanup the LAMMPS files used to extract the bispectrum and reference data
        '''
        call("rm lmp.out", shell=True)
        call("rm snap.dat", shell=True)
        call("rm log.lammps", shell=True)
        call("rm base.in", shell=True)
        call("rm atoms.lmp", shell=True)


#========================================================================================================================#
    def construct_lammps_params(self):
        '''
        Function to prepare the parameters to have a potential usable by LAMMPS
        Organized the results so that everything so that ASE can be used as an interface

        # WARNING
        For compatibility with assisted_sampling -> the snap pair_coeff needs to be last in the pair_coeff list
        # WARNING
        '''
        self.pair_coeff = []
        if self.use_zbl or self.use_coulomb:
            self.pair_style = "hybrid/overlay " 
            if self.use_zbl:
                self.pair_style += self.zbl_pair_style
                self.pair_coeff += self.zbl_pair_coeff
            self.pair_style += " snap"

            if self.use_coulomb:
                self.pair_style += " coul/long {:}".format(2*self.bispectrum["rcutfac"]+0.01)
                self.pair_coeff += ["* * coul/long"]
                self.model_post.append("dielectric  {:}\n".format( 1./ self.inv_dielec))

            pc_snap     = "* * snap " + self.folder + "SNAP.snapcoeff " + self.folder + "SNAP.snapparam "
            for n in range(len(self.elements)):
                pc_snap   += self.elements[n] + " "
            self.pair_coeff += [pc_snap]

        else:
            self.pair_style = " snap"
            pc_snap     = "* * " + self.folder + "SNAP.snapcoeff " + self.folder + "SNAP.snapparam "
            for n in range(len(self.elements)):
                pc_snap   += self.elements[n] + " "
            self.pair_coeff += [pc_snap]

        self.files      = [self.folder + "SNAP.snapcoeff", self.folder + "SNAP.snapparam"]


#========================================================================================================================#
    def write_snap_params(self):
        '''
        Function to write a SNAP.snapparam file with the SNAP parameters, usable by LAMMPS
        '''
        with open("SNAP.snapparam", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("SNAP parameters\n")
            f.write("\n")

            f.write("rcutfac       {:}\n".format(self.rcutfac))
            f.write("twojmax       {:}\n".format(self.twojmax))
            f.write("rfac0         {:}\n".format(0.99363))
            f.write("rmin0         {:}\n".format(0.0))
            f.write("chemflag      {:}\n".format(self.chemflag))
            f.write("bzeroflag     {:}\n".format(0))
            f.write("quadraticflag {:}\n".format(self.quadraticflag))
            f.write("bnormflag     {:}\n".format(self.bnormflag))


#========================================================================================================================#
    def write_snap_coeff(self, coefficients):
        '''
        '''
        ncoeff = self.n_bispectrum + 1

        # Adding a comment line to know what elements are fitted here
        coeff_strings  = "# "
        for elements in self.elements:
            coeff_strings += "{:} ".format(elements)
        coeff_strings += "SNAP coefficients\n"
        coeff_strings += "\n"

        # LAMMPS need the number of elements and of coefficient per elements in the first "real" line
        coeff_strings += "{:}  {:}\n".format(len(self.elements), ncoeff)
        for n in range(len(self.elements)):
            coeff_strings += "{:}  {:}  {:}\n".format(self.elements[n], 0.5, 1.0)
            coeff_strings += "{:35.30f}\n".format(coefficients[n])
            for icoeffs in range(self.n_bispectrum):
                coeff_strings += "{:35.30f}\n".format(coefficients[n*(self.n_bispectrum)+icoeffs+len(self.elements)])

        # And we write everything
        snap_coeff_f = open("SNAP.snapcoeff", "w")
        snap_coeff_f.write(coeff_strings)
        snap_coeff_f.close()


#========================================================================================================================#
    def compute_fit_matrix(self, atoms):
        '''
        '''
        nrows    = 3 * len(atoms) + 7


        lmp_atoms_fname =  "atoms.lmp"
        self._write_lammps_input()

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
        
        data = np.loadtxt("snap.dat", skiprows=4)[:,1:]
        
        data_bispectrum        = data[:,:-1]
        data_bispectrum[-6:]  /= atoms.get_volume()

        amatrix[:, len(self.elements):] = data_bispectrum

        # Add the B0 coefficient for the linear fit
        symb = atoms.get_chemical_symbols()
        for n in range(len(self.elements)):
            amatrix[0, n] = symb.count(self.elements[n])

#       amatrix   /= len(atoms)
#       data_true /= len(atoms)

        self.cleanup()

        return amatrix, data_true


#========================================================================================================================#
    def load_snap(self):
        '''
        '''
        pair_style = 'snap'
        pair_coeff = ' * * SNAP.snapcoeff SNAP.snapparam'
        for el in self.elements:
            pair_coeff += " " + el
        pair_coeff = [pair_coeff]
        calc       = LAMMPS()
        calc.set(pair_style=pair_style, pair_coeff=pair_coeff)
        return calc


#========================================================================================================================#
    def read_snap_coeff(self, coeff_file):
        '''
        Read the SNAP.snapcoeff file created with this snap interface

        return the coefficients
        '''
        with open(coeff_file,'r') as f:
            f.readline()
            f.readline()
            f.readline()
            coefficients = np.zeros(self.ncolumns)
            i = 0
            for iel in range(len(self.elements)):
                f.readline()
                for icoef in range(self.n_bispectrum + 1):
                    line = f.readline()
                    coefficients[i] = float(line)
                    i += 1
        return np.array(coefficients)


#========================================================================================================================#
    def get_snap_dict(self):
        """
        """
        snap_dct = {"rcutfac": self.rcutfac,
                    "twojmax": self.twojmax,
                    "chemflag": self.chemflag,
                    "quadraticflag": self.quadraticflag,
                    "ncoef": self.ncolumns
                   }
        return snap_dct
