"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import numpy as np
from subprocess import call

from ase.io.lammpsdata import write_lammps_data
from ase.calculators.lammps import Prism
from ase.calculators.lammpsrun import LAMMPS
from ase.units import GPa


# ========================================================================== #
# ========================================================================== #
class LammpsSnapInterface:
    """
    Class to interface the ML-SNAP package of LAMMPS,
    in order to create linear or quadratic SNAP potential


    Parameters
    ----------

    elements: :class:`list`
        List of elements in the fitting
    masses: :class:`list`
        Masses of the elements in the fitting
    rcut: :class:`float` (optional)
        Cutoff radius for the MLIP. Default ``5.0``.
    twojmax: :class:`int` (optional)
        twojmax parameters, used for the SNAP descriptor. Default ``8``.
    chemflag: :class:`int` (optional)
        ``1`` to enable the explicitely multi-element variation of SNAP
    radelems: :class:`list` of :class:`float` (optional)
        factor to multiply the rcut params to compute interaction.
        One parameter per elements
        If ``None``, is put to the default value of 0.5. Default ``None``.
    welems: :class:`list` of :class:`float` (optional
        Weights factor to enhance the sensibility of different species
        in the descriptors.
        If ``None``, weights factor of element n is put to
        mass_n / sum(all_masses). Default ``None``.
    quadratic : :class:`Bool` (optional)
        Whether to use the quadratic formulation of SNAP. Default ``False``.
    """
    def __init__(self,
                 elements,
                 masses,
                 Z,
                 charges=None,
                 rcut=5.0,
                 twojmax=8,
                 chemflag=0,
                 radelems=None,
                 welems=None,
                 quadratic=False,
                 reference_potential=None,
                 fit_dielectric=False):

        # Store parameters
        self.elements = np.array(elements)
        self.masses = masses
        self.Z = Z
        self.charges = charges
        self.rcut = rcut
        self.twojmax = twojmax
        self.quadratic = quadratic
        self.chemflag = chemflag
        self.bnormflag = 0
        self.fit_dielectric = fit_dielectric
        self.prepare_ref_pot(reference_potential)
        if self.chemflag == 1:
            self.bnormflag = 1

        if radelems is None:
            self.radelems = np.array([0.5 for i in self.elements])
        else:
            self.radelems = radelems
        if welems is None:
            self.welems = np.array(self.Z) / np.sum(self.Z)
        else:
            self.welems = welems

        if self.twojmax % 2 == 0:
            m = 0.5 * self.twojmax + 1
            self.ndescriptors = int(m * (m+1) * (2*m+1) / 6)
        else:
            m = 0.5 * (self.twojmax + 1)
            self.ndescriptors = int(m * (m+1) * (m+2) / 3)
        if self.chemflag == 1:
            self.ndescriptors *= len(self.elements)**3
        if self.quadratic:
            self.ndesc_linear = self.ndescriptors
            self.ndescriptors += int(self.ndescriptors *
                                     (self.ndescriptors + 1) / 2)

        self.ncolumns = int(len(self.elements) * (self.ndescriptors + 1))
        if self.fit_dielectric:
            self.ncolumns += 1
        self._get_lammps_command()

        self.snapline = "{0} 0.99363 {1} ".format(self.rcut, self.twojmax)
        for n in range(len(self.elements)):
            self.snapline += "{:} ".format(self.radelems[n])
        for n in range(len(self.elements)):
            self.snapline += "{:} ".format(self.welems[n])
        if self.chemflag == 1:
            self.snapline += "chem {:} ".format(len(self.elements))
            for n in range(len(self.elements)):
                self.snapline += "{:} ".format(n)
        if self.quadratic:
            self.snapline += "quadraticflag 1 "
        self.snapline += "bnormflag {:}".format(self.bnormflag)

        # Check inputs and raise Error if needed
        if len(np.unique(self.elements)) != len(self.elements):
            raise ValueError
        if len(self.masses) != len(self.elements):
            raise ValueError
        if len(self.welems) != len(self.elements):
            raise ValueError
        if len(self.radelems) != len(self.elements):
            raise ValueError

# ========================================================================== #
    def _write_lammps_input(self):
        '''
        Write the LAMMPS input to extract the descriptor
        and gradient value needed to fit
        '''
        pair_style = f"pair_style     {self.pair_style}  zero {self.rcut*2}\n"
        if self.pair_coeff is None:
            pair_coeff = "pair_coeff       * *\n"
        else:
            pair_coeff = "pair_coeff       * * zero\n"
            for pc in self.pair_coeff:
                pair_coeff += f"pair_coeff         {pc}\n"
        model_post = ""
        if self.model_post is not None:
            for mp in self.model_post:
                model_post += f"{mp}"

        input_string = "# LAMMPS input file for extracting MLIP descriptors\n"
        input_string += "clear\n"
        input_string += "boundary         p p p\n"
        input_string += f"atom_style      {self.atom_style}\n"
        input_string += "units            metal\n"
        input_string += "read_data        ${filename}\n"
        for n1 in range(len(self.elements)):
            input_string += f"mass             {n1+1} {self.masses[n1]}\n"

        input_string += pair_style
        input_string += pair_coeff
        input_string += model_post

        input_string += "thermo         100\n"
        input_string += "timestep       0.005\n"
        input_string += "neighbor       1.0 bin\n"
        input_string += "neigh_modify   once no every 1 delay 0 check yes\n"

        input_string += "compute        snap all snap " + self.snapline + "\n"
        input_string += "fix            snap all ave/time 1 1 1 c_snap[*] " + \
                        "file descriptor.out mode vector\n"  # format %.15f\n"

        input_string += "run              0\n"

        f_lmp = open('base.in', "w")
        f_lmp.write(input_string)
        f_lmp.close()

# ========================================================================== #
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the descriptor and gradient values
        '''
        lammps_command = self.cmd + " -var filename " + lmp_atoms_fname + \
            " -in base.in -log log.lammps > lmp.out"
        call(lammps_command, shell=True)

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd

# ========================================================================== #
    def cleanup(self):
        '''
        Function to cleanup the LAMMPS files used
        to extract the descriptor and gradient values
        '''
        os.remove("lmp.out")
        os.remove("descriptor.out")
        os.remove("log.lammps")
        os.remove("base.in")
        os.remove("atoms.lmp")
        # call("rm lmp.out", shell=True)
        # call("rm descriptor.out", shell=True)
        # call("rm log.lammps", shell=True)
        # call("rm base.in", shell=True)
        # call("rm atoms.lmp", shell=True)

# ========================================================================== #
    def _write_mlip_params(self):
        """
        Function to write the mliap.descriptor parameter files of the MLIP
        """
        with open("MLIP.snap.descriptor", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("SNAP parameters\n")
            if self.pair_coeff is not None:
                f.write("# Fitted with a reference potential\n")
                f.write("# See the mliap.model file to see the parameters\n")
            f.write("\n")

            f.write("rcutfac       {:}\n".format(self.rcut))
            f.write("twojmax       {:}\n".format(self.twojmax))
            f.write("rfac0         0.99363\n")
            f.write("rmin0         0.0\n")
            f.write("chemflag      {:}\n".format(self.chemflag))
            if self.quadratic:
                f.write("quadraticflag 1\n")
            else:
                f.write("quadraticflag 0\n")
            f.write("bnormflag     {:}\n".format(self.bnormflag))

# ========================================================================== #
    def write_mlip_coeff(self, coefficients):
        """
        Function to write the mliap.model parameter files of the MLIP
        """
        if self.fit_dielectric:
            coeff = coefficients[:-1]
        else:
            coeff = coefficients
        with open("MLIP.snap.model", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("SNAP coefficients\n")
            # add some lines showing the LAMMPS parameters
            pair_style, pair_coeff, model_post = \
                self.get_pair_coeff_and_style(coefficients[-1])
            f.write("# Parameters to be used in LAMMPS :\n")
            f.write(f"# pair_style    {pair_style}\n")
            for pc in pair_coeff:
                f.write(f"# pair_coeff   {pc}\n")
            if model_post is not None:
                for mp in model_post:
                    f.write(f"# {mp}")
            f.write(f"# atom_style   {self.atom_style}\n")

            f.write("\n")

            f.write(f"{len(self.elements)}   {self.ndescriptors+1}\n")
            for n in range(len(self.elements)):
                f.write(f"{self.elements[n]}  {self.radelems[n]}  " +
                        f"{self.welems[n]}\n")
                f.write("{:35.30f}\n".format(coeff[n]))
                for icoef in range(self.ndescriptors):
                    value = coeff[n*(self.ndescriptors) + icoef +
                                  len(self.elements)]
                    f.write(f"{value:35.30f}\n")

# ========================================================================== #
    def compute_fit_matrix(self, atoms):
        """
        Function to extract the descriptor and gradient values,
        as well as the true data
        Takes in input an atoms with a calculator attached
        """
        nrows = 3 * len(atoms) + 7

        lmp_atoms_fname = "atoms.lmp"
        self._write_lammps_input()
        self._write_mlip_params()

        amatrix = np.zeros((nrows, self.ncolumns))

        true_energy = atoms.get_potential_energy()
        true_forces = atoms.get_forces()
        true_stress = atoms.get_stress()

        # We need to reorganize the forces from ase to LAMMPS vector
        # because of the weird non orthogonal LAMMPS input
        prism = Prism(atoms.get_cell())
        for iat in range(true_forces.shape[0]):
            true_forces[iat] = prism.vector_to_lammps(true_forces[iat])

        # Same thing for the stress
        xx, yy, zz, yz, xz, xy = true_stress
        str_ten = np.array([[xx, xy, xz],
                            [xy, yy, yz],
                            [xz, yz, zz]])

        rot_mat = prism.rot_mat
        str_ten = rot_mat @ str_ten
        str_ten = str_ten @ rot_mat.T
        stress = str_ten[[0, 1, 2, 1, 0, 0],
                         [0, 1, 2, 2, 2, 1]]
        true_stress = -stress

        # Organize the data in the same order as in the LAMMPS output:
        # Energy, then 3*Nat forces,
        # then 6 stress in form xx, yy, zz, yz, xz and xy
        # order C -> lammps is in C++, not FORTRAN
        data_true = np.append(true_energy, true_forces.flatten(order="C"))
        data_true = np.append(data_true, true_stress)

        write_lammps_data(lmp_atoms_fname, atoms, atom_style=self.atom_style)
        self._run_lammps(lmp_atoms_fname)

        # I definitely hate stress units in LAMMPS
        # ASE gives eV/angs**3 - LAMMPS are in bar (WTF ?)
        # and bispectrum component are in ??????
        bispectrum = np.loadtxt("descriptor.out", skiprows=4)
        bispectrum[-6:, 1:-1] /= atoms.get_volume()
        bispectrum[-6:, -1] *= 1e-4 * GPa
        data_bispectrum = bispectrum[:, 1:-1]

        # We need to remove the reference potential values from the data
        data_true -= bispectrum[:, -1]

        if self.fit_dielectric:
            coul_calc = LAMMPS(pair_style=f"coul/long {self.rcut+0.01}",
                               pair_coeff=["* *"],
                               model_post=[f"kspace_style {self.kspace}\n",
                                           "dielectric 1\n"],
                               atom_style="charge",
                               keep_alive=False)
            coul_at = atoms.copy()
            coul_at.calc = coul_calc

            coul_energy = coul_at.get_potential_energy()
            coul_forces = coul_at.get_forces()
            coul_stress = coul_at.get_stress()

            # We need to reorganize the forces from ase to LAMMPS vector
            # because of the weird non orthogonal LAMMPS input
            for iat in range(coul_forces.shape[0]):
                coul_forces[iat] = prism.vector_to_lammps(coul_forces[iat])

            # Same thing for the stress
            xx, yy, zz, yz, xz, xy = coul_stress
            str_ten = np.array([[xx, xy, xz],
                                [xy, yy, yz],
                                [xz, yz, zz]])

            rot_mat = prism.rot_mat
            str_ten = rot_mat @ str_ten
            str_ten = str_ten @ rot_mat.T
            stress = str_ten[[0, 1, 2, 1, 0, 0],
                             [0, 1, 2, 2, 2, 1]]
            coul_stress = -stress

            data_coul = np.append(coul_energy, coul_forces.flatten(order="C"))
            data_coul = np.append(data_coul, coul_stress)

            data_bispectrum = np.hstack((data_bispectrum,
                                         data_coul[:, np.newaxis]))

            amatrix[:, len(self.elements):] = data_bispectrum
        else:
            amatrix[:, len(self.elements):] = data_bispectrum

        symb = atoms.get_chemical_symbols()
        for n in range(len(self.elements)):
            amatrix[0, n] = symb.count(self.elements[n])

        self.cleanup()

        return amatrix, data_true

# ========================================================================== #
    def load_mlip(self, dielectric=None):
        '''
        Function to load a MLIP model
        Return a LAMMPSRUN calculator from the ASE package
        '''
        cwd = os.getcwd()
        pair_style_mliap = " snap "
        pair_coeff_mliap = f"{cwd}/MLIP.snap.model " + \
                           f"{cwd}/MLIP.snap.descriptor " + \
                           " ".join(self.elements)

        pair_style = self.pair_style + pair_style_mliap
        pair_coeff = []
        if self.pair_coeff is None:
            pair_coeff = [f"* * {pair_coeff_mliap}"]
        else:
            pair_coeff.append(f"* * snap {pair_coeff_mliap}")
            for pc in self.pair_coeff:
                pair_coeff.append(pc)

        if self.model_post is None:
            model_post = None
        else:
            model_post = self.model_post.copy()

        if self.fit_dielectric:
            # We need to modifiy the pair style/coeff
            # if there is no reference potential
            if self.pair_coeff is None:
                pair_style = f"hybrid/overlay {self.pair_style} " + \
                             f"{pair_style_mliap}"
                pair_coeff = [f"* * snap {pair_coeff_mliap}"]
            pair_style = pair_style + f"  coul/long {self.rcut+0.01}"
            pair_coeff.append("* * coul/long")
            if self.model_post is not None:
                for mp in self.model_post:
                    if mp.split()[0] == "kspace_style":
                        break
                    else:
                        model_post.append(f"kspace_style   {self.kspace}\n")
                        model_post.append(f"dielectric   {1.0/dielectric}\n")
            else:
                model_post = [f"kspace_style   {self.kspace}\n"]
                model_post.append(f"dielectric   {1.0/dielectric}\n")

        calc = LAMMPS(keep_alive=False)
        calc.set(pair_style=pair_style,
                 pair_coeff=pair_coeff,
                 atom_style=self.atom_style)
        if model_post is not None:
            calc.set(model_post=model_post)
        return calc

# ========================================================================== #
    def get_pair_coeff_and_style(self, dielectric=1):
        """
        """
        cwd = os.getcwd()

        pair_style_mliap = " snap "
        pair_coeff_mliap = f"{cwd}/MLIP.snap.model " + \
                           f"{cwd}/MLIP.snap.descriptor " + \
                           " ".join(self.elements)

        pair_style = self.pair_style + pair_style_mliap
        pair_coeff = []
        if self.pair_coeff is None:
            pair_coeff = [f"* * {pair_coeff_mliap}"]
        else:
            pair_coeff.append(f"* * snap {pair_coeff_mliap}")
            for pc in self.pair_coeff:
                pair_coeff.append(pc)

        if self.model_post is None:
            model_post = None
        else:
            model_post = self.model_post.copy()

        if self.fit_dielectric:
            # We need to modifiy the pair style/coeff
            # if there is no reference potential
            if self.pair_coeff is None:
                pair_style = "hybrid/overlay    " + \
                             f"{self.pair_style} " + \
                             f"{pair_style_mliap}"
                pair_coeff = [f"* * snap {pair_coeff_mliap}"]
            pair_style = pair_style + f"  coul/long {self.rcut+0.01}"
            pair_coeff.append("* * coul/long")
            if self.model_post is not None:
                for mp in self.model_post:
                    if mp.split()[0] == "kspace_style":
                        break
                    else:
                        model_post.append(f"kspace_style   {self.kspace}\n")
                        model_post.append(f"dielectric   {1.0/dielectric}\n")
            else:
                model_post = [f"kspace_style   {self.kspace}\n"]
                model_post.append(f"dielectric   {1.0/dielectric}\n")
        return pair_style, pair_coeff, model_post

# ========================================================================== #
    def prepare_ref_pot(self, ref_pot):
        """
        """
        if ref_pot is None:
            self.pair_style = ""
            self.pair_coeff = None
            self.model_post = None
            self.atom_style = "atomic"
        else:
            self.pair_style = ref_pot.get("pair_style", "")
            self.pair_coeff = ref_pot.get("pair_coeff", None)
            self.atom_style = ref_pot.get("atom_style", "atomic")
            self.model_post = ref_pot.get("model_post", None)
            if isinstance(self.model_post, str):
                self.model_post = [self.model_post + "\n"]

        if self.pair_style != "":
            ref_ps = self.pair_style
            ref_pc = self.pair_coeff
            if isinstance(ref_ps, str):
                ref_ps = [ref_ps]

            self.pair_style = "hybrid/overlay  "
            self.pair_style += "   ".join(ref_ps)
            self.pair_coeff = []
            for i, ps in enumerate(ref_ps):
                pss = ps.split()[0]  # pair_style
                pc = ref_pc[i]
                if isinstance(pc, str):
                    pcs = pc.split()  # pair_coeff splitted
                    if len(pcs) < 2:
                        self.pair_coeff.append(f"{pcs[0]} {pcs[1]} {pss} ")
                    else:
                        self.pair_coeff.append(f"{pcs[0]} {pcs[1]} {pss} " +
                                               " ".join(pcs[2:]))
                else:
                    for p in pc:
                        pcs = p.split()
                        if len(pcs) < 2:
                            self.pair_coeff.append(f"{pcs[0]} {pcs[1]} {pss} ")
                        else:
                            self.pair_coeff.append(f"{pcs[0]} {pcs[1]} " +
                                                   f"{pss} " +
                                                   " ".join(pcs[2:]))

        if self.fit_dielectric:
            self.kspace = "pppm 1.0e-5"
            if self.model_post is not None:
                for mp in self.model_post:
                    if mp.split()[0] == "kspace_style":
                        self.kspace = mp.split()[1:]
            if self.atom_style not in ["full", "charge"]:
                self.atom_style = "charge"

# ========================================================================== #
    def get_mlip_dict(self):
        """
        Return a dictionnary with the parameters of the MLIP potential
        """
        mlip_dct = {"style": "snap",
                    "rcut": self.rcut,
                    "twojmax": self.twojmax,
                    "chemflag": self.chemflag,
                    "ncoef": self.ncolumns}
        if not self.quadratic:
            mlip_dct["model"] = "linear"
        else:
            mlip_dct["model"] = "quadratic"
        if self.chemflag == 1:
            mlip_dct["chemflag"] = 1
        else:
            mlip_dct["chemflag"] = 0
        return mlip_dct
