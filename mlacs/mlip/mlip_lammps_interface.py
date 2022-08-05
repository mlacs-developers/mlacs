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

from mlacs.utilities import write_lammps_data_full


# ========================================================================== #
# ========================================================================== #
class LammpsMlipInterface:
    """
    Class to interface the ML-IAP package of LAMMPS,
    in order to create linear or quadratic MLIP


    Parameters
    ----------

    elements: :class:`list`
        List of elements in the fitting
    masses: :class:`list`
        Masses of the elements in the fitting
    rcut: :class:`float` (optional)
        Cutoff radius for the MLIP. Default ``5.0``.
    model: :class:`string` (optional)
        ``\"linear\"`` or ``\"quadratic\"``, the model used for the MLIP.
        Default ``\"linear\"``.
    style: :class:`string` (optional)
        ``\"snap\"`` or ``\"so3\"``, the descriptor used for the MLIP.
        Default ``\"snap\"``.
    twojmax: :class:`int` (optional)
        2Jmax parameters, used for the SNAP descriptor. Default ``8``
    lmax: :class:`int` (optional)
        Angular momentum used for the SO3 descriptor. Default ``3``.
    nmax: :class:`int` (optional)
        Number of radial basis for the SO3 descriptor. Default ``5``.
    alpha: :class:`int` (optional)
        Gaussian factor for the radial basis in the SO3 descriptor.
        Default ``2.0``.
    chemflag: :class:`int` (optional)
        ``1`` to enable the explicitely multi-element variation of SNAP.
        Default ``0``.
    radelems: :class:`list` of :class:`float` (optional)
        Factor to multiply the rcut params to compute interaction.
        One parameter per elements
        If ``None``, is put to the default value of 0.5. Default ``None``.
    welems: :class:`list` of :class:`float` (optional)
        Weights factor to enhance the sensibility of different species
        in the descriptors.
        If ``None``, weights factor of element n is put to
        mass_n / sum(all_masses). Default ``None``.
    reference_potential: :class:`dict`
        Reference potential to add to the MLIP.
        If None, the MLIP will fit the difference between the
        true and reference potential.
        Needs to be a dict with parameters for LAMMPS calculator.
        At least a pair_coeff and a pair_style are needed in the dictionary.
        Default ```None``
    bonds: :class:`np.array`
        Numpy array of bonds list
    angles: :class:`np.array`
        Numpy array of angles list
    """
    def __init__(self,
                 elements,
                 masses,
                 Z,
                 rcut=5.0,
                 model="linear",
                 style="snap",
                 twojmax=8,
                 lmax=4,
                 nmax=4,
                 alpha=2.0,
                 chemflag=0,
                 radelems=None,
                 welems=None,
                 reference_potential=None,
                 fit_dielectric=False,
                 bonds=None,
                 angles=None):

        # Store parameters
        self.elements = np.array(elements)
        self.masses = masses
        self.Z = Z
        self.rcut = rcut
        self.model = model
        self.style = style
        self.twojmax = twojmax
        self.lmax = lmax
        self.nmax = nmax
        self.alpha = alpha
        self.chemflag = chemflag
        self.fit_dielectric = fit_dielectric
        self.prepare_ref_pot(reference_potential)
        if bonds is None:
            self.bonds = None
        else:
            self.bonds = bonds
        if angles is None:
            self.angles = None
        else:
            self.angles = angles
        if self.chemflag == 1:
            self.bnormflag = 1

        if radelems is None:
            self.radelems = np.array([0.5 for i in self.elements])
        else:
            self.radelems = radelems
        if welems is None:
            self.welems = np.array(self.Z) / np.sum(self.Z)
            if style == "so3":
                self.welems /= np.min(self.welems)
        else:
            self.welems = welems

        # Initialize the descriptor dimension, depending on the descriptor
        if self.style == "so3":
            self.ndescriptors = \
                int(self.nmax * (self.nmax + 1) * (self.lmax + 1) / 2)
        elif self.style == "snap":
            if self.twojmax % 2 == 0:
                m = 0.5 * self.twojmax + 1
                self.ndescriptors = int(m * (m+1) * (2*m+1) / 6)
            else:
                m = 0.5 * (self.twojmax + 1)
                self.ndescriptors = int(m * (m+1) * (m+2) / 3)
            if self.chemflag == 1:
                self.ndescriptors *= len(self.elements)**3
        if self.model == "quadratic":
            self.ndesc_linear = self.ndescriptors
            self.ndescriptors += \
                int(self.ndescriptors * (self.ndescriptors + 1) / 2)

        self.ncolumns = int(len(self.elements) * (self.ndescriptors + 1))
        if self.fit_dielectric:
            self.ncolumns += 1
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

# ========================================================================== #
    def _write_lammps_input(self):
        '''
        Write the LAMMPS input to extract the descriptor
        and gradient value needed to fit
        '''
        if self.bond_style is not None:
            bond_style = f"bond_style         {self.bond_style}\n"
            bond_coeff = ""
            for bc in self.bond_coeff:
                bond_coeff += f"bond_coeff       {bc}\n"
        else:
            bond_style = ""
            bond_coeff = ""
        if self.angle_style is not None:
            angle_style = f"angle_style       {self.angle_style}\n"
            angle_coeff = ""
            for angc in self.angle_coeff:
                angle_coeff += f"angle_coeff      {angc}\n"
        else:
            angle_style = ""
            angle_coeff = ""

        pair_style = f"pair_style         {self.pair_style}  " + \
                     f"zero {self.rcut*2}\n"

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
        input_string += bond_style
        input_string += bond_coeff
        input_string += angle_style
        input_string += angle_coeff
        input_string += pair_style
        input_string += pair_coeff
        input_string += model_post

        input_string += "thermo         100\n"
        input_string += "timestep       0.005\n"
        input_string += "neighbor       1.0 bin\n"
        input_string += "neigh_modify   once no every 1 delay 0 check yes\n"

        if self.style == "snap":
            input_string += "compute          mliap all mliap  descriptor " + \
                            "sna MLIP.mliap.descriptor  model " + \
                            self.model + "\n"
        elif self.style == "so3":
            input_string += "compute          mliap all mliap  descriptor " + \
                            "so3 MLIP.mliap.descriptor  model " + \
                            self.model + " gradgradflag 1\n"
        input_string += "fix          mliap all ave/time 1 1 1 c_mliap[*] " + \
                        "file descriptor.out mode vector format %15.5f  \n"

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
        with open("MLIP.mliap.descriptor", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("MLIP parameters\n")
            f.write("# Descriptor:  " + self.style + "\n")
            f.write("# Model:       " + self.model + "\n")
            if self.pair_coeff is not None:
                f.write("# Fitted with a reference potential\n")
                f.write("# See the mliap.model file to see the parameters\n")
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

# ========================================================================== #
    def write_mlip_coeff(self, coefficients):
        """
        Function to write the mliap.model parameter files of the MLIP
        """
        with open("MLIP.mliap.model", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            # One line to tel what are the parameters of the MLIP
            f.write("MLIP parameters\n")
            f.write("# Descriptor:  " + self.style + "\n")
            f.write("# Model:       " + self.model + "\n")
            # If there is a reference potential
            # add some lines showing the LAMMPS parameters
            pair_style, pair_coeff, model_post = \
                self.get_pair_coeff_and_style(coefficients[-1])
            atom_style, bond_style, bond_coeff, angle_style, angle_coeff = \
                self.get_bond_angle_coeff_and_style()

            f.write("# Parameters to be used in LAMMPS :\n")
            f.write(f"#bond_style     {bond_style}\n")
            for bc in bond_coeff:
                f.write(f"#bond_coeff    {bc}\n")
            f.write(f"#angle_style     {angle_style}\n")
            for angc in bond_coeff:
                f.write(f"#angle_coeff    {angc}\n")
            f.write(f"# pair_style    {pair_style}\n")
            for pc in pair_coeff:
                f.write(f"# pair_coeff   {pc}\n")
            if model_post is not None:
                for mp in model_post:
                    f.write(f"# {mp}")
            f.write(f"# atom_style   {self.atom_style}\n")

            f.write("\n")
            f.write("# nelems   ncoefs\n")
            f.write(f"{len(self.elements)}  {self.ndescriptors+1}\n")
            for icoef in range(len(coefficients-1)):
                f.write("{:35.30f}\n".format(coefficients[icoef]))

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

        if self.bonds is not None or self.angles is not None:
            write_lammps_data_full(lmp_atoms_fname,
                                   atoms,
                                   self.bonds,
                                   self.angles)
        else:
            write_lammps_data(lmp_atoms_fname,
                              atoms,
                              atom_style=self.atom_style)
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

        amatrix[:] = data_bispectrum

        self.cleanup()

        return amatrix, data_true

# ========================================================================== #
    def load_mlip(self, dielectric=None):
        '''
        Function to load a MLIP model
        Return a LAMMPSRUN calculator from the ASE package
        '''
        cwd = os.getcwd()
        if self.style == "snap":
            style = "sna"
        elif self.style == "so3":
            style = "so3"
        pair_style_mliap = f" mliap  model {self.model} " + \
                           f"{cwd}/MLIP.mliap.model  descriptor " + \
                           f"{style} {cwd}/MLIP.mliap.descriptor"
        pair_coeff_mliap = " ".join(self.elements)

        pair_style = self.pair_style + pair_style_mliap
        pair_coeff = []
        if self.pair_coeff is None:
            pair_coeff = [f"* * {pair_coeff_mliap}"]
        else:
            pair_coeff.append(f"* * mliap {pair_coeff_mliap}")
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
                pair_coeff = ["* * mliap " + " ".join(self.elements)]
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

        if self.style == "snap":
            style = "sna"
        elif self.style == "so3":
            style = "so3"

        pair_style_mliap = f" mliap  model {self.model} " + \
                           f"{cwd}/MLIP.mliap.model  descriptor " + \
                           f"{style} {cwd}/MLIP.mliap.descriptor "
        pair_coeff_mliap = " ".join(self.elements) + " "

        pair_style = self.pair_style + pair_style_mliap
        pair_coeff = []
        if self.pair_coeff is None:
            pair_coeff = [f"* * {pair_coeff_mliap}"]
        else:
            pair_coeff.append(f"* * mliap {pair_coeff_mliap}")
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
                pair_coeff = ["* * mliap " + " ".join(self.elements)]
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
    def get_bond_angle_coeff_and_style(self):
        """
        """
        atom_style = self.atom_style
        bond_style = self.bond_style
        bond_coeff = []
        if self.bond_coeff is None:
            bond_coeff = [""]
        else:
            for bc in self.bond_coeff:
                bond_coeff.append(bc)

        angle_style = self.angle_style
        angle_coeff = []
        if self.angle_coeff is None:
            angle_coeff = [""]
        else:
            for angc in self.angle_coeff:
                angle_coeff.append(angc)
        return atom_style, bond_style, bond_coeff, angle_style, angle_coeff

# ========================================================================== #
    def get_bonds_angles(self):
        """
        """
        bonds = self.bonds
        angles = self.angles
        return bonds, angles

# ========================================================================== #
    def prepare_ref_pot(self, ref_pot):
        """
        """
        if ref_pot is None:
            self.bond_style = None
            self.bond_coeff = None
            self.angle_style = None
            self.angle_coeff = None
            self.pair_style = ""
            self.pair_coeff = None
            self.model_post = None
            self.atom_style = "atomic"
        else:
            self.bond_style = ref_pot.get("bond_style", None)
            self.bond_coeff = ref_pot.get("bond_coeff", None)
            self.angle_style = ref_pot.get("angle_style", None)
            self.angle_coeff = ref_pot.get("angle_coeff", None)
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

        if self.bond_style is not None:
            ref_bs = self.bond_style
            ref_bc = self.bond_coeff
            if isinstance(ref_bs, str):
                ref_bs = [ref_bs]

            self.bond_style = "hybrid  "
            self.bond_style += "   ".join(ref_bs)
            self.bond_coeff = []

            for i, bs in enumerate(ref_bs):
                # bss = bs.split()[0] # pair_style
                bc = ref_bc[i]
                if isinstance(bc, str):
                    bcs = bc.split()  # pair_coeff splitted
                    self.bond_coeff.append(" ".join(bcs[:]))

                else:
                    for b in bc:
                        bcs = b.split()
                        self.bond_coeff.append(" ".join(bcs[:]))

        if self.angle_style is not None:
            ref_angs = self.angle_style
            ref_angc = self.angle_coeff
            # if isinstance(ref_angs, str):
            # ref_as = [ref_angs]
            self.angle_style = "hybrid  "
            self.angle_style += "   ".join(ref_angs)
            self.angle_coeff = []

            for i, angs in enumerate(ref_angs):
                # angss = angs.split()[0] # pair_style
                angc = ref_angc[i]
                if isinstance(angc, str):
                    angcs = angc.split()  # pair_coeff splitted
                    self.angle_coeff.append(" ".join(angcs[0:]))
                else:
                    for ang in angc:
                        angcs = ang.split()
                        self.angle_coeff.append(" ".join(angcs[0:]))

        if self.fit_dielectric:
            # self.kspace = "pppm 1.0e-5"
            self.kspace = "ewald 1e-6"
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
        mlip_dct = {"style": self.style,
                    "rcut": self.rcut,
                    "twojmax": self.twojmax,
                    "lmax": self.lmax,
                    "nmax": self.nmax,
                    "alpha": self.alpha,
                    "chemflag": self.chemflag,
                    "model": self.model,
                    "ncoef": self.ncolumns}
        return mlip_dct
