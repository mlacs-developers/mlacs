import os
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
from ase.io.lammpsdata import write_lammps_data

from ..utilities import get_elements_Z_and_masses
from .mliap_descriptor import default_snap
from .descriptor import Descriptor, combine_reg


# ========================================================================== #
# ========================================================================== #
class SnapDescriptor(Descriptor):
    """
    Interface to the SNAP potential of LAMMPS.

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Reference structure, with the elements for the descriptor

    rcut: :class:`float`
        The cutoff of the descriptor, in angstrom
        Default 5.0

    parameters: :class:`dict`
        A dictionnary of parameters for the descriptor input

        The default values are
            - twojmax = 8
            - rfac0 = 0.99363
            - rmin0 = 0.0
            - switchflag = 1
            - bzeroflag = 1
            - wselfallflag = 0

    model: :class:`str`
        The type of model use. Can be either 'linear' or 'quadratic'
        Default `linear`

    alpha: :class:`float`
        The multiplication factor to the regularization parameter for
        ridge regression.
        Default 1.0

    alpha_quad: :class:`float`
        A multiplication factor for the regularization that apply only to
        the quadratic component of the descriptor
        Default 1.0
    """
    def __init__(self, atoms, rcut=5.0, parameters=dict(),
                 model="linear", alpha=1.0, alpha_quad=1.0, folder="Snap"):
        self.chemflag = parameters.pop("chemflag", 0)
        Descriptor.__init__(self, atoms, rcut, alpha)
        self.alpha_quad = alpha_quad
        self.folder = Path(folder).absolute()
        self.get_pair_style_coeff()

        self.model = model

        # Initialize the parameters for the descriptors
        self.radelems = parameters.pop("radelems", None)
        if self.radelems is None:
            self.radelems = np.array([0.5 for i in self.elements])
        self.welems = parameters.pop("welems", None)
        if self.welems is None:
            self.welems = np.array(self.Z) / np.sum(self.Z)
        self.params = default_snap
        self.params.update(parameters)
        if self.chemflag:
            self.params["bnormflag"] = 1
        twojmax = self.params["twojmax"]
        if twojmax % 2 == 0:
            m = 0.5 * twojmax + 1
            self.ndesc = int(m * (m+1) * (2*m+1) / 6)
        else:
            m = 0.5 * (twojmax + 1)
            self.ndesc = int(m * (m+1) * (m+2) / 3)
        if self.chemflag:
            self.ndesc *= self.nel**3
        if self.model == "quadratic":
            self.ndesc_quad = int(self.ndesc * (self.ndesc + 1) / 2)
            self.ndesc_lin = self.ndesc
            self.ndesc += int(self.ndesc * (self.ndesc + 1) / 2)
        self.ncolumns = int(self.nel * (self.ndesc + 1))

        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd

# ========================================================================== #
    def _compute_descriptor(self, atoms, forces=True, stress=True):
        """
        """
        self.folder.mkdir(parents=True, exist_ok=True)

        nat = len(atoms)
        el, z, masses, charges = get_elements_Z_and_masses(atoms)
        chemsymb = np.array(atoms.get_chemical_symbols())

        lmp_atfname = self.folder / "atoms.lmp"
        self._write_lammps_input(masses, atoms.get_pbc())
        self._write_mlip_params()

        amat_e = np.zeros((1, self.ncolumns))
        amat_f = np.zeros((3 * nat, self.ncolumns))
        amat_s = np.zeros((6, self.ncolumns))

        write_lammps_data(lmp_atfname,
                          atoms,
                          specorder=self.elements.tolist())
        self._run_lammps(lmp_atfname)

        bispectrum = np.loadtxt(self.folder / "descriptor.out",
                                skiprows=4)
        bispectrum[-6:, 1:-1] /= -atoms.get_volume()

        amat_e[0, self.nel:] = bispectrum[0, 1:-1]
        amat_f[:, self.nel:] = bispectrum[1:3*nat+1, 1:-1]
        amat_s[:, self.nel:] = bispectrum[3*nat+1:, 1:-1]

        for i, el in enumerate(self.elements):
            amat_e[0, i] = np.count_nonzero(chemsymb == el)

        self.cleanup()
        res = dict(desc_e=amat_e,
                   desc_f=amat_f,
                   desc_s=amat_s)
        return res

# ========================================================================== #
    def _write_lammps_input(self, masses, pbc):
        """
        """
        input_string = "# LAMMPS input file for extracting MLIP descriptors\n"
        input_string += "clear\n"
        input_string += "boundary         "
        for ppp in pbc:
            if ppp:
                input_string += "p "
            else:
                input_string += "f "
        input_string += "\n"
        input_string += "atom_style      atomic\n"
        input_string += "units            metal\n"
        input_string += "read_data        atoms.lmp\n"
        for n1 in range(len(self.masses)):
            input_string += f"mass             {n1+1} {self.masses[n1]}\n"

        input_string += f"pair_style       zero {2*self.rcut}\n"
        input_string += "pair_coeff       * *\n"

        input_string += "thermo         100\n"
        input_string += "timestep       0.005\n"
        input_string += "neighbor       1.0 bin\n"
        input_string += "neigh_modify   once no every 1 delay 0 check yes\n"

        input_string += f"compute      ml all snap {self._snap_opt_str()}\n"
        input_string += "fix          ml all ave/time 1 1 1 c_ml[*] " + \
                        "file descriptor.out mode vector\n"
        input_string += "run              0\n"

        with open(self.folder / "base.in", "w") as fd:
            fd.write(input_string)

# ========================================================================== #
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the descriptor and gradient values
        '''
        lammps_command = self.cmd + ' -in base.in -log none -sc lmp.out'
        lmp_handle = run(lammps_command,
                         shell=True,
                         stderr=PIPE,
                         cwd=self.folder)

        # There is a bug in LAMMPS that makes compute_mliap crashes at the end
        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    def cleanup(self):
        '''
        Function to cleanup the LAMMPS files used
        to extract the descriptor and gradient values
        '''
        (self.folder / "lmp.out").unlink()
        (self.folder / "descriptor.out").unlink()
        (self.folder / "base.in").unlink()
        (self.folder / "atoms.lmp").unlink()

# ========================================================================== #
    def _write_mlip_params(self):
        """
        Function to write the mliap.descriptor parameter files of the MLIP
        """
        with open(self.folder / "MLIP.descriptor", "w") as f:
            f.write("# ")
            # Adding a commment line to know what elements are fitted here
            for elements in self.elements:
                f.write("{:} ".format(elements))
            f.write("MLIP parameters\n")
            f.write("# Descriptor:  SNAP\n")
            f.write(f"# Model:       {self.model}\n")
            f.write("\n")
            f.write(f"rcutfac         {self.rcut}\n")
            for key in self.params.keys():
                f.write(f"{key:12}    {self.params[key]}\n")
            if self.chemflag:
                f.write("\n\n")
                f.write("chemflag     1\n")
                f.write("bnormflag    1\n")
            if self.model == "quadratic":
                f.write("quadraticflag  1")

# ========================================================================== #
    def write_mlip(self, coefficients, folder=None, comments=""):
        """
        """

        intercepts = coefficients[:self.nel]
        coefs = coefficients[self.nel:]

        with open(self.folder / "MLIP.model", "w") as fd:
            fd.write("# ")
            fd.write(" ".join(self.elements))
            fd.write(" MLIP parameters\n")
            fd.write("# Descriptor   SNAP\n")
            fd.write(comments)
            fd.write("\n")
            fd.write(f"{self.nel} {self.ndesc+1}\n")

            for iel in range(self.nel):
                iidx = iel * self.ndesc
                fidx = (iel + 1) * self.ndesc
                el = self.elements[iel]
                rel = self.radelems[iel]
                wel = self.welems[iel]
                fd.write(f"{el} {rel} {wel}\n")
                fd.write(f"{intercepts[iel]:35.30f}\n")
                np.savetxt(fd, coefs[iidx:fidx], fmt="%35.30f")

# ========================================================================== #
    def _regularization_matrix(self):
        # no regularization for the intercept
        d2 = [np.zeros((self.nel, self.nel))]
        if self.model == "linear":
            d2.append(np.eye(self.ndesc) * self.alpha)
        elif self.model == "quadratic":
            for i in range(self.nel):
                d2.append(np.eye(self.ndesc_lin) * self.alpha)
                d2.append(np.eye(self.ndesc_quad) * self.alpha_quad)

            print(d2)
        for d in d2:
            print(d.shape)
        return combine_reg(d2)

# ========================================================================== #
    def get_pair_style_coeff(self):
        """
        """
        modelfile = self.folder / "MLIP.model"
        descfile = self.folder / "MLIP.descriptor"
        pair_style = "snap"
        pair_coeff = [f"* * {modelfile}  {descfile} " +
                      ' '.join(self.elements)]
        return pair_style, pair_coeff

# ========================================================================== #
    def _snap_opt_str(self):
        params = self.params.copy()
        rfac0 = params.pop("rfac0")
        twojmax = params.pop("twojmax")
        radelems = [str(rad) for rad in self.radelems]
        welems = [str(wel) for wel in self.welems]
        quadflag = 0
        if self.model == "quadratic":
            quadflag = 1
        bnormflag = 0
        if self.chemflag:
            bnormflag = 1

        txt = f"{self.rcut} {rfac0} {twojmax} "
        txt += ' '.join(radelems) + " "
        txt += ' '.join(welems) + " "
        txt += f"quadraticflag {quadflag} "
        if self.chemflag:
            txt += f"chem {self.nel} "
            for i in range(self.nel):
                txt += f"{i} "
            txt += f"bnormflag {bnormflag} "
        for key, val in params.items():
            txt += f"{key} {val} "
        return txt

# ========================================================================== #
    def __str__(self):
        txt = " ".join(self.elements)
        txt += " SNAP descriptor,"
        txt += f" rcut = {self.rcut}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "SNAP descriptor\n"
        txt += f"{(len(txt) - 1) * '-'}\n"
        txt += "Elements :\n"
        txt += " ".join(self.elements) + "\n"
        txt += "Parameters :\n"
        txt += f"rcut                {self.rcut}\n"
        txt += f"chemflag            {self.chemflag}\n"
        for key, val in self.params.items():
            txt += f"{key:12}        {val}\n"
        txt += f"dimension           {self.ncolumns}\n"
        return txt