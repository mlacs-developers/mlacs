import os
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
from ase.io.lammpsdata import write_lammps_data

from ..utilities import get_elements_Z_and_masses, subfolder
from .descriptor import Descriptor, combine_reg
from ..utilities.io_lammps import LammpsInput, LammpsBlockInput


default_snap = {"twojmax": 8,
                "rfac0": 0.99363,
                "rmin0": 0.0,
                "switchflag": 1,
                "bzeroflag": 1,
                "wselfallflag": 0}

default_so3 = {"nmax": 4,
               "lmax": 4,
               "alpha": 1.0}


# ========================================================================== #
# ========================================================================== #
class MliapDescriptor(Descriptor):
    """
    Interface to the MLIAP potential of LAMMPS.

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Reference structure, with the elements for the descriptor

    rcut: :class:`float`
        The cutoff of the descriptor, in angstrom
        Default 5.0

    parameters: :class:`dict`
        A dictionnary of parameters for the descriptor input

        If the `style` is set to `snap`, then the default values are
            - twojmax = 8
            - rfac0 = 0.99363
            - rmin0 = 0.0
            - switchflag = 1
            - bzeroflag = 1
            - wselfallflag = 0

        If the `style` is set to `so3`, then the default values are
            - nmax = 4
            - lmax = 4
            - alpha = 1.0

    model: :class:`str`
        The type of model use. Can be either 'linear' or 'quadratic'
        Default `linear`

    style: :class:`str`
        The style of the descriptor used. Can be either 'snap' or 'so3'
        Default 'snap'

    alpha: :class:`float`
        The multiplication factor to the regularization parameter for
        ridge regression.
        Default 1.0
    """
    def __init__(self, atoms, rcut=5.0, parameters={},
                 model="linear", style="snap", alpha=1.0):
        self.chemflag = parameters.pop("chemflag", False)
        Descriptor.__init__(self, atoms, rcut, alpha)

        self.model = model
        self.style = style

        # Initialize the parameters for the descriptors
        self.radelems = parameters.pop("radelems", None)
        if self.radelems is None:
            self.radelems = np.array([0.5 for i in self.elements])
        self.welems = parameters.pop("welems", None)
        if self.welems is None:
            self.welems = np.array(self.Z) / np.sum(self.Z)
        if self.style == "snap":
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
        elif self.style == "so3":
            self.params = default_so3
            self.params.update(parameters)
            nmax = self.params["nmax"]
            lmax = self.params["lmax"]
            self.welems /= self.welems.min()
            self.ndesc = int(nmax * (nmax + 1) * (lmax + 1) / 2)
        if self.model == "quadratic":
            self.ndesc += int(self.ndesc * (self.ndesc + 1) / 2)
        self.ncolumns = int(self.nel * (self.ndesc + 1))

        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd

# ========================================================================== #
    def compute_descriptor(self, atoms, forces=True, stress=True):
        """
        """
        nat = len(atoms)
        el, z, masses, charges = get_elements_Z_and_masses(atoms)

        lmp_atfname = "atoms.lmp"
        self._write_lammps_input(masses, atoms.get_pbc())
        self._write_mlip_params()

        amat_e = np.zeros((1, self.ncolumns))
        amat_f = np.zeros((3 * nat, self.ncolumns))
        amat_s = np.zeros((6, self.ncolumns))

        write_lammps_data(lmp_atfname,
                          atoms,
                          specorder=self.elements.tolist())
        self._run_lammps(lmp_atfname)

        bispectrum = np.loadtxt("descriptor.out",
                                skiprows=4)
        bispectrum[-6:, 1:-1] /= -atoms.get_volume()

        amat_e[0] = bispectrum[0, 1:-1]
        amat_f = bispectrum[1:3*nat+1, 1:-1]
        amat_s = bispectrum[3*nat+1:, 1:-1]

        self.cleanup()
        res = dict(desc_e=amat_e,
                   desc_f=amat_f,
                   desc_s=amat_s)
        return res

# ========================================================================== #
    def _write_lammps_input(self, masses, pbc):
        """
        """
        txt = "LAMMPS input file for extracting MLIP descriptors"
        lmp_in = LammpsInput(txt)

        block = LammpsBlockInput("init", "Initialization")
        block("clear", "clear")
        pbc_txt = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))
        block("boundary", f"boundary {pbc_txt}")
        block("atom_style", "atom_style  atomic")
        block("units", "units metal")
        block("read_data", "read_data atoms.lmp")
        for i, m in enumerate(masses):
            block(f"mass{i}", f"mass   {i+1} {m}")
        lmp_in("init", block)

        block = LammpsBlockInput("interaction", "Interactions")
        block("pair_style", f"pair_style zero {2*self.rcut}")
        block("pair_coeff", "pair_coeff  * *")
        lmp_in("interaction", block)

        block = LammpsBlockInput("fake_dynamic", "Fake dynamic")
        block("thermo", "thermo 100")
        block("timestep", "timestep 0.005")
        block("neighbor", "neighbor 1.0 bin")
        block("neigh_modify", "neigh_modify once no every 1 delay 0 check yes")
        lmp_in("fake_dynamic", block)

        block = LammpsBlockInput("compute", "Compute")
        if self.style == "snap":
            style = "sna"
        elif self.style == "so3":
            style = "so3"
        txt = f"compute ml all mliap descriptor {style} MLIP.descriptor " + \
              f"model {self.model}"
        block("compute", txt)
        block("fix", "fix ml all ave/time 1 1 1 c_ml[*] " +
              "file descriptor.out mode vector")
        block("run", "run 0")
        lmp_in("compute", block)

        with open("lammps_input.in", "w") as fd:
            fd.write(str(lmp_in))

# ========================================================================== #
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the descriptor and gradient values
        '''
        lmp_cmd = f"{self.cmd} -in lammps_input.in -log none -sc lmp.out"
        lmp_handle = run(lmp_cmd,
                         shell=True,
                         stderr=PIPE)

        # There is a bug in LAMMPS that makes compute_mliap crashes at the end
        if lmp_handle.returncode != 0:
            pass
            """
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)
            """

# ========================================================================== #
    def cleanup(self):
        '''
        Function to cleanup the LAMMPS files used
        to extract the descriptor and gradient values
        '''
        Path("lmp.out").unlink()
        Path("descriptor.out").unlink()
        Path("lammps_input.in").unlink()
        Path("atoms.lmp").unlink()

# ========================================================================== #
    @subfolder
    def _write_mlip_params(self):
        """
        Function to write the mliap.descriptor parameter files of the MLIP
        """
        self.mlip_desc = Path.cwd()
        with open("MLIP.descriptor", "w") as f:
            f.write(self.get_mlip_params())

# ========================================================================== #
    def get_mlip_params(self):
        s = ("# ")
        # Adding a commment line to know what elements are fitted here
        for elements in self.elements:
            s += ("{:} ".format(elements))
        s += ("MLIP parameters\n")
        s += (f"# Descriptor:  {self.style}\n")
        s += (f"# Model:       {self.model}\n")
        s += ("\n")
        s += (f"rcutfac         {self.rcut}\n")
        for key in self.params.keys():
            s += (f"{key:12}    {self.params[key]}\n")
        s += ("\n\n\n")
        s += (f"nelems      {self.nel}\n")
        s += ("elems       ")
        for n in range(len(self.elements)):
            s += (self.elements[n] + " ")
        s += ("\n")
        s += ("radelems   ")
        for n in range(len(self.elements)):
            s += (f" {self.radelems[n]}")
        s += ("\n")
        s += ("welems    ")
        for n in range(len(self.elements)):
            s += (f"  {self.welems[n]}")
        s += ("\n")

        if self.style == "snap" and self.chemflag:
            s += ("\n\n")
            s += ("chemflag     1\n")
            s += ("bnormflag    1\n")
        return s

# ========================================================================== #
    @subfolder
    def write_mlip(self, coefficients):
        """
        """
        if Path("MLIP.model").is_file():
            Path("MLIP.model").unlink()
        self.mlip_model = Path.cwd()
        with open("MLIP.model", "w") as fd:
            fd.write("# ")
            fd.write(" ".join(self.elements))
            fd.write(" MLIP parameters\n")
            fd.write(f"# Descriptor   {self.style}\n")
            fd.write("\n")

            fd.write("# nelems   ncoefs\n")
            fd.write(f"{self.nel} {self.ndesc + 1}\n")
            np.savetxt(fd, coefficients, fmt="%35.30f")
        return "MLIP.model"

# ========================================================================== #
    @subfolder
    def read_mlip(self):
        """
        Read MLIP parameters from a file.
        """
        fn = Path("MLIP.model")
        if not fn.is_file():
            raise FileNotFoundError(f"File {fn.absolute()} does not exist.")

        with open(fn, "r") as fd:
            lines = fd.readlines()

        coefs = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            line = line.split()
            if len(line) == 2:  # Consistency check: nel, ndesc+1
                assert int(line[0]) == self.nel, "The descriptor changed"
                assert int(line[1]) == self.ndesc+1, "The descriptor changed"
                continue
            coefs.append(float(line[0]))
        return coefs

# ========================================================================== #
    def _regularization_matrix(self):
        # no regularization for the intercept
        d2 = [np.zeros((self.nel, self.nel))]
        d2.append(np.eye(self.ncolumns - self.nel))
        return combine_reg(d2)

# ========================================================================== #
    def get_pair_style(self, folder):
        if self.style == "snap":
            style = "sna"
        elif self.style == "so3":
            style = "so3"
        modelfile = folder / "MLIP.model"
        descfile = folder / "MLIP.descriptor"
        pair_style = f"mliap model {self.model} {modelfile} " + \
                     f"descriptor {style} {descfile}"
        return pair_style

# ========================================================================== #
    def get_pair_coeff(self, folder=None):
        return [f"* * {' '.join(self.elements)}"]

# ========================================================================== #
    def get_pair_style_coeff(self, folder):
        return self.get_pair_style(folder), self.get_pair_coeff(folder)

# ========================================================================== #
    def __str__(self):
        txt = " ".join(self.elements)
        txt += f" {self.style} MLIAP descriptor,"
        txt += f" rcut = {self.rcut}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = f"{self.style} MLIAP descriptor\n"
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
