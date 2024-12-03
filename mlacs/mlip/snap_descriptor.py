"""
// Copyright (C) 2022-2024 MLACS group (AC, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import shutil
import shlex
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
from ase.atoms import Atoms
from ase.io.lammpsdata import write_lammps_data

from ..core.manager import Manager
from ..utilities import get_elements_Z_and_masses
from .mliap_descriptor import default_snap
from .descriptor import Descriptor, combine_reg
from ..utilities.io_lammps import (LammpsInput, LammpsBlockInput,
                                   get_lammps_command)


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

    Examples
    --------

    >>> from ase.build import bulk
    >>> atoms = bulk("Cu", cubic=True).repeat(2)
    >>>
    >>> from mlacs.mlip import SnapDescriptor, LinearPotential
    >>> desc = SnapDescriptor(atoms, rcut=4.2, parameters=dict(twojmax=6))
    >>> desc.compute_descriptor(atoms)
    """
    def __init__(self, atoms, rcut=5.0, parameters=dict(),
                 model="linear", alpha=1.0, alpha_quad=1.0, prefix='SNAP',
                 **kwargs):
        self.chemflag = parameters.pop("chemflag", 0)
        Descriptor.__init__(self, atoms, rcut, alpha, prefix=prefix, **kwargs)
        self.alpha_quad = alpha_quad
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

        self.cmd = get_lammps_command()

# ========================================================================== #
    def compute_descriptor(self, atoms, forces=True, stress=True):
        """
        """
        nat = len(atoms)
        el, _, _, _ = get_elements_Z_and_masses(atoms)
        chemsymb = np.array(atoms.get_chemical_symbols())

        lmp_atfname = "atoms.lmp"
        self._write_lammps_input(atoms.get_pbc())
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
    @Manager.exec_from_path
    def compute_descriptors(self, atoms, forces=True, stress=True):
        """
        Compute the descriptors of multiples structures in one Lammps launch
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if len(atoms) == 1:
            return super().compute_descriptors(atoms,
                                               forces=forces,
                                               stress=stress)
        if Path("Atoms").exists():
            shutil.rmtree("Atoms")
        Path("Atoms").mkdir()

        for i, at in enumerate(atoms):
            if np.any(at.get_pbc() != atoms[0].get_pbc()):
                raise ValueError("PBC cannot change between states")
            lmp_atfname = f"Atoms/atoms{i+1}.lmp"
            write_lammps_data(lmp_atfname,
                              at,
                              specorder=self.elements.tolist())
        self._write_mlip_params()
        self._write_looping_lammps_input(atoms)

        self._run_lammps(lmp_atfname)

        desc = []
        for idx in range(len(atoms)):
            nat = len(atoms[idx])
            chemsymb = np.array(atoms[idx].get_chemical_symbols())

            amat_e = np.zeros((1, self.ncolumns))
            amat_f = np.zeros((3 * nat, self.ncolumns))
            amat_s = np.zeros((6, self.ncolumns))

            bispectrum = np.loadtxt(f"descriptor{idx+1}.out",
                                    skiprows=4)
            bispectrum[-6:, 1:-1] /= -atoms[idx].get_volume()
            amat_e[0, self.nel:] = bispectrum[0, 1:-1]
            amat_f[:, self.nel:] = bispectrum[1:3*nat+1, 1:-1]
            amat_s[:, self.nel:] = bispectrum[3*nat+1:, 1:-1]

            for i, el in enumerate(self.elements):
                amat_e[0, i] = np.count_nonzero(chemsymb == el)

            desc.append(dict(desc_e=amat_e,
                             desc_f=amat_f,
                             desc_s=amat_s))
        self.loop_cleanup(len(atoms))
        return desc

# ========================================================================== #
    @Manager.exec_from_path
    def loop_cleanup(self, nconf):
        '''
        Function to cleanup the LAMMPS files used
        to extract the descriptor and gradient values
        '''
        Path("lmp.out").unlink(missing_ok=True)
        Path("lammps_input.in").unlink(missing_ok=True)
        shutil.rmtree("Atoms")
        for i in range(nconf):
            Path(f"descriptor{i+1}.out").unlink(missing_ok=True)

# ========================================================================== #
    @Manager.exec_from_path
    def _write_looping_lammps_input(self, atoms):
        """
        Write one lammps file for all the config in atoms
        """
        txt = "LAMMPS input file for extracting MLIP descriptors"
        lmp_in = LammpsInput(txt)

        block = LammpsBlockInput("init", "Initialization")
        block("loop", f"variable a loop {len(atoms)}")
        block("label", "label loopstart")

        pbc_txt = "{0} {1} {2}".format(
                *tuple("sp"[int(x)] for x in atoms[0].get_pbc()))
        block("boundary", f"boundary {pbc_txt}")
        block("atom_style", "atom_style  atomic")
        block("units", "units metal")
        block("read_data", "read_data Atoms/atoms${a}.lmp")
        for i, m in enumerate(self.masses):
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
        block("compute", f"compute ml all snap {self._snap_opt_str()}")
        block("fix", "fix ml all ave/time 1 1 1 c_ml[*] " +
              "file descriptor${a}.out mode vector")
        block("run", "run 0")
        lmp_in("compute", block)

        block = LammpsBlockInput("loopback", "Loop Back")
        block("clear data", "clear")
        block("iterate", "next a")
        block("loopend", "jump SELF loopstart")
        lmp_in("loopback", block)

        with open("lammps_input.in", "w") as fd:
            fd.write(str(lmp_in))

# ========================================================================== #
    def _write_lammps_input(self, pbc):
        """
        """
        txt = "LAMMPS input file for extracting SNAP descriptors"
        lmp_in = LammpsInput(txt)

        block = LammpsBlockInput("init", "Initialization")
        block("clear", "clear")
        pbc_txt = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))
        block("boundary", f"boundary {pbc_txt}")
        block("atom_style", "atom_style  atomic")
        block("units", "units metal")
        block("read_data", "read_data atoms.lmp")
        for i, m in enumerate(self.masses):
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
        block("compute", f"compute ml all snap {self._snap_opt_str()}")
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
        lmp_handle = run(shlex.split(lmp_cmd),
                         stderr=PIPE)

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
        Path("lmp.out").unlink()
        Path("descriptor.out").unlink()
        Path("lammps_input.in").unlink()
        Path("atoms.lmp").unlink()

# ========================================================================== #
    @Manager.exec_from_subdir
    def _write_mlip_params(self):
        """
        Function to write the mliap.descriptor parameter files of the MLIP
        """
        fname = f"{self.prefix}.descriptor"
        with open(fname, "w") as f:
            f.write(self.get_mlip_params())

# ========================================================================== #
    def get_mlip_params(self):
        s = ("# ")
        # Adding a commment line to know what elements are fitted here
        for elements in self.elements:
            s += ("{:} ".format(elements))
        s += ("MLIP parameters\n")
        s += ("# Descriptor:  SNAP\n")
        s += (f"# Model:       {self.model}\n")
        s += ("\n")
        s += (f"rcutfac         {self.rcut}\n")
        for key in self.params.keys():
            s += (f"{key:12}    {self.params[key]}\n")
        if self.chemflag:
            s += ("\n\n")
            s += ("chemflag     1\n")
            s += ("bnormflag    1\n")
        if self.model == "quadratic":
            s += ("quadraticflag  1")
        return s

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def write_mlip(self, coefficients):
        """
        """
        filepath = Path(self.get_filepath('.model'))
        if filepath.exists():
            filepath.unlink()
        fname = filepath.relative_to(self.subsubdir)

        intercepts = coefficients[:self.nel]
        coefs = coefficients[self.nel:]
        with open(filepath, "w") as fd:
            fd.write("# ")
            fd.write(" ".join(self.elements))
            fd.write(" MLIP parameters\n")
            fd.write("# Descriptor   SNAP\n")
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

        return fname

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def get_coef(self, filename=None):
        """
        Read MLIP coefficients from a file.
        """
        if filename:
            filename = Path(filename)
        else:
            filename = Path(self.get_filepath('.model'))

        if not filename.is_file():
            filename = filename.absolute()
            raise FileNotFoundError(f"File {filename} does not exist")

        with open(filename, "r") as fd:
            lines = fd.readlines()

        coefs = []
        coefficients = []
        intercepts = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            line = line.split()
            if len(line) == 2:  # Consistency check: nel, ndesc+1
                assert int(line[0]) == self.nel, "The descriptor changed"
                assert int(line[1]) == self.ndesc+1, "The descriptor changed"
                continue

            if len(line) == 3:  # el, radelems, welems
                continue
            if (len(coefs)+len(intercepts)) % (self.ndesc+1) == 0:
                intercepts.append(float(line[0]))
            else:
                coefs.append(float(line[0]))
        coefficients = np.r_[intercepts, coefs]
        return coefficients

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
        return combine_reg(d2)

# ========================================================================== #
    def get_pair_style(self):
        return "snap"

# ========================================================================== #
    def get_pair_coeff(self):
        modelfile = self.get_filepath('.model')
        descfile = self.subdir / f"{self.prefix}.descriptor"
        pair_coeff = [f"* * {modelfile}  {descfile} " +
                      ' '.join(self.elements)]
        return pair_coeff

# ========================================================================== #
    def get_pair_style_coeff(self):
        return self.get_pair_style(), self.get_pair_coeff()

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
