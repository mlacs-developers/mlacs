"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path
from subprocess import run, Popen, PIPE
from os import symlink
import shutil
import shlex

import numpy as np
from ase.calculators.lammpsrun import LAMMPS
from ase.units import GPa

from .descriptor import BlankDescriptor
from .mlip_manager import SelfMlipManager
from ..utilities import compute_correlation
from ..utilities.io import write_cfg, read_cfg_data


default_mtp_parameters = dict(level=8,
                              radial_basis_type="RBChebyshev",
                              min_dist=1.0,
                              max_dist=5.0,
                              radial_basis_size=8)

default_fit_parameters = dict(scale_by_forces=0,
                              max_iter=100,
                              bfgs_conv_tol=1e-3,
                              weighting="vibrations",
                              weight_scaling=None,
                              weight_scaling_forces=None,
                              init_params="random",
                              update_mindist=True)


# ========================================================================== #
# ========================================================================== #
class MomentTensorPotential(SelfMlipManager):
    """
    Interface to the Moment Tensor Potential of the MLIP package.

    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        Prototypical configuration for the MLIP. Should have the desired
        species.

    mlpbin: :class:`str`
        The path to the  `mlp` binary. If mpi is desired, the command
        should be set as 'mpirun /path/to/mlp'

    mtp_parameters: :class:`dict`
        The dictionnary with inputs for the potential.

        The default values are set to

        - level = 8
        - radial_basis_type = 'RBChebyshev'
        - min_dist=1.0,
        - max_dist=5.0,
        - radial_basis_size=8

    fit_parameters: :class:`dict`
        The parameters for the fit of the potential

        The default parameters are set to
            - scale_by_forces=0
            - max_iter=1000
            - bfgs_conv_tol=1e-3
            - weighting='vibrations'
            - (weight_scaling=1, weight_scaling_forces=0 for MLIP-3)
            - init_params='random'
            - update_mindist=False

    Examples
    --------

    >>> from ase.io import read
    >>> confs = read('Trajectory.traj', index=':')
    >>>
    >>> from mlacs.mlip import MomentTensorPotential
    >>> mlip = MomentTensorPotential(confs[0], mtp_parameters=dict(level=6))
    >>> mlip.update_matrices(confs)
    >>> mlip.train_mlip()
    """
    def __init__(self,
                 atoms,
                 mlpbin="mlp",
                 mtp_parameters={},
                 fit_parameters={},
                 folder='MTP',
                 **kwargs):

        self.cmd = mlpbin
        self.version = 2
        testcmd = shlex.split(f'{mlpbin} list')
        mlp_info = Popen(testcmd, stdout=PIPE).communicate()[0]
        if b'calculate_efs' in mlp_info:
            self.version = 3

        SelfMlipManager.__init__(self,
                                 descriptor=BlankDescriptor(atoms),
                                 weight=None,
                                 folder=folder,
                                 **kwargs)

        self.level = mtp_parameters.pop("level", 8)
        if self.level % 2 or self.level > 28:
            msg = "Only even number between 2 and 28 are available as level"
            raise ValueError(msg)
        self.mtp_parameters = default_mtp_parameters
        for key in mtp_parameters.keys():
            if key not in list(default_mtp_parameters.keys()):
                msg = f"{key} is not a parameter for the MTP potential"
                raise ValueError(msg)
        self.mtp_parameters.update(mtp_parameters)

        self.fit_parameters = default_fit_parameters
        self.fit_parameters.update(fit_parameters)

# ========================================================================== #
    def read_parent_mlip(self, traj):
        """
        Get a list of all the mlip that have generated a conf in traj
        and get the coefficients of all these mlip
        """
        parent_mlip = []
        mlip_coef = []
        # Make the MBAR variable Nk and mlip_coef
        for state in traj:
            for conf in state:
                if "parent_mlip" not in conf.info:  # Initial or training
                    continue
                else:  # A traj
                    model = conf.info['parent_mlip']
                    if not Path(model).exists:
                        err = "Some parent MLIP are missing. "
                        err += "Rerun MLACS with DatabaseCalculator and "
                        err += "OtfMlacs.keep_tmp_files=True on your traj"
                        raise FileNotFoundError(err)
                    if model not in parent_mlip:  # New state
                        parent_mlip.append(model)
                        coef = model
                        mlip_coef.append(coef)
        return parent_mlip, np.array(mlip_coef)

# ========================================================================== #
    def next_coefs(self, mlip_coef):
        """
        Update MLACS just like train_mlip, but without actually computing
        the coefficients
        """
        pass

# ========================================================================== #
    def get_pair_style(self):
        if self.version == 3:
            return f"mlip load_from={self.subdir / 'pot.mtp'}"
        return f"mlip {self.subdir / 'mlip.ini'}"

# ========================================================================== #
    def get_pair_coeff(self):
        return ["* *"]

# ========================================================================== #
    def get_pair_style_coeff(self):
        return self.get_pair_style(), \
               self.get_pair_coeff()

# ========================================================================== #
    def train_mlip(self):
        """
        """
        # We need to remove the old subfolder. If calculation ended at
        # some specific weird moments, it can cause access problem
        # for the mlp binary otherwise
        if self.subfolder:
            if self.subsubdir.exists():
                shutil.rmtree(self.subsubdir)

        self.subsubdir.mkdir(parents=True, exist_ok=True)

        mtpfile = self.subdir / "pot.mtp"

        # Move old potential in new folder, to start BFGS from
        # previously trained MTP
        if mtpfile.exists():
            shutil.move(mtpfile, self.subsubdir / "pot.mtp")

        self._clean_folder()
        self._write_configurations()
        self._write_input()
        self._write_mtpfile()
        self._run_mlp()

        # Symlink new MTP in the main folder
        if self.subfolder:
            # We might need to remove the old symlink before the new one
            if mtpfile.exists():
                mtpfile.unlink()

            src = self.subsubdir / "pot.mtp"
            symlink(src, self.subdir / "pot.mtp")

        with open(self.subdir / "mlip.ini", "w") as fd:
            fd.write(f"mtp-filename    {mtpfile}\n")
            fd.write("select           FALSE")

        msg = "number of configurations for training: " + \
              f"{len(self.natoms):}\n"
        msg += "number of atomic environments for training: " + \
               f"{self.natoms.sum():}\n"
        msg += self._compute_test(msg)

        if self.log:
            self.log.write(msg)

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        calc = LAMMPS(pair_style=self.pair_style,
                      pair_coeff=self.pair_coeff,
                      atom_style=self.atom_style,
                      keep_alive=False)
        if self.model_post is not None:
            calc.set(model_post=self.model_post)
        return calc

# ========================================================================== #
    def _clean_folder(self):
        """
        """
        files = ["train.cfg",
                 "mlip.ini",
                 "initpot.mtp",
                 "out.cfg"]
        for fn in files:
            if (filepath := self.subsubdir / fn).exists():
                filepath.unlink()

# ========================================================================== #
    def _write_configurations(self):
        """
        """
        confs = self.configurations
        chemmap = self.descriptor.elements
        write_cfg(self.subsubdir / "train.cfg", confs, chemmap)

# ========================================================================== #
    def _write_input(self):
        """
        """
        mtpfile = self.subsubdir / "pot.mtp"
        with open(self.subsubdir / "mlip.ini", "w") as fd:
            fd.write(f"mtp-filename    {mtpfile}\n")
            fd.write("select           FALSE")

# ========================================================================== #
    def _write_mtpfile(self):
        """
        """
        writenewmtp = True
        mtpfile = self.subsubdir / "initpot.mtp"
        lvl = self.level
        level = f"level{lvl}"
        if (potfile := self.subsubdir / "pot.mtp").exists():
            import re
            with open(potfile, "r", encoding="ISO-8859-1") as fd:
                for line in fd.readlines():
                    if line.startswith("potential_name"):
                        oldlevel = int(re.search(r'\d+$', line).group())
                        break
            if oldlevel == lvl:
                potfile.rename(mtpfile)
                writenewmtp = False
        if writenewmtp:
            from . import _mtp_data
            leveltxt = getattr(_mtp_data, level)
            nel = self.descriptor.nel
            btype = self.mtp_parameters["radial_basis_type"]
            min_dist = self.mtp_parameters["min_dist"]
            max_dist = self.mtp_parameters["max_dist"]
            bsize = self.mtp_parameters["radial_basis_size"]
            with open(mtpfile, "w") as fd:
                fd.write("MTP\n")
                fd.write("version = 1.1.0\n")
                fd.write(f"potential_name = MTP-{level}\n")
                fd.write(f"species_count = {nel}\n")
                fd.write("potential_tag = \n")
                fd.write(f"radial_basis_type = {btype}\n")
                fd.write(f"min_dist = {min_dist}\n")
                fd.write(f"max_dist = {max_dist}\n")
                fd.write(f"radial_basis_size = {bsize}\n")
                fd.write(leveltxt)

# ========================================================================== #
    def _run_mlp(self):
        """
        """
        if self.version == 3:
            mlp_command = self._get_cmd_mlip3()
        else:
            mlp_command = self._get_cmd_mlip2()
        with open(self.subsubdir / "mlip.log", "w") as fd:
            mlp_handle = run(mlp_command.split(),
                             stderr=PIPE,
                             stdout=fd,
                             cwd=self.subsubdir)
        if mlp_handle.returncode != 0:
            msg = "mlp stopped with the exit code \n" + \
                  f"{mlp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    def _get_cmd_mlip2(self):
        """
        Get mlp command from MLIP-2 package.
        """
        initpotfile = self.subsubdir / "initpot.mtp"
        potfile = self.subsubdir / "pot.mtp"
        trainfile = self.subsubdir / "train.cfg"
        cmd = self.cmd + f" train {initpotfile} {trainfile}"
        cmd += f" --trained-pot-name={potfile}"
        up_mindist = self.fit_parameters["update_mindist"]
        if up_mindist:
            cmd += " --update-mindist"
        init_params = self.fit_parameters["init_params"]
        cmd += f" --init-params={init_params}"
        max_iter = self.fit_parameters["max_iter"]
        cmd += f" --max-iter={max_iter}"
        bfgs_conv_tol = self.fit_parameters["bfgs_conv_tol"]
        cmd += f" --bfgs-conv-tol={bfgs_conv_tol}"
        scale_by_forces = self.fit_parameters["scale_by_forces"]
        cmd += f" --scale-by-force={scale_by_forces}"
        cmd += f" --energy-weight={self.weight.energy_coefficient}"
        cmd += f" --force-weight={self.weight.forces_coefficient}"
        cmd += f" --stress-weight={self.weight.stress_coefficient}"
        weighting = self.fit_parameters["weighting"]
        cmd += f" --weighting={weighting}"
        return cmd

# ========================================================================== #
    def _get_cmd_mlip3(self):
        """
        Get mlp command from MLIP-3 package.
        """
        initpotfile = self.subsubdir / "initpot.mtp"
        potfile = self.subsubdir / "pot.mtp"
        trainfile = self.subsubdir / "train.cfg"
        cmd = self.cmd + f" train {initpotfile} {trainfile}"
        cmd += f" --save_to={potfile}"
        # Default in MLIP-3 is mindist update, which is the opposite of MLIP-2.
        # Need to check this ...
        up_mindist = not self.fit_parameters["update_mindist"]
        cmd += f" --no_mindist_update={up_mindist}"
        if self.fit_parameters["init_params"] == "random":
            cmd += " --init_random=True"
        max_iter = self.fit_parameters["max_iter"]
        cmd += f" --iteration_limit={max_iter}"
        bfgs_conv_tol = self.fit_parameters["bfgs_conv_tol"]
        cmd += f" --tolerance={bfgs_conv_tol}"
        scale_by_forces = self.fit_parameters["scale_by_forces"]
        cmd += f" --scale_by_force={scale_by_forces}"
        cmd += f" --energy_weight={self.weight.energy_coefficient}"
        cmd += f" --force_weight={self.weight.forces_coefficient}"
        cmd += f" --stress_weight={self.weight.stress_coefficient}"
        w_scale = self.fit_parameters["weight_scaling"]
        w_f_scale = self.fit_parameters["weight_scaling_forces"]
        if w_scale is not None or w_f_scale is not None:
            if w_scale is not None:
                cmd += f" --weight_scaling={w_scale}"
            if w_f_scale is not None:
                cmd += f" --weight_scaling_forces={w_f_scale}"
            return cmd
        weighting = self.fit_parameters["weighting"]
        if weighting == "vibrations":
            cmd += " --weight_scaling=1 --weight_scaling_forces=0"
        elif weighting == "molecules":
            cmd += " --weight_scaling=0 --weight_scaling_forces=0"
        elif weighting == "structures":
            cmd += " --weight_scaling=2 --weight_scaling_forces=1"
        return cmd

# ========================================================================== #
    def _get_pair_style(self):
        return self.get_pair_style()

# ========================================================================== #
    def _get_pair_coeff(self):
        return self.get_pair_coeff()

# ========================================================================== #
    def predict(self, atoms, coef=None):
        msg = "Getting the MLIP energy from descriptor is not accessible " + \
              "for SelfMlipManager"
        raise NotImplementedError(msg)

# ========================================================================== #
    def _run_test(self):
        """
        """
        trainfile = self.subsubdir / "train.cfg"
        outfile = self.subsubdir / "out.cfg"
        potfile = self.subsubdir / "pot.mtp"
        mlp_command = self.cmd
        if self.version == 3:
            mlp_command += f" calculate_efs {potfile} {trainfile}"
            mlp_command += f" --output_filename={outfile}"
            outfile = self.subsubdir / "out.cfg.0"
        else:
            mlp_command += f" calc-efs {potfile} {trainfile} {outfile}"
        mlp_handle = run(mlp_command.split(),
                         stderr=PIPE)
        if mlp_handle.returncode != 0:
            msg = "mlp stopped with the exit code \n" + \
                  f"{mlp_handle.stderr.decode()}"
            raise RuntimeError(msg)
        e_mlip, f_mlip, s_mlip = read_cfg_data(outfile)
        return e_mlip, f_mlip, s_mlip

# ========================================================================== #
    def _compute_test(self, msg):
        """
        """
        e_mlip, f_mlip, s_mlip = self._run_test()

        confs = self.configurations
        e_dft = np.array([at.get_potential_energy() / len(at)for at in confs])
        f_dft = []
        s_dft = []
        for at in confs:
            f_dft.extend(at.get_forces().flatten())
            s_dft.extend(at.get_stress())

        rmse_e, mae_e, rsq_e = compute_correlation(np.c_[e_dft, e_mlip])
        rmse_f, mae_f, rsq_f = compute_correlation(np.c_[f_dft, f_mlip])
        rmse_s, mae_s, rsq_s = compute_correlation(np.c_[s_dft, s_mlip] / GPa)

        # Prepare message to the log
        msg += f"RMSE Energy    {rmse_e:.4f} eV/at\n"
        msg += f"MAE Energy     {mae_e:.4f} eV/at\n"
        msg += f"RMSE Forces    {rmse_f:.4f} eV/angs\n"
        msg += f"MAE Forces     {mae_f:.4f} eV/angs\n"
        msg += f"RMSE Stress    {rmse_s:.4f} GPa\n"
        msg += f"MAE Stress     {mae_s:.4f} GPa\n"
        msg += "\n"

        header = f"rmse: {rmse_e:.5f} eV/at,    " + \
                 f"mae: {mae_e:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.c_[e_dft, e_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_f:.5f} eV/angs   " + \
                 f"mae: {mae_f:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.c_[f_dft, f_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_s:.5f} GPa       " + \
                 f"mae: {mae_s:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat",
                   np.c_[s_dft, s_mlip] / GPa,
                   header=header, fmt="%25.20f  %25.20f")
        return msg

# ========================================================================== #
    def __str__(self):
        txt = f"Moment Tensor Potential, level = {self.level}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "Moment Tensor Potential\n"
        txt += "Parameters:\n"
        txt += "-----------\n"
        txt += "Descriptor:\n"
        txt += "-----------\n"
        txt += f"level :                 {self.level}\n"
        basis = self.mtp_parameters["radial_basis_type"]
        nbasis = self.mtp_parameters["radial_basis_size"]
        min_dist = self.mtp_parameters["min_dist"]
        max_dist = self.mtp_parameters["max_dist"]
        txt += f"radial basis function : {basis}\n"
        txt += f"Radial basis size :     {nbasis}\n"
        txt += f"Minimum distance :      {min_dist}\n"
        txt += f"Cutoff :                {max_dist}\n"
        return txt
