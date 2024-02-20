'''
'''
from pathlib import Path
from subprocess import run, PIPE
from os import symlink
import shutil

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
                              init_params="random",
                              update_mindist=False)


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

    mpt_parameters: :class:`dict`
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
            - init_params='random'
            - update_mindist=False

    nthrow: :class:`int`

    energy_coefficient: :class:`float`

    forces_coefficient: :class:`float`

    stress_coefficient: :class:`float`
    """
    def __init__(self,
                 atoms,
                 mlpbin="mlp",
                 folder=Path("MTP").absolute(),
                 mtp_parameters={},
                 fit_parameters={},
                 nthrow=0,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0):
        SelfMlipManager.__init__(self,
                                 BlankDescriptor(atoms),
                                 nthrow,
                                 folder,
                                 energy_coefficient,
                                 forces_coefficient,
                                 stress_coefficient)

        self.cmd = mlpbin

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
    def get_pair_style(self, folder=None):
        return f"mlip {self.folder / 'mlip.ini'}"

# ========================================================================== #
    def get_pair_coeff(self, folder=None):
        return ["* *"]

# ========================================================================== #
    def get_pair_style_coeff(self, folder=None):
        return self.get_pair_style(), self.get_pair_coeff()

# ========================================================================== #
    def train_mlip(self, mlip_subfolder=None):
        """
        """
        if mlip_subfolder is None:
            subfolder = self.folder
        else:
            subfolder = self.folder / mlip_subfolder

        subfolder.mkdir(parents=True, exist_ok=True)

        mtpfile = self.folder / "pot.mtp"

        # Move old potential in new folder, to start BFGS from
        # previously trained MTP
        if mtpfile.exists():
            shutil.move(mtpfile,
                        subfolder / "pot.mtp")

        self._clean_folder(subfolder=subfolder)
        self._write_configurations(subfolder=subfolder)
        self._write_input(subfolder=subfolder)
        self._write_mtpfile(subfolder=subfolder)
        self._run_mlp(subfolder=subfolder)

        # Symlink new MTP in the main folder
        if mlip_subfolder is not None:
            src = subfolder / "pot.mtp"
            symlink(src, mtpfile)

        with open(self.folder / "mlip.ini", "w") as fd:
            fd.write(f"mtp-filename    {mtpfile}\n")
            fd.write("select           FALSE")

        idx_e, idx_f, idx_s = self._get_idx_fit()
        msg = "number of configurations for training: " + \
              f"{len(self.natoms[idx_e:]):}\n"
        msg += "number of atomic environments for training: " + \
               f"{self.natoms[idx_e:].sum():}\n"
        msg += self._compute_test(msg, idx_e, subfolder=subfolder)
        return msg

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
    def _clean_folder(self, subfolder):
        """
        """
        files = ["train.cfg",
                 "mlip.ini",
                 "initpot.mtp",
                 "out.cfg"]
        for file in files:
            if (subfolder / file).exists():
                (subfolder / file).unlink()

# ========================================================================== #
    def _write_configurations(self, subfolder):
        """
        """
        idx_e, idx_f, idx_s = self._get_idx_fit()
        confs = self.configurations[idx_e:]
        chemmap = self.descriptor.elements
        write_cfg(subfolder / "train.cfg", confs, chemmap)

# ========================================================================== #
    def _write_input(self, subfolder):
        """
        """
        mtpfile = subfolder / "pot.mtp"
        with open(subfolder / "mlip.ini", "w") as fd:
            fd.write(f"mtp-filename    {mtpfile}\n")
            fd.write("select           FALSE")

# ========================================================================== #
    def _write_mtpfile(self, subfolder):
        """
        """
        writenewmtp = True
        mtpfile = subfolder / "initpot.mtp"
        lvl = self.level
        level = f"level{lvl}"
        if (subfolder / "pot.mtp").exists():
            import re
            with open(subfolder / "pot.mtp", "r") as fd:
                for line in fd.readlines():
                    if line.startswith("potential_name"):
                        oldlevel = int(re.search(r'\d+$', line).group())
                        break
            if oldlevel == lvl:
                (subfolder / "pot.mtp").rename(mtpfile)
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
    def _run_mlp(self, subfolder):
        """
        """
        initpotfile = subfolder / "initpot.mtp"
        potfile = subfolder / "pot.mtp"
        trainfile = subfolder / "train.cfg"
        mlp_command = self.cmd + f" train {initpotfile} {trainfile}"
        mlp_command += f" --trained-pot-name={potfile}"
        up_mindist = self.fit_parameters["update_mindist"]
        if up_mindist:
            mlp_command += " --update-mindist"
        init_params = self.fit_parameters["init_params"]
        mlp_command += f" --init-params={init_params}"
        max_iter = self.fit_parameters["max_iter"]
        mlp_command += f" --max-iter={max_iter}"
        bfgs_conv_tol = self.fit_parameters["bfgs_conv_tol"]
        mlp_command += f" --bfgs-conv-tol={bfgs_conv_tol}"
        scale_by_forces = self.fit_parameters["scale_by_forces"]
        mlp_command += f" --scale-by-force={scale_by_forces}"
        mlp_command += f" --energy-weight={self.ecoef}"
        mlp_command += f" --force-weight={self.fcoef}"
        mlp_command += f" --stress-weight={self.scoef}"
        with open(subfolder / "mlip.log", "w") as fd:
            mlp_handle = run(mlp_command.split(),
                             stderr=PIPE,
                             stdout=fd,
                             cwd=subfolder)
        if mlp_handle.returncode != 0:
            msg = "mlp stopped with the exit code \n" + \
                  f"{mlp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    def _run_test(self, subfolder):
        """
        """
        trainfile = subfolder / "train.cfg"
        outfile = subfolder / "out.cfg"
        potfile = subfolder / "pot.mtp"
        mlp_command = self.cmd + f" calc-efs {potfile} {trainfile} {outfile}"
        mlp_handle = run(mlp_command.split(),
                         stderr=PIPE)
        if mlp_handle.returncode != 0:
            msg = "mlp stopped with the exit code \n" + \
                  f"{mlp_handle.stderr.decode()}"
            raise RuntimeError(msg)
        e_mlip, f_mlip, s_mlip = read_cfg_data(outfile)
        return e_mlip, f_mlip, s_mlip

# ========================================================================== #
    def _compute_test(self, msg, idx_e, subfolder):
        """
        """
        e_mlip, f_mlip, s_mlip = self._run_test(subfolder=subfolder)

        confs = self.configurations[idx_e:]
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
        txt += f"energy coefficient :    {self.ecoef}\n"
        txt += f"forces coefficient :    {self.fcoef}\n"
        txt += f"stress coefficient :    {self.scoef}\n"
        txt += "\n"
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
