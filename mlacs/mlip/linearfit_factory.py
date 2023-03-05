"""
"""
import os
import itertools
import json
import shutil

import numpy as np
from ase.units import GPa

from ..utilities import get_elements_Z_and_masses
from ..utilities.log import FitFactoryLog
from .mlip_lammps_interface import LammpsMlipInterface

default_fit = {"method": "ols",
               "energy_coefficient": 1.0,
               "forces_coefficient": 1.0,
               "stress_coefficient": 1.0,
               "lambda_ridge": 1e-8}
default_snap = {"rcutfac": 5.0,
                "twojmax": 8,
                "rfac0": 0.99363,
                "rmin0": 0.0,
                "chemflag": 0,
                "bnormflag": 0,
                "switchflag": 1,
                "bzeroflag": 1,
                "wselfallflag": 0}
default_so3 = {"rcutfac": 5.0,
               "nmax": 4,
               "lmax": 3,
               "alpha": 1.0}


# ========================================================================== #
# ========================================================================== #
class FitLammpsMlip:
    """
    Class to fit a linear or quadratic MLIP using the ML-IAP module of LAMMPS
    The class split the dataset into a training and testing set,
    remove outliers and perform the fit according to different descriptor
    and fitting hyperparameters.

    Parameters:
    -----------
    atoms : :class:`ase.atoms`
        An atom object with all the elements that will be observed in the
        dataset
    confs: :class:`list` of :class:`ase.atoms` object
        The full dataset on which to do the training. The atoms should have
        a calculator with energy, forces and stress attached
    style: :class:`str`
        The descriptor style used. Can be "snap" or "so3".
        Default "snap"
    model: :class:`str`
        The model for the MLIP. Can be "linear" or "quadratic"
        Default "linear"
    desc_params: :class:`dict`
        The parameters for the descriptor. To realize different fit
        with different parameters, the parameter should be a list.
        The following input realize two fits with different rcutfac
        `desc_params = {"rcutfac": [5.0, 5.25]}
    fit_params: :class:`dict`
        The parameters for the fit. To realize different fit
        with different parameters, the parameter should be a list.
        The following input realize two fits with different stress coefficients
        `fit_params = {"stress_coefficient": [1.0, 5.0]}
    costenergy: :class:`float`
        The coefficient that multiply the energy RMSE when computing the cost
        function to select the best fit.
        Default 100.0
    costforces: :class:`float`
        The coefficient that multiply the forces RMSE when computing the cost
        function to select the best fit.
        Default 1.0
    coststress: :class:`float`
        The coefficient that multiply the stress RMSE when computing the cost
        function to select the best fit.
        Default 1.0
    testsetratio: :class:`float`
        The ratio giving the size of the testing dataset, between 0 and 1.
        Default 0.25
    removeoutliers: :class:`bool`
        If true, configurations with either energy, forces or stress values
        that are > 5 standard deviation over the mean will be removed before
        the fit.
        Default True
    """
    def __init__(self,
                 atoms,
                 confs,
                 style="snap",
                 model="linear",
                 desc_params=None,
                 fit_params=None,
                 costenergy=100.0,
                 costforces=1.0,
                 coststress=1.0,
                 testsetratio=0.25,
                 removeoutliers=True):
        self.elements, self.Z, self.masses, self.charges = \
            get_elements_Z_and_masses(atoms)
        self.confs = confs
        self.style = style
        self.model = model
        self.coste = costenergy
        self.costf = costforces
        self.costs = coststress
        self.log = FitFactoryLog("FitFactory.log")
        self.log.write_header()
        self.log.write_info_mlip(self.elements, self.model, self.style,
                                 self.coste, self.costf, self.costs)
        self._run(desc_params, fit_params, testsetratio, removeoutliers)

# ========================================================================== #
    def _run(self, desc_params=None, fit_params=None,
             testsetratio=0.25, removeoutliers=True):
        """
        Function that launches everything
        """
        if self.confs is None:
            msg = "You need to add atoms in the dataset"
            raise ValueError(msg)

        # Get the parameters for a given descriptor and add the default ones
        if self.style == "snap":
            default_desc = default_snap
        if self.style == "so3":
            default_desc = default_so3
        if desc_params is not None:
            default_desc.update(desc_params)

        # Get the parameters for the fit and add the default ones
        fitparams = default_fit
        if fit_params is not None:
            fitparams.update(fit_params)

        # First we need to get how many params have to be optimized,
        # For this we split the parameters in two dictionaries
        descdct = {}
        descdct_list = {}
        ndescopt = 0
        for key, val in default_desc.items():
            if isinstance(val, (list, np.ndarray)):
                descdct_list[key] = val
                ndescopt += 1
            else:
                descdct[key] = val

        # Now we do the train/test datasets splitting
        self._split_datasets(testsetratio, removeoutliers)

        self.log.descriptor(descdct, descdct_list)
        self.log.splitprint()

        if descdct_list:
            # This create a list of all posible combinations of parameters
            cost = []
            listkey = [key for key in descdct_list.keys()]
            listval = [val for val in descdct_list.values()]
            listval = list(itertools.product(*listval))
            for values in listval:
                tmp_desc = descdct.copy()
                folder = ""
                variableparam = {}
                for key, val in zip(listkey, values):
                    folder += f"{key}{val}"
                    tmp_desc[key] = val
                    variableparam[key] = val
                self.log.splitprint()
                self.log.print_descriptor_variable(variableparam)
                cost.append(self._run_fitonedesc(tmp_desc, fitparams, folder))
            costlist = [dct["costfunction"] for dct in cost]
            idx_best = np.argmin(costlist)
            best_fit = cost[idx_best]
            folder = ""
            for key, val in zip(listkey, listval[idx_best]):
                folder += f"{key}{val}"
                best_fit[f"{key}"] = val
            src = f"{folder}/MLIP.model"
            dst = "MLIP.model"
            shutil.copy(src, dst)
            src = f"{folder}/MLIP.descriptor"
            dst = "MLIP.descriptor"
            shutil.copy(src, dst)
        else:
            best_fit = self._run_fitonedesc(descdct, fitparams, ".")
        self.log.print_bestfit(best_fit)

# ========================================================================== #
    def _split_datasets(self, testratio, removeoutliers):
        """
        Call the little MC function to split the dataset and remove outliers
        """
        self.log.splitprint()
        self.log.titleblock("Splitting datasets")
        nat = np.array([len(at) for at in self.confs])
        energies = np.array([at.get_potential_energy() for at in self.confs])
        forces = np.array([at.get_forces() for at in self.confs])
        stress = np.array([at.get_stress() for at in self.confs])

        energies /= nat

        indices = mc_stratify(energies, forces, stress,
                              testratio, removeoutliers)
        self.idx_train = indices[0]
        self.idx_test = indices[1]

        ntrain = len(self.idx_train)
        emean = energies[self.idx_train].mean()
        fmean = forces[self.idx_train].mean()
        smean = stress[self.idx_train].mean()
        estd = energies[self.idx_train].std()
        fstd = forces[self.idx_train].std()
        sstd = stress[self.idx_train].std()
        msg = "Stats in training set :\n"
        msg += "-----------------------\n"
        msg += f"Number of configurations:          {ntrain}\n"
        msg += f"Energy average                    {emean:7.4f} eV/atoms\n"
        msg += f"Energy standrad deviation         {estd:7.4f} eV/atoms\n"
        msg += f"Forces average                    {fmean:7.4f} eV/angs\n"
        msg += f"Forces standard deviation         {fstd:7.4f} eV/angs\n"
        msg += f"Stress average                    {smean:7.4f} eV/angs^3\n"
        msg += f"Stress standard deviation         {sstd:7.4f} eV/angs^3\n"
        self.log.logger_log.info(msg)

        ntrain = len(self.idx_test)
        emean = energies[self.idx_test].mean()
        fmean = forces[self.idx_test].mean()
        smean = stress[self.idx_test].mean()
        estd = energies[self.idx_test].std()
        fstd = forces[self.idx_test].std()
        sstd = stress[self.idx_test].std()
        msg = "Stats in training set :\n"
        msg += "-----------------------\n"
        msg += f"Number of configurations:          {ntrain}\n"
        msg += f"Energy average                    {emean:7.4f} eV/atoms\n"
        msg += f"Energy standrad deviation         {estd:7.4f} eV/atoms\n"
        msg += f"Forces average                    {fmean:7.4f} eV/angs\n"
        msg += f"Forces standard deviation         {fstd:7.4f} eV/angs\n"
        msg += f"Stress average                    {smean:7.4f} eV/angs^3\n"
        msg += f"Stress standard deviation         {sstd:7.4f} eV/angs^3\n"
        msg += "\n"
        self.log.logger_log.info(msg)
        self.log.splitprint()

# ========================================================================== #
    def _run_fitonedesc(self, descdct, fitparams, folder):
        """
        Do all the fit given parameters for the descriptor
        Compute the feature and design matrix and perform all the fit
        with the different fit parameters.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        # First get the Lammps interface and compute the descriptor
        rcut = descdct.pop("rcutfac")
        lmp_itrf = LammpsMlipInterface(self.elements, self.masses, self.Z,
                                       rcut, self.model, self.style,
                                       descdct, folder=folder)
        amat_train, ymat_train, amat_test, ymat_test = \
            self._compute_descriptor(lmp_itrf, folder)

        # Now we do the splitting for the fitting hyperparameters
        listkey = [key for key in fitparams.keys()]
        listval = [val for val in fitparams.values()]

        # Now we split the parameters
        fitdct = {}
        fitdct_list = {}
        nfitopt = 0
        for key, val in fitparams.items():
            if isinstance(val, (list, np.ndarray)):
                fitdct_list[key] = val
                nfitopt += 1
            else:
                fitdct[key] = val

        if fitdct_list:
            cost = []
            # This create a list of all posible combinations of parameters
            listkey = [key for key in fitdct_list.keys()]
            listval = [val for val in fitdct_list.values()]
            listval = list(itertools.product(*listval))
            for values in listval:
                tmp_fit = fitdct.copy()
                folderfit = f"{folder}/"
                for key, val in zip(listkey, values):
                    folderfit += f"{key}{val}"
                    tmp_fit[key] = val
                folderfit += "/"
                self.log.logger_log.info("")
                self.log.smallsplitprint()
                self.log.print_fitparam(tmp_fit)
                res = self._run_fit(amat_train, ymat_train,
                                    amat_test, ymat_test,
                                    tmp_fit, folderfit, lmp_itrf)
                rmse_e = res["rmse_energy_train"]
                rmse_f = res["rmse_forces_train"]
                rmse_s = res["rmse_stress_train"]
                cost_tmp = rmse_e * self.coste + \
                    rmse_f * self.costf + \
                    rmse_s * self.costs
                self.log.print_results(res, cost_tmp)
                cost.append(cost_tmp)
            idx_best = np.argmin(cost)
            best_fit = {"costfunction": cost[idx_best]}
            folderfit = f"{folder}/"
            for key, val in zip(listkey, listval[idx_best]):
                folderfit += f"{key}{val}"
                best_fit[f"{key}"] = val
            # We copy the best input in the corresponding folder
            src = f"{folderfit}/MLIP.model"
            dst = f"{folder}/MLIP.model"
            shutil.copy(src, dst)
        else:
            folderfit = f"{folder}/"
            res = self._run_fit(amat_train, ymat_train,
                                amat_test, ymat_test,
                                fitdct, folderfit, lmp_itrf)
            rmse_e = res["rmse_energy_train"]
            rmse_f = res["rmse_forces_train"]
            rmse_s = res["rmse_stress_train"]
            cost = rmse_e * self.coste + \
                rmse_f * self.costf + \
                rmse_s * self.costs
            self.log.print_results(res, cost)
            best_fit = {"costfunction": cost}
        return best_fit

# ========================================================================== #
    def _run_fit(self, amat_train, ymat_train, amat_test, ymat_test,
                 fitparams, folder, lmp_itrf):
        """
        The least squares function. At this moment, the descriptor and fit
        parameters have been separated
        """
        sigma_e = np.std(amat_train["amat_e"])
        sigma_f = np.std(amat_train["amat_f"])
        sigma_s = np.std(amat_train["amat_s"])

        ecoef = fitparams["energy_coefficient"]
        fcoef = fitparams["forces_coefficient"]
        scoef = fitparams["stress_coefficient"]
        ecoef = ecoef / sigma_e / len(ymat_train["ymat_e"])
        fcoef = fcoef / sigma_f / len(ymat_train["ymat_f"])
        scoef = scoef / sigma_s / len(ymat_train["ymat_s"])

        amat = np.r_[amat_train["amat_e"] * ecoef,
                     amat_train["amat_f"] * fcoef,
                     amat_train["amat_s"] * scoef]
        ymat = np.r_[ymat_train["ymat_e"] * ecoef,
                     ymat_train["ymat_f"] * fcoef,
                     ymat_train["ymat_s"] * scoef]

        coef = np.linalg.lstsq(amat, ymat, rcond=None)[0]

        if not os.path.exists(folder):
            os.makedirs(folder)
        lmp_itrf.write_mlip_model(coef, folder)

        metric = self._compute_metrics(amat_train, ymat_train,
                                       amat_test, ymat_test,
                                       coef, folder)

        # Prepare message to the log
        msg = "Results training set\n"
        msg += "RMSE Energy    {:.4f} eV/at\n".format(
            metric["rmse_energy_train"])
        msg += "MAE Energy     {:.4f} eV/at\n".format(
            metric["mae_energy_train"])
        msg += "RMSE Forces    {:.4f} eV/angs\n".format(
            metric["rmse_forces_train"])
        msg += "MAE Forces     {:.4f} eV/angs\n".format(
            metric["mae_forces_train"])
        msg += "RMSE Stress    {:.4f} GPa\n".format(
            metric["rmse_stress_train"])
        msg += "MAE Stress     {:.4f} GPa\n".format(
            metric["mae_stress_train"])
        msg += "\n"
        msg += "Results test set\n"
        msg += "RMSE Energy    {:.4f} eV/at\n".format(
            metric["rmse_energy_test"])
        msg += "MAE Energy     {:.4f} eV/at\n".format(
            metric["mae_energy_test"])
        msg += "RMSE Forces    {:.4f} eV/angs\n".format(
            metric["rmse_forces_test"])
        msg += "MAE Forces     {:.4f} eV/angs\n".format(
            metric["mae_forces_test"])
        msg += "RMSE Stress    {:.4f} GPa\n".format(
            metric["rmse_stress_test"])
        msg += "MAE Stress     {:.4f} GPa\n".format(
            metric["mae_stress_test"])
        msg += "\n"
        return metric

# ========================================================================== #
    def _compute_metrics(self, amat_train, ymat_train, amat_test, ymat_test,
                         coef, folder):
        """
        Compute RMSE and MAE for training and testing datasets
        """
        # First compute the metrics for the training part
        e_true = ymat_train["ymat_e"]
        f_true = ymat_train["ymat_f"]
        s_true = ymat_train["ymat_s"] / GPa
        e_mlip = np.einsum("i,ji->j", coef, amat_train["amat_e"])
        f_mlip = np.einsum("i,ji->j", coef, amat_train["amat_f"])
        s_mlip = np.einsum("i,ji->j", coef, amat_train["amat_s"]) / GPa

        rmse_e = np.sqrt(np.mean((e_true - e_mlip)**2))
        rmse_f = np.sqrt(np.mean((f_true - f_mlip)**2))
        rmse_s = np.sqrt(np.mean((s_true - s_mlip)**2))
        mae_e = np.mean(np.abs(e_true - e_mlip))
        mae_f = np.mean(np.abs(f_true - f_mlip))
        mae_s = np.mean(np.abs(s_true - s_mlip))

        header = f"rmse: {rmse_e:.5f} eV/at,    " + \
                 f"mae: {mae_e:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt(f"{folder}MLIP-Energy_comparison.dat",
                   np.vstack((e_true, e_mlip)).T,
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_f:.5f} eV/angs   " + \
                 f"mae: {mae_f:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt(f"{folder}MLIP-Forces_comparison.dat",
                   np.vstack((f_true.flatten(), f_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_s:.5f} GPa       " + \
                 f"mae: {mae_s:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt(f"{folder}MLIP-Stress_comparison.dat",
                   np.vstack((s_true.flatten(), s_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")

        res = {"rmse_energy_train": rmse_e,
               "rmse_forces_train": rmse_f,
               "rmse_stress_train": rmse_s,
               "mae_energy_train": mae_e,
               "mae_forces_train": mae_f,
               "mae_stress_train": mae_s}

        # Then for the testing part
        e_true = ymat_test["ymat_e"]
        f_true = ymat_test["ymat_f"]
        s_true = ymat_test["ymat_s"] / GPa
        e_mlip = np.einsum("i,ji->j", coef, amat_test["amat_e"])
        f_mlip = np.einsum("i,ji->j", coef, amat_test["amat_f"])
        s_mlip = np.einsum("i,ji->j", coef, amat_test["amat_s"]) / GPa

        rmse_e = np.sqrt(np.mean((e_true - e_mlip)**2))
        rmse_f = np.sqrt(np.mean((f_true - f_mlip)**2))
        rmse_s = np.sqrt(np.mean((s_true - s_mlip)**2))
        mae_e = np.mean(np.abs(e_true - e_mlip))
        mae_f = np.mean(np.abs(f_true - f_mlip))
        mae_s = np.mean(np.abs(s_true - s_mlip))

        res["rmse_energy_test"] = rmse_e
        res["rmse_forces_test"] = rmse_f
        res["rmse_stress_test"] = rmse_s
        res["mae_energy_test"] = mae_e
        res["mae_forces_test"] = mae_f
        res["mae_stress_test"] = mae_s

        header = f"rmse: {rmse_e:.5f} eV/at,    " + \
                 f"mae: {mae_e:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt(f"{folder}TestSet-Energy_comparison.dat",
                   np.vstack((e_true, e_mlip)).T,
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_f:.5f} eV/angs   " + \
                 f"mae: {mae_f:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt(f"{folder}TestSet-Forces_comparison.dat",
                   np.vstack((f_true.flatten(), f_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_s:.5f} GPa       " + \
                 f"mae: {mae_s:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt(f"{folder}TestSet-Stress_comparison.dat",
                   np.vstack((s_true.flatten(), s_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")
        with open(f"{folder}metric.json", "w") as fd:
            fd.write(json.dumps(res, indent=1))
        return res

# ========================================================================== #
    def _compute_descriptor(self, lmp_itrf, folder):
        """
        Call the mlip_lammps_interface to compute the descriptor given
        some parameters
        """
        # Compute descriptor for train set
        at = self.confs[self.idx_train[0]]
        natoms = len(at)
        descriptor, data = lmp_itrf.compute_fit_matrix(at)
        amat_e = descriptor[0] / natoms
        amat_f = descriptor[1:1+3*natoms]
        amat_s = descriptor[1+3*natoms:]
        ymat_e = data[0] / natoms
        ymat_f = data[1:1+3*natoms]
        ymat_s = data[1+3*natoms:]
        for idxtrain in self.idx_train[1:]:
            at = self.confs[idxtrain]
            lmp_itrf.compute_fit_matrix(at)
            descriptor, data = lmp_itrf.compute_fit_matrix(at)
            amat_etmp = descriptor[0] / natoms
            amat_ftmp = descriptor[1:1+3*natoms]
            amat_stmp = descriptor[1+3*natoms:]
            ymat_etmp = data[0] / natoms
            ymat_ftmp = data[1:1+3*natoms]
            ymat_stmp = data[1+3*natoms:]
            amat_e = np.vstack((amat_etmp, amat_e))
            amat_f = np.r_[amat_ftmp, amat_f]
            amat_s = np.r_[amat_stmp, amat_s]
            ymat_e = np.r_[ymat_etmp, ymat_e]
            ymat_f = np.r_[ymat_ftmp, ymat_f]
            ymat_s = np.r_[ymat_stmp, ymat_s]
        amat_train = {"amat_e": amat_e, "amat_f": amat_f, "amat_s": amat_s}
        ymat_train = {"ymat_e": ymat_e, "ymat_f": ymat_f, "ymat_s": ymat_s}

        # Compute descriptor for test set
        natoms = len(self.confs[self.idx_test[0]])
        descriptor, data = lmp_itrf.compute_fit_matrix(at)
        amat_e = descriptor[0] / natoms
        amat_f = descriptor[1:1+3*natoms]
        amat_s = descriptor[1+3*natoms:]
        ymat_e = data[0] / natoms
        ymat_f = data[1:1+3*natoms]
        ymat_s = data[1+3*natoms:]
        for idxtest in self.idx_test[1:]:
            at = self.confs[idxtest]
            lmp_itrf.compute_fit_matrix(at)
            descriptor, data = lmp_itrf.compute_fit_matrix(at)
            descriptor, data = lmp_itrf.compute_fit_matrix(at)
            amat_etmp = descriptor[0] / natoms
            amat_ftmp = descriptor[1:1+3*natoms]
            amat_stmp = descriptor[1+3*natoms:]
            ymat_etmp = data[0] / natoms
            ymat_ftmp = data[1:1+3*natoms]
            ymat_stmp = data[1+3*natoms:]
            amat_e = np.vstack((amat_etmp, amat_e))
            amat_f = np.r_[amat_ftmp, amat_f]
            amat_s = np.r_[amat_stmp, amat_s]
            ymat_e = np.r_[ymat_etmp, ymat_e]
            ymat_f = np.r_[ymat_ftmp, ymat_f]
            ymat_s = np.r_[ymat_stmp, ymat_s]
        amat_test = {"amat_e": amat_e, "amat_f": amat_f, "amat_s": amat_s}
        ymat_test = {"ymat_e": ymat_e, "ymat_f": ymat_f, "ymat_s": ymat_s}
        return amat_train, ymat_train, amat_test, ymat_test


def mc_stratify(energies, forces, stress, testratio, removeoutliers):
    """
    Little function to ensure somewhat similar mean and variances
    between train and test sets
    """
    # Initial random splitting
    nsample = len(energies)
    rng = np.random.default_rng()
    all_idx = np.arange(nsample)
    if removeoutliers:
        emean = energies.mean()
        fmean = forces.mean()
        smean = stress.mean()
        evar = energies.std()
        fvar = forces.std()
        svar = stress.std()
        idxrmv = []
        for i in all_idx:
            if np.abs(energies[i] - emean) > 5 * evar:
                idxrmv.append(i)
            elif np.abs(np.abs(forces[i]).max() - fmean) > 5 * fvar:
                idxrmv.append(i)
            elif np.abs(np.abs(stress[i]).max() - smean) > 5 * svar:
                idxrmv.append(i)
        all_idx = np.delete(all_idx, idxrmv)
    nsample = len(all_idx)

    idx_test = rng.choice(all_idx, int(nsample * testratio), False)
    idx_train = []
    for i in all_idx:
        if i not in idx_test:
            idx_train.append(i)
    idx_train = np.array(idx_train)

    def mean_var_idx(idx):
        _emean = energies[idx].mean()
        _fmean = forces[idx].mean()
        _smean = stress[idx].mean()
        _evar = energies[idx].var()
        _fvar = forces[idx].var()
        _svar = stress[idx].var()
        return np.array([_emean, _fmean, _smean, _evar, _fvar, _svar])

    stat_train = mean_var_idx(idx_train)
    stat_test = mean_var_idx(idx_test)

    def costfunc(stat_train, stat_test):
        return np.sqrt((stat_train - stat_test)**2).sum()

    costold = costfunc(stat_train, stat_test)
    costnew = costfunc(stat_train, stat_test)
    idx_trainnew = idx_train.copy()
    idx_testnew = idx_test.copy()

    arange_train = np.arange(len(idx_train))
    arange_test = np.arange(len(idx_test))
    for _ in range(1000):
        itest = rng.choice(arange_test, 1)
        itrain = rng.choice(arange_train, 1)

        idx_trainnew[itrain] = idx_test[itest]
        idx_testnew[itest] = idx_train[itrain]
        stat_train = mean_var_idx(idx_trainnew)
        stat_test = mean_var_idx(idx_testnew)

        costnew = costfunc(stat_train, stat_test)
        if costnew < costold:
            idx_train[itrain] = idx_trainnew[itrain]
            idx_test[itest] = idx_testnew[itest]
            costold = costnew
        else:
            idx_trainnew[itrain] = idx_train[itrain]
            idx_testnew[itest] = idx_test[itest]
    return idx_train, idx_test
