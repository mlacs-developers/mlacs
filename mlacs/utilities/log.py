"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import logging
import datetime

import numpy as np

from ..version import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ========================================================================== #
# ========================================================================== #
class MlacsLog:
    '''
    Logging class
    '''
    def __init__(self, logfile, restart=False):
        if not restart:
            if os.path.isfile(logfile):
                prev_step = 1
                while os.path.isfile(logfile + '{:04d}'.format(prev_step)):
                    prev_step += 1
                os.rename(logfile, logfile + "{:04d}".format(prev_step))

        self.logger_log = logging.getLogger('output')
        self.logger_log.addHandler(logging.FileHandler(logfile, 'a'))

        if not restart:
            self.write_header()
        else:
            self.write_restart()

# ========================================================================== #
    def write_header(self):
        msg = '============================================================\n'
        msg += '    On-the-fly Machine-Learning Assisted Canonical Sampling\n'
        msg += '======================= version  ' + str(__version__) + \
               ' =====================\n'
        now = datetime.datetime.now()
        msg += 'date: ' + now.strftime('%d-%m-%Y  %H:%M:%S')
        msg += '\n'
        msg += '\n'
        self.logger_log.info(msg)

# ========================================================================== #
    def write_restart(self):
        msg = '\n'
        msg += '\n'
        msg += '============================================================\n'
        msg += '                 Restarting simulation\n'
        msg += '============================================================\n'
        now = datetime.datetime.now()
        msg += 'date: ' + now.strftime('%d-%m-%Y  %H:%M:%S')
        msg += '\n'
        msg += '\n'
        self.logger_log.info(msg)

# ========================================================================== #
    def recap_mlip(self, mlip_params):
        rcut = mlip_params['rcut']
        model = mlip_params['model']
        msg = "Machine-Learning Interatomic Potential parameters\n"
        msg += f"The model used is {model}\n"
        msg += f"Cutoff radius:                               {rcut}\n"
        msg += "Descriptor\n"
        if mlip_params["style"] == "snap":
            twojmax = mlip_params['parameters']['twojmax']
            msg += "Spectral Neighbor Analysis Potential\n"
            msg += f"2Jmax:                                       {twojmax}\n"
            if mlip_params['parameters']["chemflag"] == 1:
                msg += "Multi-element version\n"
        elif mlip_params["style"] == "so3":
            nmax = mlip_params['parameters']['nmax']
            lmax = mlip_params['parameters']['lmax']
            alpha = mlip_params['parameters']['alpha']
            msg += "Smooth SO(3) Power Spectrum\n"
            msg += f"nmax                                         {nmax}\n"
            msg += f"lmax                                         {lmax}\n"
            msg += f"alpha                                        {alpha}\n"
        if mlip_params['regularization'] is not None:
            lam = mlip_params['regularization']
            msg += f"L2-norm regularization with lambda           {lam}\n"
        ecoef = mlip_params['energy_coefficient']
        fcoef = mlip_params['forces_coefficient']
        scoef = mlip_params['stress_coefficient']
        msg += f"Energy coefficient                           {ecoef}\n"
        msg += f"Forces coefficient                           {fcoef}\n"
        msg += f"Stress coefficient                           {scoef}\n"
        msg += "\n"
        self.logger_log.info(msg)

# ========================================================================== #
    def write_input_atoms(self, atoms):
        """
        """
        pos = atoms.get_scaled_positions(wrap=False)
        cell = atoms.get_cell()

        msg = "Initial configuration:\n"
        msg += "----------------------\n"
        msg += "Number of atoms:         {:}\n".format(len(atoms))
        msg += "Elements:\n"
        i = 0
        for symb in atoms.get_chemical_symbols():
            msg += symb + '  '
            i += 1
            if i == 10:
                i = 0
                msg += '\n'
        msg += "\n"
        msg += "\n"
        msg += "Supercell vectors in angstrom:\n"
        for alpha in range(3):
            a, b, c = cell[alpha]
            msg += f'{a:18.16f}  {b:18.16f}  {c:18.16f}\n'
        msg += '\n'
        msg += 'Reduced positions:\n'
        for iat in range(len(atoms)):
            x, y, z = pos[iat]
            msg += '{x:16.16f}  {y:20.16f}  {z:20.16f}\n'
        msg += '\n'
        msg += '\n'
        msg += '============================================================\n'
        self.logger_log.info(msg)

# ========================================================================== #
    def init_new_step(self, step):
        msg = '============================================================\n'
        msg += f'Step {step}'
        self.logger_log.info(msg)


# ========================================================================== #
# ========================================================================== #
class FitFactoryLog:
    '''
    Logging class for the fit factory
    '''
    def __init__(self, logfile):
        self.logger_log = logging.getLogger('output')
        self.logger_log.addHandler(logging.FileHandler(logfile, 'w'))

# ========================================================================== #
    def write_header(self):
        self.logger_log.info('=' * 79)
        self.logger_log.info('Polynomial Lammps Fit Factory'.center(79))
        self.logger_log.info(f" version {__version__} ".center(79, '='))
        now = datetime.datetime.now()
        now = f"date: {now.strftime('%d-%m-%Y  %H:%M:%S')}".center(79)
        self.logger_log.info(now)
        self.logger_log.info("")
        self.logger_log.info("")

# ========================================================================== #
    def write_info_mlip(self, elements, model, style,
                        coste, costf, costs):
        """
        """
        self.titleblock("General information on the potential")
        self.logger_log.info(f"Number of species : {len(elements)}")
        self.logger_log.info(f"Species : {' '.join(elements)}")
        self.logger_log.info(f"Model : {model}")
        self.logger_log.info(f"Descriptor style : {style}")
        self.logger_log.info("")
        self.logger_log.info("Hyperparameter cost:")
        self.logger_log.info("--------------------")
        self.logger_log.info(f"Energy :     {coste}")
        self.logger_log.info(f"Forces :     {costf}")
        self.logger_log.info(f"Stress :     {costs}")
        self.logger_log.info("")

# ========================================================================== #
    def splitprint(self):
        """
        """
        msg = "=" * 79
        self.logger_log.info(msg)

# ========================================================================== #
    def smallsplitprint(self):
        """
        """
        msg = "-" * 79
        self.logger_log.info(msg)

# ========================================================================== #
    def titleblock(self, msg):
        """
        """
        self.logger_log.info(msg.center(79))
        undertitle = "-"*len(msg)
        self.logger_log.info(undertitle.center(79))

# ========================================================================== #
    def subtitleblock(self, msg):
        """
        """
        self.logger_log.info(msg.center(40))
        undertitle = "-"*len(msg)
        self.logger_log.info(undertitle.center(40))

# ========================================================================== #
    def descriptor(self, descdct, descdct_list):
        """
        """
        self.titleblock("Descriptor hyperparameters")
        msg = "Fixed hyperparameters\n"
        msg += "---------------------\n"
        for key, val in descdct.items():
            msg += f"{key}                {val}\n"
        msg += "\n"
        msg += "Hyperparameters to be optimized\n"
        msg += "-------------------------------\n"
        msg += f"Number of hyperparameters : {len(descdct_list)}\n"
        if descdct_list:
            nval = []
            for key, val in descdct_list.items():
                nval.append(len(val))
                msg += f"{key}    nb of values {len(val)}\n"
            nvar = np.prod(nval)
            msg += f"Total number of descriptor variations : {nvar}\n"
        self.logger_log.info(msg)

# ========================================================================== #
    def print_results(self, res, cost):
        """
        """
        self.logger_log.info("Result of the fit")
        self.logger_log.info("Training dataset".center(30))
        rmse_e = res["rmse_energy_train"]
        mae_e = res["mae_energy_train"]
        rmse_f = res["rmse_forces_train"]
        mae_f = res["mae_forces_train"]
        rmse_s = res["rmse_stress_train"]
        mae_s = res["mae_stress_train"]
        self.logger_log.info(f"Energy RMSE          {rmse_e:6.4f} eV/at")
        self.logger_log.info(f"Energy MAE           {mae_e:6.4f} eV/at")
        self.logger_log.info(f"Forces RMSE          {rmse_f:6.4f} eV/angs")
        self.logger_log.info(f"Forces MAE           {mae_f:6.4f} eV/angs")
        self.logger_log.info(f"Stress MAE           {rmse_s:6.4f} eV/angs^3")
        self.logger_log.info(f"Stress MAE           {mae_s:6.4f} eV/angs^3")
        self.logger_log.info("")
        self.logger_log.info("Test dataset".center(30))
        rmse_e = res["rmse_energy_test"]
        mae_e = res["mae_energy_test"]
        rmse_f = res["rmse_forces_test"]
        mae_f = res["mae_forces_test"]
        rmse_s = res["rmse_stress_test"]
        mae_s = res["mae_stress_test"]
        self.logger_log.info(f"Energy RMSE          {rmse_e:6.4f} eV/at")
        self.logger_log.info(f"Energy MAE           {mae_e:6.4f} eV/at")
        self.logger_log.info(f"Forces RMSE          {rmse_f:6.4f} eV/angs")
        self.logger_log.info(f"Forces MAE           {mae_f:6.4f} eV/angs")
        self.logger_log.info(f"Stress MAE           {rmse_s:6.4f} eV/angs^3")
        self.logger_log.info(f"Stress MAE           {mae_s:6.4f} eV/angs^3")
        self.logger_log.info("")
        self.logger_log.info(f"Cost function        {cost}")
        self.logger_log.info("")

# ========================================================================== #
    def print_descriptor_variable(self, dct):
        """
        """
        self.subtitleblock("Descriptor hyperparmaters")
        for key, val in dct.items():
            self.logger_log.info(f"{key} :         {val}")

# ========================================================================== #
    def print_fitparam(self, dct):
        """
        """
        self.logger_log.info("")
        self.logger_log.info("Fitting hyperparameters")
        self.logger_log.info("-----------------------")
        if dct["method"] == "ols":
            dct.pop("method")
            dct.pop("lambda_ridge")
            self.logger_log.info("Linear least-squares")
        else:
            dct.pop("method")
            self.logger_log.info("Ridge regression")
        for key, val in dct.items():
            self.logger_log.info(f"{key} :         {val}")
        self.logger_log.info("")

# ========================================================================== #
    def print_bestfit(self, best_fit):
        """
        """
        self.splitprint()
        cost = best_fit.pop("costfunction")
        self.logger_log.info("Among all fits, the best cost function is :")
        self.logger_log.info(f"{cost}")
        if best_fit:
            self.logger_log.info("The corresponding hyperparameters are :")
            for key, val in best_fit.items():
                self.logger_log.info(f"{key} :      {val}")

        self.logger_log.info("")
        self.logger_log.info("You can find the corresponding MLIP files " +
                             "in the current folder")
        self.logger_log.info("Enjoy !")
