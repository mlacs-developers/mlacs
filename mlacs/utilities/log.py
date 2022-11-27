"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import logging
import datetime
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
