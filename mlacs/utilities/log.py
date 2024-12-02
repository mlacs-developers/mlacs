"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
import logging
import datetime

from ..version import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
_size_log = 79


# ========================================================================== #
# ========================================================================== #
class _Log:
    """

    """
    def __init__(self, logfile, restart=False):
        if not restart:
            if os.path.isfile(logfile):
                prev_step = 1
                while os.path.isfile(logfile + '{:04d}'.format(prev_step)):
                    prev_step += 1
                os.rename(logfile, logfile + "{:04d}".format(prev_step))

        self.logger_log = logging.getLogger(__name__)
        self.logger_log.addHandler(logging.FileHandler(logfile, 'a'))
        self.logger_log.setLevel(logging.INFO)

        if not restart:
            self.write_header()
        else:
            self.write_restart()

# ========================================================================== #
    def write(self, msg="", center=False, underline=False):
        """
        """
        if underline:
            ul = "*" * len(msg)
        if center:
            msg = msg.center(_size_log, " ").rstrip()
            if underline:
                ul = ul.center(_size_log, " ").rstrip()
        if underline:
            msg = msg + "\n" + ul
        self.logger_log.info(msg)

# ========================================================================== #
    def write_header(self):
        self._delimiter()
        self.write_self()
        self.write()
        self.write_copyright()
        self.write("Please read ACKNOWLEDGMENTS.md for suggested", True)
        self.write("acknowledgments of the MLACS effort.", True)
        self._delimiter()
        self.write()
        self.write(f"version {__version__}")
        self.write()
        now = datetime.datetime.now()
        self.write(f"date: {now.strftime('%d-%m-%Y %H:%M:%S')}")
        self.write()
        self.write()

# ========================================================================== #
    def write_footer(self):
        self._delimiter()
        self.write_copyright()
        self._delimiter()
        self.write()
        self._delimiter()
        self.write("Suggested acknowledgments of the MLACS usage", True, True)
        self.write()
        self.citations()
        self.write("The MLACS package")
        msg = "A. Castellano, R. Béjaud, P. Richard, O. Nadeau, " + \
              "G. Geneste, \nG. Antonius, J. Bouchet, A. Levitt, G. Stoltz" + \
              ", F. Bottin\n" + \
              "(To be submitted (2024))"
        self.write(msg)
        self._delimiter()

# ========================================================================== #
    def write_copyright(self):
        self.write("Copyright (C) 2022-2024 MLACS group.", True)
        self.write("MLACS comes with ABSOLUTELY NO WARRANTY.", True)
        self.write("This package is distributed under the terms of the", True)
        self.write("GNU General Public License, see LICENSE.md", True)
        self.write("or http://www.gnu.org.copyleft/gpl.txt.", True)
        self.write()
        self.write("MLACS is common project of the CEA,", True)
        self.write("Université de Liège, Université du Québec à Trois-Rivières",  # noqa
                   True)
        self.write("and other collaborators, see CONTRIBUTORS.md.", True)

# ========================================================================== #
    def _delimiter(self):
        """
        Just a function to delimitate iterations in the log
        """
        self.write("=" * _size_log)


# ========================================================================== #
# ========================================================================== #
class MlacsLog(_Log):
    '''
    Logging class
    '''
    def __init__(self, logfile, restart=False):
        _Log.__init__(self, logfile, restart)

# ========================================================================== #
    def write_self(self):
        """
        """
        self.write("On-the-fly Machine-Learning Assisted Canonical Sampling",
                   True, True)

# ========================================================================== #
    def citations(self):
        """
        """
        self.write("The MLACS theory and algorithm")
        msg = "A. Castellano, F. Bottin, J. Bouchet, A. Levitt, G. Stoltz" + \
              "\nPhys. Rev. B 106, L161110 (2022)"
        self.write(msg)
        self.write()

# ========================================================================== #
    def write_restart(self):
        self.write()
        self._delimiter()
        self.write("Restarting simulation", True)
        self._delimiter()
        now = datetime.datetime.now()
        self.write(f"date: {now.strftime('%d-%m-%Y %H:%M:%S')}")
        self.write()
        self.write()

# ========================================================================== #
    def write_end(self, isearlystop=False):
        """

        """
        self.write()
        self._delimiter()
        if isearlystop:
            self.write("Convergence criteria reached, stopping the simulation",
                       True)
        else:
            self.write("Max number of step reached, stopping the simulation",
                       True)
        self._delimiter()
        self.write()

# ========================================================================== #
    def write_copyright(self):
        self.write("Copyright (C) 2022-2024 MLACS group.", True)
        self.write("MLACS comes with ABSOLUTELY NO WARRANTY.", True)
        self.write("This package is distributed under the terms of the", True)
        self.write("GNU General Public License, see LICENSE.md", True)
        self.write("or http://www.gnu.org.copyleft/gpl.txt.", True)
        self.write()
        self.write("MLACS is a common project of the CEA,", True)
        self.write("Université de Liège, Université du Québec à Trois-Rivières",  # noqa
                   True)
        self.write("and other collaborators, see CONTRIBUTORS.md.", True)

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
        self._delimiter()
        self.write(f"Step {step}")


# ========================================================================== #
class ThermoLog(_Log):
    """
    Logging class for the thermodynamic integration module
    """
    def __init__(self, logfile: str, restart=False):

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
    def write_self(self):
        self.write("Non equilibrium thermodynamic integration",
                   True, True)

# ========================================================================== #
    def citations(self):
        """
        """
        self.write("Non-equilibrium thermodynamic integration for solids")
        msg = "R. Freitas, M. Asta and M. de Koning"
        msg += "\nComput Mater. Sci. 112, 333 (2016)"
        self.write(msg)
        self.write()
        self.write("Non-equilibrium thermodynamic integration for fluids")
        msg = "R.P. Leite and M. de Koning"
        msg += "\nComput Mater. Sci. 159, 316 (2018)"
        self.write(msg)
        self.write()
        self.write("MLACS implementation")
        msg = "P Richard, A Castellano, R Béjaud, L Baguet, J Bouchet, "
        msg += "G Geneste, F Bottin"
        msg += "\nPhys. Rev. Lett. 131, 201101 (2023)"
        self.write(msg)
        self.write()
