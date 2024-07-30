"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
import logging
import datetime
from .. import __version__

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ========================================================================== #
class ThermoLog:
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
    def write_header(self):
        msg = '============================================================\n'
        msg += '           Thermodynamic Integration for MLACS\n'
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
