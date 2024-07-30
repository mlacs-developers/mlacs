"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np

'''
functions for equations of state
Vinet, Birch-Murnaghan, Murnaghan
To be completed if needed
'''


def e_vinet(v, e0, b0, b0p, v0):
    fvinet = (v / v0) ** (1.0 / 3.0)
    e_vinet = e0 + 2.*b0*v0/((b0p-1.)**2) * \
        (2. - (5. + 3.*b0p*(fvinet - 1.)
               - 3.*fvinet) * np.exp(3.*(b0p-1.)*(1.-fvinet)/2.))
    return e_vinet


def e_murnaghan(v, e0, b0, b0p, v0):
    e_murnaghan = e0 + b0*v0/b0p * ((v/v0)**(1. - b0p) / (b0p - 1.)
                                    + v/v0 - b0p/(b0p - 1.))
    return e_murnaghan


def e_bm(v, e0, b0, b0p, v0):
    fvinet = (v / v0) ** (1.0 / 3.0)
    e_bm = e0 + 9.*b0*v0/16. * (fvinet**2 - 1.)**2 * (b0p * (fvinet**2 - 1)
                                                      + (6. - 4.*fvinet**2))
    return e_bm


def p_vinet(v, b0, b0p, v0):
    fvinet = (v / v0) ** (1.0 / 3.0)
    pression_vinet = 3.*b0*((1.-fvinet)/(fvinet**2)) * \
        np.exp(3.*(b0p - 1.0) * (1.0 - fvinet) / 2.0)
    return pression_vinet


def p_murnaghan(v, b0, b0p, v0):
    p_murnaghan = b0/b0p * ((v/v0)**(-b0p) - 1.)
    return p_murnaghan


def p_bm(v, b0, b0p, v0):
    fvinet = (v / v0) ** (1.0 / 3.0)
    p_bm = 3.*b0/2. * (fvinet**(-7) - fvinet**(-5)) * (1. + 3./4. * (b0p - 4.)
                                                       * (fvinet**(-2) - 1.))
    return p_bm
