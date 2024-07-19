"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
import mlacs.utilities.eos_functions as eos_functions

from scipy.optimize import curve_fit

Bohr_GPa = 29421.033


def eos_fit(x=None, y=None, y_type='presssure', eos='vinet', **kwargs):
    '''
    fit an eos from E(V) or P(V) on an isotherm
    return fitted isothermal params (e0,) b0, b0p, v0

    be aware of used units:
    - for P(V) GPa and Angstrom^3 work
    - for E(V) all should be in Ha (Bohr^3, Ha/Bohr^3)
      or in eV (Angstrom^3 and eV/Angstrom^3)

    Parameters
    ----------

    x: volume 1D array,
       per atom pay an attention to hexa phases for instance
    y: energy or pressure 1D array
    y_type: str
        pressure or volume
    model: str
        birch-murnaghan
        murnaghan
        vinet

    Optionnal
    ---------

    plot:
    save:
    error:
    '''

    # get x and y from canonical sampling if not given
    # initial guess
    a, b, c = np.polyfit(x, y, 2)
    v0 = -b/(2*a)
    b0 = 2*a*v0
    b0p = 4.0

    if y_type == 'energy':
        e0 = a * v0**2 + b * v0 + c
        bounds = ([e0 * 2, b0 / 100, b0p / 10, v0 / 2],
                  [e0 / 2, b0 * 100, b0p * 10, v0 * 2])
        if eos == 'vinet':
            fitted_params, ecov_ = curve_fit(eos_functions.e_vinet, x, y,
                                             bounds=bounds)
        if eos == 'murnaghan':
            fitted_params, ecov_ = curve_fit(eos_functions.e_murnaghan, x, y,
                                             bounds=bounds)
        if eos == 'bm':
            fitted_params, ecov_ = curve_fit(eos_functions.e_bm, x, y,
                                             bounds=bounds)

        str = 'Fitted Parameters from EoS fit\n'
        str += '------------------------------\n'
        str += 'V0 = {:12.8f} Angstrom^3\n'.format(fitted_params[3])
        str += 'B0 = {:12.8f} GPa\n'.format(fitted_params[1])
        str += 'B0p = {:12.8f} \n'.format(fitted_params[2])
        str += 'E0 = {:12.8f} eV/at\n'.format(fitted_params[0])

    else:
        bounds = ([b0 / 100, b0p / 10, v0 / 2], [b0 * 100, b0p * 10, v0 * 2])
        # ph = 1
        if eos == 'vinet':
            fitted_params, pcov_ = curve_fit(eos_functions.p_vinet, x, y,
                                             bounds=bounds)
        if eos == 'murnaghan':
            fitted_params, pcov_ = curve_fit(eos_functions.p_murnaghan, x, y,
                                             bounds=bounds)
        if eos == 'bm':
            fitted_params, pcov_ = curve_fit(eos_functions.p_bm, x, y,
                                             bounds=bounds)

        str = 'Fitted Parameters from EoS fit\n'
        str += '------------------------------\n'
        str += 'V0 = {:12.8f} Angstrom^3\n'.format(fitted_params[2])
        str += 'B0 = {:12.8f} GPa\n'.format(fitted_params[0])
        str += 'B0p = {:12.8f}\n'.format(fitted_params[1])

    print(str)
    return fitted_params
