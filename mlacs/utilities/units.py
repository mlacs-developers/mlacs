"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ase.units import Bohr, Hartree, GPa, bar


# ========================================================================== #
def unit_converter(obs_arr, obs_unit, style='metal'):
    """
    Convert observable from atomic units to target units specified by `style`.

    Parameters
    ----------

    obs_arr: :class:`numpy.ndarray`
        Array of observable in atomic units.

    obs_unit: :class:`str`
        Unit of `obs_arr`.

    style: :class:`str` (optional)
        String defining the target unit system. Either `custom` or `metal`.
        Default: `metal` (defined in LAMMPS)

    Returns
    ----------

    new_arr: :class:`numpy.ndarray`
        Converted observable.

    new_unit: :class:`str`
        Unit of `new_arr`.

    Notes
    ----------

    `custom` units
        - energy: 'eV',
        - distance: 'Ang',
        - volume: 'Ang^3',
        - force: 'eV/Ang',
        - pressure/stress: 'GPa'

    `metal` units (cf. LAMMPS)
        - energy: 'eV',
        - distance: 'Ang',
        - volume: 'Ang^3',
        - force: 'eV/Ang',
        - pressure/stress: 'bar'

    ASE units
        - energy: 'eV',
        - distance: 'Ang',
        - volume: 'Ang^3',
        - force: 'eV/Ang',
        - pressure/stress: 'eV/Ang^3'
    """
    Ha2eV = Hartree
    Bohr2Ang = Bohr

    # Format of dict: {'atomic_unit': [conv_factor, 'target_unit']}
    unit_convert_dict = {'Ha': [Ha2eV, 'eV'],
                         'Bohr': [Bohr2Ang, 'Ang'],
                         'Ha/Bohr': [Ha2eV/Bohr2Ang, 'eV/Ang'],
                         'Bohr^3': [Bohr2Ang**3, 'Ang^3'],
                         'Ha/Bohr^3': [(Ha2eV/Bohr2Ang**3)/bar, 'bar'],
                         }

    if style == 'custom':
        unit_convert_dict = {'Ha': [Ha2eV, 'eV'],
                             'Bohr': [Bohr2Ang, 'Ang'],
                             'Ha/Bohr': [Ha2eV/Bohr2Ang, 'eV/Ang'],
                             'Bohr^3': [Bohr2Ang**3, 'Ang^3'],
                             'Ha/Bohr^3': [(Ha2eV/Bohr2Ang**3)/GPa, 'GPa'],
                             }

    if obs_unit in unit_convert_dict:
        conv_factor, new_unit = unit_convert_dict[obs_unit]
        new_arr = obs_arr*conv_factor
        return new_arr, new_unit
    else:
        return obs_arr, obs_unit
