"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
import netCDF4 as nc

from ..core.manager import Manager
from .calc_property import (CalcRoutineFunction,
                            CalcPressure,
                            CalcAcell,
                            CalcAngles,
                            CalcSpinAt,
                            CalcElectronicEntropy)


# ========================================================================== #
# ========================================================================== #
class PropertyManager(Manager):
    """
    Parent Class managing the calculation of differents properties
    """

    def __init__(self,
                 prop,
                 folder='Properties',
                 **kwargs):

        if prop is None:
            self.check = [False]
            self.manager = None
            folder = ''
        elif isinstance(prop, list):
            self.manager = prop
            self.check = [False for _ in range(len(prop))]
        else:
            self.manager = [prop]
            self.check = [False]

        if prop is not None:
            if not any([prop.needdir for prop in self.manager]):
                folder = ''

        Manager.__init__(self, folder=folder, **kwargs)
        if self.manager is not None:
            for observable in self.manager:
                if observable.needdir:
                    observable.workdir = self.workdir
                    observable.folder = self.folder

# ========================================================================== #
    @property
    def check_criterion(self):
        """
        Check all criterions. They have to be all converged at the same time.
        Return True if all elements in list are True, else return False.
        """
        return np.all(self.check)

# ========================================================================== #
    @Manager.exec_from_subdir
    def run(self, step):
        """
        Run property calculation.
        """
        msg = ""
        for i, observable in enumerate(self.manager):
            if step % observable.freq == 0:
                if observable.needdir:
                    observable.workdir = self.workdir
                    observable.folder = self.folder
                    observable.subfolder = f'Step{step}'
                self.check[i] = observable._exec()
                msg += repr(observable)
        return msg

# ========================================================================== #
    def calc_initialize(self, **kwargs):
        """
        Add on the fly arguments for calculation of properties.
        """
        for observable in self.manager:
            if observable.useatoms:
                observable.get_atoms(kwargs['atoms'])

# ========================================================================== #
    def save_prop(self, step):
        """
        Save the values of observables contained in a PropertyManager object.

        Parameters
        ----------

        step: :class:`int`
            The index of MLAS iteration
        """

        if self.manager is not None:
            for observable in self.manager:
                if step % observable.freq == 0:
                    to_be_saved = observable.new
                    nc_name = observable.nc_name

                    if nc_name is not None:
                        ncpath = self.ncfile.ncpath
                        for idx, val_state in enumerate(to_be_saved):
                            with nc.Dataset(ncpath, 'a') as ncfile:
                                index_state = idx+1
                                metadata = [step, index_state]
                                idx_db = np.ma.count(
                                        ncfile[nc_name+'_meta'][:, 0])
                                # idx_db is index of conf in dtbase
                                # for observable
                                ncfile[nc_name][idx_db] = val_state
                                ncfile[nc_name+'_meta'][idx_db] = metadata
                                ncfile['mdtime'][idx_db] = idx_db + 1

# ========================================================================== #
    def save_weighted_prop(self, step, weighting_pol):
        """
        For all observables in a PropertyManager object, save the values
        of the observables, weighted by the weighting policy.

        Parameters
        ----------

        step: :class:`int`
            The index of MLAS iteration

        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class.

        """
        ncpath = self.ncfile.ncpath

        if weighting_pol is not None:
            for observable in self.manager:
                nc_name = observable.nc_name
                weights = weighting_pol.weight[2:].copy()
                obs = self.ncfile.read_obs(nc_name)
                observable_values = obs[:len(weights)]

                if len(weights) > 0:
                    w_name = 'weighted_' + nc_name
                    weights /= np.sum(weights)
                    dim_array = np.array([1]*observable_values.ndim)
                    dim_array[0] = -1
                    r_weights = weights.reshape(dim_array)
                    if r_weights.shape[0] == observable_values.shape[0]:
                        weighted_observ = np.sum(r_weights*observable_values)
                        with nc.Dataset(ncpath, 'a') as ncfile:
                            ncfile[w_name][len(weights)-1] = weighted_observ

# ========================================================================== #
    def save_weights(self, step, weighting_pol, ncformat):
        """
        Save the MBAR weights.

        Parameters
        ----------

        step: :class:`int`
            The index of MLAS iteration

        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class, Default: `None`.

        ncformat: :class:`str`
            The format of the *HIST.nc file. One of the five flavors of netCDF
            files format available in netCDF4 python package 'NETCDF3_CLASSIC',
            'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA','NETCDF4_CLASSIC',
            'NETCDF4'.
        """

        if weighting_pol is not None:
            # The first two confs of self.mlip.weight.database are never used
            # in the properties computations, so they are throwned out here
            # by the slicing operator [2:]
            weights = weighting_pol.weight[2:].copy()
            if len(weights) > 0:
                nb_effective_conf = np.sum(weights)**2 / np.sum(weights**2)
            else:
                nb_effective_conf = 0
            nb_conf = len(weights)

            # Save weights into HIST file
            weights_ncpath = self.ncfile.ncpath
            if 'NETCDF3' in ncformat:
                weights_ncpath = weights_ncpath.replace('HIST', 'WEIGHTS')
            with nc.Dataset(weights_ncpath, 'a') as ncfile:
                idx_db = np.ma.count(ncfile['weights_meta'][:, 0])
                for idx, value in enumerate(weights):
                    ncfile['weights'][idx_db+idx] = value
                    # weights_meta keeps track of db index for given cycle
                    # and of number of effective configurations
                    metadata = [idx + 1, nb_effective_conf, nb_conf]
                    ncfile['weights_meta'][idx_db+idx] = metadata

            # Weight of first two confs (that are throwned out)
            w_first2 = abs(np.sum(weighting_pol.weight) - np.sum(weights))
            return w_first2


# ========================================================================== #
# ========================================================================== #
class RoutinePropertyManager(PropertyManager):
    """
    Class to handle the list of CalcRoutineFunction.

    Parameters
    ----------

    ncfile: :class:`OtfMlacsHist`
        The netcdf *HIST.nc file.
    """

    def __init__(self, ncfile, launched):

        # Get variables names, dimensions, and units conventions
        var_dim_dict = ncfile.var_dim_dict
        ase_units_dict = ncfile.ase_units_dict
        abinit_units_dict = ncfile.abinit_units_dict

        # Build RoutinePropertyManager
        routine_prop_list = []
        for x in var_dim_dict:
            var_name, var_dim = var_dim_dict[x]
            var_abinit_unit = abinit_units_dict[x]
            var_ase_unit = ase_units_dict[x]
            lammps_func = 'get_' + x.lower()
            observable = CalcRoutineFunction(lammps_func,
                                             label=x,
                                             nc_name=var_name,
                                             nc_dim=var_dim,
                                             nc_unit=var_abinit_unit,
                                             ase_unit=var_ase_unit,
                                             frequence=1)
            routine_prop_list.append(observable)
        other_observables = [CalcPressure(), CalcAcell(), CalcAngles(),
                             CalcSpinAt(), CalcElectronicEntropy()]
        routine_prop_list += other_observables

        PropertyManager.__init__(self, prop=routine_prop_list)

        if not launched:
            ncfile.create_nc_var(self.manager)

        self.workdir = self.workdir
        self.isfirstlaunched = not launched
        self.ncfile = ncfile
