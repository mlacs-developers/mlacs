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

        Manager.__init__(self, folder=folder, **kwargs)

        if prop is None:
            self.check = [False]
            self.manager = None

        elif isinstance(prop, list):
            self.manager = prop
            self.check = [False for _ in range(len(prop))]
        else:
            self.manager = [prop]
            self.check = [False]

# ========================================================================== #
    @property
    def check_criterion(self):
        """
        Check all criterions. They have to be all converged at the same time.
        Return True if all elements in list are True, else return False.
        """
        return np.all(self.check)

# ========================================================================== #
    @Manager.exec_from_workdir
    def run(self, step):
        """
        Run property calculation.
        """
        dircheck = False
        for observable in self.manager:
            if step % observable.freq == 0:
                dircheck = True
        if dircheck:
            self.path.mkdir(exist_ok=True, parents=True)
        msg = ""
        for i, observable in enumerate(self.manager):
            if step % observable.freq == 0:
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

        If an observable is scalar, a .dat file is saved as:
            1st col.: index of MLAS iteration
            2nd col.: index of state at given MLAS iteration
            3rd col.: value of observable
            Columns are separated by blanks of 5 caracters.

        Parameters
        ----------

        step: :class:`int`
            The index of MLAS iteration
        """
        path_save = self.workdir / self.folder
        ncpath = self.ncfile.ncpath

        if self.manager is not None:
            for observable in self.manager:
                to_be_saved = observable.new
                nc_name = observable.nc_name

                if nc_name is not None:
                    for idx, val_state in enumerate(to_be_saved):
                        with nc.Dataset(ncpath, 'a') as ncfile:
                            index_state = idx+1
                            metadata = [step, index_state]
                            idx_db = np.ma.count(ncfile[nc_name+'_meta'][:, 0])
                            # idx_db is index of conf in dtbase for observable

                            ncfile[nc_name][idx_db] = val_state
                            ncfile[nc_name+'_meta'][idx_db] = metadata
                            ncfile['mdtime'][idx_db] = idx_db + 1

                    # Scalar observables are saved in .dat files
                    observable_is_scalar = (len(observable.shape) == 0)
                    if observable_is_scalar:
                        namefile = path_save / (observable.label + ".dat")
                        for idx, value in enumerate(to_be_saved):
                            index_state = idx+1
                            row = [step, index_state, value]
                            self._append_row_to_dat(namefile, row)

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
                weights = weighting_pol.weight[2:]

                # with nc.Dataset(ncpath, 'r') as ncfile:
                #     observable_values = ncfile[nc_name][:len(weights)].data
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

        The .dat file is formatted as:
            1st col.: index of MLAS iteration
            2nd col.: index of config in database
            3rd col.: weights

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
            weights = weighting_pol.weight[2:]
            if len(weights) > 0:
                nb_effective_conf = np.sum(weights)**2 / np.sum(weights**2)
            else:
                nb_effective_conf = 0
            nb_conf = len(weights)

            path_save = self.workdir / self.folder
            namefile = path_save / ('Weights' + ".dat")
            # If the properties correspond to the nth MLAS cycle
            # The weights correspond to the (n-1)th cycle
            # Hence below the occurrence of step-1
            for idx, value in enumerate(weights):
                row = [step-1, idx+1, value]
                self._append_row_to_dat(namefile, row)

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
    def _append_row_to_dat(self, namefile, row, hspace=" "*5):
        """
        Define format of .dat file.

        The .dat file is formatted as:
            1st column: int [10 caract.]
            2nd column: int [10 caract.]
            3rd column: float [20 caract.]
            Columns are separated by blanks of hspace caracters (default 5).
        """
        int1, int2, value = row
        row_to_be_saved = f"{int1:10.0f}" + hspace
        row_to_be_saved += f"{int2:10.0f}" + hspace
        row_to_be_saved += f"{value:20.15f}" + hspace
        row_to_be_saved += "\n"
        with open(namefile, "a") as f:
            f.write(row_to_be_saved)

# ========================================================================== #
    def _read_prop(self, observable):
        """
        Read previous values of a given observable from .dat file.
        """
        path_save = self.workdir / self.folder
        namefile = path_save / (observable.label + ".dat")
        with open(namefile, "r") as f:
            beingread = []
            for line in f:
                beingread.append(float(line[25:50]))
        hasbeenread = np.array(beingread)
        return hasbeenread
