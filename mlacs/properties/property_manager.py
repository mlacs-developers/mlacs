"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import numpy as np

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
        Check all criterions. They have to converged at the same time.
        """
        for _ in self.check:
            if not _:
                return False
        return True

# ========================================================================== #
    @Manager.exec_from_workdir
    def run(self, step, wdir):
        """
        Run property calculation.
        """
        dircheck = False
        for observable in self.manager:
            if step % observable.freq == 0:
                dircheck = True
        if dircheck:
            wdir.mkdir(exist_ok=True, parents=True)
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
                
    def save_prop(self, step):
        """
        Save the values of observables contained in a PropertyManager object.
        
        The .txt file is formatted as:
            1st col. [10 caract.]: index of MLAS iteration (int)
            2nd col. [10 caract.]: index of state at given MLAS iteration (int)
            3rd col. [20 caract.]: value of observable (float)
            Columns are separated by blanks of 5 caracters.

        Parameters
        ----------
        
        step: :class:`int`
            The index of MLAS iteration
            
        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class, Default: `None`.        
        
        """
        path_save = self.workdir / self.folder
        hspace = " "*5
        for observable in self.manager:
            to_be_saved = observable.new
            observable_is_scalar = (len(to_be_saved[0].shape) == 0)
            if observable_is_scalar:
                namefile = path_save / (observable.label + ".dat")
                for idx,value in enumerate(to_be_saved):
                    row_to_be_saved = f"{step:10.0f}" + hspace
                    row_to_be_saved += f"{idx+1:10.0f}" + hspace
                    row_to_be_saved += f"{value:20.15f}" + hspace
                    row_to_be_saved += "\n"
                    with open(namefile, "a") as f:
                        f.write(row_to_be_saved)
                        
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
    
    def save_weighted_prop(self, step, weighting_pol):
        """
        For all observables in a PropertyManager object, save the values
        of the observables, weighted by the weighting policy.
        
        The .dat file is formatted as:
            1st col. [10 caract.]: index of MLAS iteration (int)
            2nd col. [10 caract.]: number of configs used in the database (int)
            3rd col. [20 caract.]: value of weighted observable (float)
            Columns are separated by blanks of 5 caracters.
            
        Warning: At the first MLAS iteration, no weights are computed.

        Parameters
        ----------
        
        step: :class:`int`
            The index of MLAS iteration
            
        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class, Default: `None`.        
        
        """
        path_save = self.workdir / self.folder
        hspace = " "*5
        if weighting_pol is not None:
            for observable in self.manager:
                weights = weighting_pol.weight[2:]
                observable_values = self._read_prop(observable)[:len(weights)]
                nconfs_used = len(weights)
                if len(weights) > 0:
                    weighted_observable = np.sum(weights*observable_values)
                    weighted_observable /= np.sum(weights)
                    namefile = path_save / ('Weighted' + observable.label + \
                                            ".dat")
                    row_to_be_saved = f"{step:10.0f}" + hspace
                    row_to_be_saved += f"{nconfs_used:10.0f}" + hspace
                    row_to_be_saved += f"{weighted_observable:20.15f}" + hspace
                    row_to_be_saved += "\n"
                    with open(namefile, "a") as f:
                        f.write(row_to_be_saved)


    def save_weights(self, step, weighting_pol):
        """
        Save the MBAR weights.
        
        The .dat file is formatted as:
            1st col. [10 caract.]: index of MLAS iteration (int)
            2nd col. [10 caract.]: index of config in database (int)
            3rd col. [20 caract.]: weights (float)
            Columns are separated by blanks of 5 caracters.

        Parameters
        ----------
        
        step: :class:`int`
            The index of MLAS iteration
            
        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class, Default: `None`.        
        
        """
        if weighting_pol is not None:
            #The first two confs of self.mlip.weight.database are never used
            #in the properties computations, so they are throwned out here
            #by the slicing operator [2:]
            weights = weighting_pol.weight[2:]
        
            path_save = self.workdir / self.folder
            to_be_saved = weights
            namefile = path_save / ('Weights' + ".dat")
            hspace = " "*5
            #If the properties correspond to the nth MLAS cycle
            #The weights correspond to the (n-1)th cycle
            #Hence below the occurrence of step-1
            for idx,value in enumerate(to_be_saved):
                row_to_be_saved = f"{step-1:10.0f}" + hspace
                row_to_be_saved += f"{idx+1:10.0f}" + hspace
                row_to_be_saved += f"{value:20.15f}" + hspace
                row_to_be_saved += "\n"
                with open(namefile, "a") as f:
                    f.write(row_to_be_saved)
