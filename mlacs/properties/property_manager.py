"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import numpy as np
import h5py

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
                observable.isfirstcomputation = observable.isfirst
                #Note: observable.isfirst becomes False after the 1st _exec()
                #while, observable.isfirstcomputation stays True until the
                #2nd _exec().
                #This distinction proves useful e.g. to initialize datasets
                self.check[i] = observable._exec()
                msg += repr(observable)
                
                #Check if first mlas run AND if first observable computation
                if self.isfirstlaunched and observable.isfirstcomputation:
                    self._initialize_hdf5_dataset(observable)
                    #Note: here the dataset is initialized AFTER the observable
                    #has been first computed, but BEFORE save_prop() is called,
                    #Hence, the dataset's shape is customized at will
        return msg

# ========================================================================== #
    def _initialize_hdf5_dataset(self, observable):
        """Create hdf5 dataset corresponding to a calc property object"""
        maxshape_val = (None,) + observable.shape
        hpath = self.workdir / "HIST.hdf5"
        hfile = h5py.File(hpath, "a")
        dtst_path = self.folder + '/' + observable.label
        #Some dummy data (with the correct shape) to initialize the dataset
        dummy_init = np.ones(shape=(1,)+observable.shape)*-1
        hfile.create_dataset(dtst_path, data=dummy_init, maxshape=maxshape_val)
        hfile.close()

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
            
        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class, Default: `None`.        
        
        """
        path_save = self.workdir / self.folder
        for observable in self.manager:
            to_be_saved = observable.new

            hpath = self.workdir / "HIST.hdf5"
            hfile = h5py.File(hpath, "a")
            
            if observable.shape is not None:
                
                dataset_path = self.folder + '/' + observable.label
                for idx,val_state in enumerate(to_be_saved):
                    index_state = idx+1
                    metadata = [observable.isfirstcomputation,
                                index_state,
                                dataset_path,
                                ]
                    self._append_array_to_hdf5(hfile, metadata, val_state)
                
                #scalar observables are saved in .dat files
                observable_is_scalar = (len(observable.shape) == 0)
                if observable_is_scalar:
                    namefile = path_save / (observable.label + ".dat")
                    for idx,value in enumerate(to_be_saved):
                        index_state = idx+1
                        row = [step, index_state, value]
                        self._append_row_to_dat(namefile, row)
                        
            hfile.close()
                                            
# ========================================================================== #          
    def save_weighted_prop(self, step, weighting_pol):
        """
        For all observables in a PropertyManager object, save the values
        of the observables, weighted by the weighting policy.
        
        The .dat file is formatted as:
            1st col.: index of MLAS iteration
            2nd col.: number of configs used in the database
            3rd col.: value of weighted observable
            
        Warning: At the first MLAS iteration, no weights are computed.

        Parameters
        ----------
        
        step: :class:`int`
            The index of MLAS iteration
            
        weighting_pol: :class:`WeightingPolicy`
            WeightingPolicy class, Default: `None`.        
        
        """
        path_save = self.workdir / self.folder
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
                    row = [step, nconfs_used, weighted_observable]
                    self._append_row_to_dat(namefile, row)

# ========================================================================== #        
    def save_weights(self, step, weighting_pol):
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
        
        """
              
        
        if weighting_pol is not None:
            #The first two confs of self.mlip.weight.database are never used
            #in the properties computations, so they are throwned out here
            #by the slicing operator [2:]
            weights = weighting_pol.weight[2:]
        
            path_save = self.workdir / self.folder
            to_be_saved = weights
            namefile = path_save / ('Weights' + ".dat")
            #If the properties correspond to the nth MLAS cycle
            #The weights correspond to the (n-1)th cycle
            #Hence below the occurrence of step-1
            for idx,value in enumerate(to_be_saved):
                row = [step-1, idx+1, value]
                self._append_row_to_dat(namefile, row)

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

# ========================================================================== #
    def _append_row_to_hdf5(self, hfile, dataset_path, row):
        """
        Append a new value to an .hdf5 dataset
        The dataset has to be of shape (n,), i.e. the observable must be scalar
        """
        new_value = row[2]
        current_data_length = hfile[dataset_path].shape[0]
        hfile[dataset_path].resize((current_data_length + 1), axis=0)
        hfile[dataset_path][-1:] = new_value

# ========================================================================== #
    def _append_array_to_hdf5(self, hfile, metadata, array):
        """
        Append a new array (of arbitrary shape) to an .hdf5 dataset
        
        Note:
        Due to the initialization constraint on hdf5 datasets, an initial dummy
        array is set for hfile[dataset_path][0], cf. _initialize_hdf5_dataset()
        Hence, the very first call to _append_array_to_hdf5() replaces this 
        dummy data, instead of being appended to it.
        """
        isfirstcomputation, index_state, dataset_path = metadata
        if isfirstcomputation and index_state == 1 and self.isfirstlaunched:
            hfile[dataset_path][0] = array
        else:
            current_data_length = hfile[dataset_path].shape[0]
            hfile[dataset_path].resize((current_data_length + 1), axis=0)
            hfile[dataset_path][-1:] = array
