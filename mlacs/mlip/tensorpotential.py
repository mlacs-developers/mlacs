'''
'''
import numpy as np
from ase.units import GPa

from . import MlipManager
from ..utilities import compute_correlation
from ..utilities import update_dataframe, create_dataframe

from ase import Atoms
from pathlib import Path

default_parameters = {"method": "ols",
                      "lambda_ridge": 1e-8,
                      "hyperparameters": {},
                      "gridcv": {}}


# ========================================================================== #
# ========================================================================== #
class TensorpotPotential(MlipManager):
    """
    Potential that use Tensorpotential to minimize a cost function.
    Parameters
    ----------
    descriptor: :class:`Descriptor`
        The descriptor used in the model.
    nthrow: :class: int
        Number of first configurations to ignore when doing the fit
    energy_coefficient: :class:`float`
        Weight of the energy in the fit
        Default 1.0
    forces_coefficient: :class:`float`
        Weight of the forces in the fit
        Default 1.0
    mbar: :class:`MbarManager`
        Weight with the multistate Bennett acceptance ratio (MBAR) method.
        Default :class:`None`
    """
    def __init__(self,
                 descriptor,
                 nthrow=0,
                 parameters={},
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 folder="tensorpot",
                 mbar=None):
        MlipManager.__init__(self,
                             descriptor,
                             nthrow,
                             energy_coefficient,
                             forces_coefficient,
                             mbar=mbar)
        self.natomenv = 0 
        self.folder = Path(folder).absolute()
        self.folder.mkdir(parents=True, exist_ok=True)        
        self.descriptor.set_folder(self.folder)

        self.parameters = default_parameters
        self.parameters.update(parameters)

        pair_style, pair_coeff = self.descriptor.get_pair_style_coeff()
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.mbar=mbar

        if self.parameters["method"] != "ols":
            if self.parameters["hyperparameters"] is None:
                hyperparam = {}
            else:
                hyperparam = self.parameters["hyperparameters"]
            hyperparam["fit_intercept"] = False
            self.parameters["hyperparameters"] = hyperparam

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        Update the pandas dataframe and write the pckl.gzip file
        """
        df = None
        # Even during a restart, we pass through here everytime
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if self.mbar is not None:
            raise NotImplementedError("Mbar not implemented for ACE")
            self.mbar.update_database(atoms)

        #### This should probably be a descriptor function ####
        if not Path.exists(self.descriptor.df_fn):
            df = create_dataframe()
        df = update_dataframe(atoms, descriptor=self.descriptor, add_result=True, mbar=self.mbar, df=df)
        df.to_pickle(self.descriptor.df_fn, compression="gzip")

        if self.descriptor.acefit is None:
            self.descriptor.create_acefit()
        self.nconfs += len(atoms)
        self.natomenv += sum(map(len, atoms))

# ========================================================================== #
    def train_mlip(self):
        """
        """
        # We start the fitting from previous coefficients.
        self.descriptor.initialize_x0()
        self.descriptor.do_fit()
        if self.mbar is not None:
            raise NotImplementedError("Another thing to implement when using mbar")
            if self.mbar.train_mlip:
                amat, ymat = self.mbar.reweight_mlip(amat, ymat)

        msg = "Number of configurations for training: " + \
               f"{self.nconfs}\n"
        msg += "Number of atomic environments for training: " + \
               f'{self.natomenv}\n'

        if self.mbar is not None:
            raise NotImplementedError("Another thing to implement when using mbar")
            #if self.mbar.train_mlip:
                #msg += self.mbar.compute_tests(amat_e, amat_f, amat_s,
                #                               ymat_e, ymat_f, ymat_s,
                #                               self.coefficients)
            #else:
            #    msg += self.compute_tests(amat_e, amat_f, amat_s,
            #                              ymat_e, ymat_f, ymat_s)
            #msg += self.mbar.run_weight(amat_e, self.coefficients)

        return msg

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        from .calculator import MlipCalculator
        calc = MlipCalculator(self)
        return calc

# ========================================================================== #
    def predict(self, atoms):
        """
        Give the energy forces stress of atoms according to the potential.
        """
        df = create_dataframe()
        import pandas as pd
        #df2=pd.read_pickle("/home/nadeauo/Projects/00-Test/03-Testerclean/ACE/ACE.pckl.gzip", compression="gzip")
        #print(f"df: {df}")
        #print(f"df2 {df2}")
        df = update_dataframe(atoms=[atoms],  
                              descriptor=self.descriptor, 
                              add_result=False,
                              df=df, 
                              mbar=self.mbar)
        #df2=pd.read_pickle("/home/nadeauo/Projects/00-Test/03-Testerclean/ACE/ACE.pckl.gzip", compression="gzip")
        #print(f"df {df}")
        #print(f"df2: {df2}")
        return self.descriptor.predict(df)

# ========================================================================== #
    def __str__(self):
        txt = f"Tensorpotential potential with {type(self.descriptor).__name__}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "Tensorpotential\n"
        txt += "Parameters:\n"
        txt += "-----------\n"
        txt += f"energy coefficient :    {self.ecoef}\n"
        txt += f"forces coefficient :    {self.fcoef}\n"
        txt += f"Using MBAR : {'True' if self.mbar else 'False'}"
        txt += "\n"
        txt += "Descriptor used in the potential:\n"
        txt += repr(self.descriptor)
        return txt
