"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

mfep_default = {'pair_style': None,
                'pair_coeff': None,
                'model_post': None,
                'atom_style': "atomic",
                'bonds': None,
                'angles': None,
                'bond_style': None,
                'bond_coeff': None,
                'angle_style': None,
                'angle_coeff': None}

method_dict = {'mfep': mfep_default} 

# ========================================================================== #
# ========================================================================== #
class CalcProperty:
    """
    Parent Class managing the true potential being simulated

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object
    magmoms: :class:`np.ndarray` (optional)
        An array for the initial magnetic moments for each computation
        If ``None``, no initial magnetization. (Non magnetic calculation)
        Default ``None``.
    """
    
    def __init__(self,
                 method,
                 parameter
                 state=None):

        if not method in method_dict.keys():
            error = f'This {method} method is not implemented in MLACS'
            raise AttributeError(error)
        self.method = method
        self.state = state
        self.kwargs = method_dict[method]
        for keys, values in parameter.item():
            self.kwargs[keys] = values

# ========================================================================== #
    def run(self, step):
        """
        """
        workdir = self._update_wkdir(step)
        if self.method == 'mfep':
            self.state.run_MFEP(**self.kwargs, workdir=workdir)
        


