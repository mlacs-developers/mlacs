"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import numpy as np

default_args = {'pair_style': None,
                'pair_coeff': None,
                'model_post': None,
                'atom_style': "atomic",
                'bonds': None,
                'angles': None,
                'bond_style': None,
                'bond_coeff': None,
                'angle_style': None,
                'angle_coeff': None}


# ========================================================================== #
# ========================================================================== #
class CalcMfep:
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

    def __init__(method='maxf'
                 criterion=0.001,
                 freq=1,
                 args):

        from mlacs.state import PafiLammpsState
        self.freq = freq
        self.stop = criterion
        self.method = method
        self.kwargs = default_args
        for keys, values in args.item():
            self.kwargs[keys] = values
        self.isfirst = True

        T = self.kwargs['temperature']
        configs = self.kwargs['configurations']
        del self.kwargs['temperature']
        del self.kwargs['configurations']
        self.state = PafiLammpsState(temperature=T,
                                     configurations=configs)

    def _exec(self, wdir):
        self.kwargs['workdir'] = wdir
        state.run_MFEP(**self.kwargs)
        new = np.loadtxt(wdir + 'free_energy.dat').T[5]
        if self.isfirst:
            old = np.zero(len(new))
            self.isfirst = False
            results = (new, old) 
        else:
            results = (new, old)
            old = new
        return results
        
    def _check(self, results):
        new, old = results
        self.maxf = max(new-old)
        self.avef = np.average(new-old)
        if self.method == 'max' and self.maxf < self.stop: 
            return True
        elif self.method == 'maxave' and self.avef < self.stop: 
            return True
        else:
            return False

    def log_recap(): 
        msg = self.state.log_recap_state()
        msg += 'Difference of free energy along the path with previous step:\n'
        msg += '        - Maximum  : {self.maxf}\n'
        msg += '        - Averaged : {self.avef}\n'
        msg += '\n'
        return msg


