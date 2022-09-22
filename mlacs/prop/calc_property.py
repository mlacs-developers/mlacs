"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import numpy as np

pafi_args = ['temperature',
             'configurations',
             'Kspring',
             'maxjump',
             'dt',
             'damp',
             'brownian']


# ========================================================================== #
# ========================================================================== #
class CalcMfep:
    """
    Class to set a minimum free energy calculation.


    Parameters
    ----------
    See PafiLammpsState and PafiLammpsState.run_MFEP parameters.
    method: :class:`str`
        Type of criterion :
            - max, maximum difference between to consecutive step < criterion
            - ave, average difference between to consecutive step < criterion
        Default ``max``
    criterion: :class:`float`
        Stopping criterion value (eV). Default ``0.001``
    frequence : :class:`int`
        Interval of Mlacs step to compute the property. Default ``1``
    """

    def __init__(self,
                 args,
                 method='max',
                 criterion=0.001,
                 frequence=1):

        from mlacs.state import PafiLammpsState
        self.freq = frequence
        self.stop = criterion
        self.method = method
        for keys, values in args.item():
            if keys in pafi_args:
                self.pafi[keys] = values
            else:
                self.kwargs[keys] = values
        self.isfirst = True
        self.state = PafiLammpsState(**self.pafi)

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec a MFEP calculation with lammps. Use replicas.
        """
        self.kwargs['workdir'] = wdir
        self.state.run_MFEP(**self.kwargs)
        new = np.loadtxt(wdir + 'free_energy.dat').T[5]
        if self.isfirst:
            old = np.zero(len(new))
            self.isfirst = False
            results = (new, old)
        else:
            results = (new, old)
            old = new
        return results

# ========================================================================== #
    def _check(self, results):
        """
        Check if convergence is achived.
        """
        new, old = results
        self.maxf = max(new-old)
        self.avef = np.average(new-old)
        if self.method == 'max' and self.maxf < self.stop:
            return True
        elif self.method == 'ave' and self.avef < self.stop:
            return True
        else:
            return False

# ========================================================================== #
    def log_recap(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = self.state.log_recap_state()
        msg += 'Difference of free energy along the path with previous step:\n'
        msg += '        - Maximum  : {self.maxf}\n'
        msg += '        - Averaged : {self.avef}\n'
        msg += '\n'
        return msg
