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
neb_args = ['configurations',
            'Kspring',
            'dt']
rdf_args = ['temperature',
            'dt',
            'nsteps',
            'nsteps_eq',
            'langevin',
            'logfile',
            'rdffile']

# ========================================================================== #
# ========================================================================== #
class CalcMfep:
    """
    Class to set a minimum free energy calculation.
    See PafiLammpsState and PafiLammpsState.run_MFEP parameters.

    Parameters
    ----------
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
        self.pafi = {}
        self.kwargs = {}
        for keys, values in args.items():
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
        self.kwargs['workdir'] = wdir + 'Mfep_Calculation/'
        self.state.run_MFEP(**self.kwargs)
        self.new = np.loadtxt(self.state.workdir + 'free_energy.dat').T[5]
        if self.isfirst:
            self.old = np.zeros(len(self.new))
            check = self._check
            self.old = self.new
            self.isfirst = False
        else:
            check = self._check
            self.old = self.new
        return check

# ========================================================================== #
    @property
    def _check(self):
        """
        Check if convergence is achived.
        """
        self.maxf = np.abs(max(self.new-self.old))
        self.avef = np.abs(np.average(self.new-self.old))
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
        msg = 'Computing the minimum free energy path:\n'
        msg += self.state.log_recap_state()
        msg += 'Free energy difference along the path with previous step:\n'
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcNeb:
    """
    Class to set a NEB calculation.
    See NebLammpsState and NebLammpsState.run_NEB parameters.

    Parameters
    ----------
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

        from mlacs.state import NebLammpsState
        self.freq = frequence
        self.stop = criterion
        self.method = method
        self.neb = {}
        self.kwargs = {}
        for keys, values in args.items():
            if keys in neb_args:
                self.neb[keys] = values
            else:
                self.kwargs[keys] = values
        self.isfirst = True
        self.state = NebLammpsState(**self.neb)

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec a NEB calculation with lammps. Use replicas.
        """
        self.kwargs['workdir'] = wdir + '/NEB_Calculation/'
        self.state.run_NEB(**self.kwargs)
        self.state.extract_NEB_configurations()
        self.new = self.state.true_energies
        if self.isfirst:
            self.old = np.zeros(len(self.new))
            check = self._check
            self.old = self.new
            self.isfirst = False
        else:
            check = self._check
            self.old = self.new
        return check

# ========================================================================== #
    @property
    def _check(self):
        """
        Check if convergence is achived.
        """
        self.maxf = np.abs(max(self.new-self.old))
        self.avef = np.abs(np.average(self.new-self.old))
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
        msg = 'Computing the minimum energy path from a NEB calculation:\n'
        msg += self.state.log_recap_state()
        msg += 'Energy difference along the reaction '
        msg += 'path with previous step:\n'
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg

# ========================================================================== #
# ========================================================================== #
class CalcRdf:
    """
    Class to set a radial distribution function calculation.
    See RdfLammpsState and RdfLammpsState.run_dynamics parameters.

    Parameters
    ----------
    method: :class:`str`
        Type of criterion :
            - max, maximum difference between to consecutive step < criterion
            - ave, average difference between to consecutive step < criterion
        Default ``max``
    criterion: :class:`float`
        Stopping criterion value. Default ``0.1``
    frequence : :class:`int`
        Interval of Mlacs step to compute the property. Default ``1``

    """

    def __init__(self,
                 args,
                 atoms,
                 method='max',
                 criterion=0.05,
                 frequence=5):

        from mlacs.state import RdfLammpsState
        self.freq = frequence
        self.stop = criterion
        self.method = method
        self.atoms = atoms
        self.rdf = {}
        self.kwargs = {}
        for keys, values in args.items():
            if keys in rdf_args:
                self.rdf[keys] = values
            else:
                self.kwargs[keys] = values
        self.isfirst = True
        self.state = RdfLammpsState(**self.rdf)

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec a Rdf calculation with lammps.
        """
        self.kwargs['supercell'] = self.atoms
        self.kwargs['workdir'] = wdir + '/Rdf_Calculation/'
        self.state.run_dynamics(**self.kwargs)
        self.new = np.loadtxt(self.kwargs['workdir'] + self.rdf['rdffile'], skiprows=4, usecols=(2))
        if self.isfirst:
            self.old = np.zeros(len(self.new))
            check = self._check
            self.old = self.new
            self.isfirst = False
        else:
            check = self._check
            self.old = self.new
        return check

# ========================================================================== #
    @property
    def _check(self):
        """
        Check if convergence is achived.
        """       
        self.maxf = np.abs(max(self.new-self.old))
        self.avef = np.abs(np.average(self.new-self.old))
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
        msg = 'For the radial distribution function g(r):\n'
        msg += self.state.log_recap_state()
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg

