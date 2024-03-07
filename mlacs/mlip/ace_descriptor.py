import os
import warnings

from pathlib import Path
from subprocess import run, PIPE
import shlex

import numpy as np
from ase import Atoms
from ..utilities import update_dataframe, create_dataframe
from pyace.basis import BBasisConfiguration

from ..utilities import subfolder
from .descriptor import Descriptor

try:
    import pandas as pd
    ispandas = True
except ImportError:
    ispandas = False

warnings.filterwarnings("ignore", category=Warning, module="tensorflow")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove GPU warning for tf
    import tensorflow as tf  # noqa
    istf = True
except ImportError:
    istf = False

try:
    import pyace
    from pyace.generalfit import GeneralACEFit
    from pyace import create_multispecies_basis_config
    from pyace.metrics_aggregator import MetricsAggregator
    ispyace = True
    def_bconf = {'deltaSplineBins': 0.001,
                 'embeddings': {"ALL": {'npot': 'FinnisSinclairShiftedScaled',
                                        'fs_parameters': [1, 1, 1, 0.5],
                                        'ndensity': 2}},
                 'bonds': {"ALL": {'radbase': 'ChebExpCos',
                                   'NameOfCutoffFunction': 'cos',
                                   'radparameters': [5.25],
                                   'rcut': 7,
                                   'dcut': 0.01}},
                 'functions': {"ALL": {'nradmax_by_orders': [15, 3, 2],
                                       'lmax_by_orders': [0, 2, 2]}}}

    def_loss = {'kappa': 0.3, 'L1_coeffs': 1e-12, 'L2_coeffs': 1e-12,
                'w0_rad': 1e-12, 'w1_rad': 1e-12, 'w2_rad': 1e-12}

    def_fitting = {'maxiter': 100, 'fit_cycles': 1, 'repulsion': 'auto',
                   'optimizer': 'BFGS',
                   'optimizer_options': {'disp': True, 'gtol': 0, 'xrtol': 0}}

    def_backend = {'evaluator': 'tensorpot', 'parallel_mode': 'parallel',
                   'batch_size': 100, 'display_step': 50}
except ImportError:
    ispyace = False


# TODO : 1. Restarts from previous calc
#        2. Create/Update dataframe in utilities
#        3. Mbar
#        4. Lammps
#        5. Random Function
#        6. Error if DeltaLearningPotential with ACE
#        7. Remove electronic contribution during the fitting
#        8. Start the fitting with a SNAP
#        9. Make sure scipy has the time to write the coefficients before
#           the StopIteration flag in Python 3.7
#       10. Can predict deal with stress ?
#       11. Add a check for stress coeff to be 0 until I figure it out
# ========================================================================== #
# ========================================================================== #
class AceDescriptor(Descriptor):
    """
    Interface to the ACE potential using python-ace. Pace potential in LAMMPS.
    Usually a better generalization than SNAP but harder to train.
    Can be used with state at different pressure.
    Note : You will probably want to use 10-20 states since the
           fitting is relatively slow

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Reference structure, with the elements for the descriptor

    free_at_e : :class:`dict`
        The energy of one atom for every element.
        The atom MUST be isolated in a big box (eV/at)
        e.g. : {'Cu': 12.1283, 'O': 3.237}

    rcut: :class:`float`
        The cutoff of the descriptor, in angstrom
        Default 5.0

    tol_e: :class:`float`
        The tolerance on energy between ACE and DFT (meV/at)
        Default `5`

    tol_f: :class:`float`
        The tolerance on forces between ACE and DFT (meV/ang)
        Default `25`

    bconf_dict: :class:`dict`
        A dictionnary of parameters for the BBasisConfiguration
        The default values are
            - deltaSplineBins: 0.001
            - elements: Found dynamically (e.g. Cu2O: ['Cu', 'O'])
            - embeddings: {"ALL": {npot: 'FinnisSinclairShiftedScaled',
                                   fs_parameters: [1, 1, 1, 0.5],
                                   ndensity: 2}}
            - bonds: {"ALL": {radbase: 'ChebExpCos',
                              NameOfCutoffFunction: 'cos',
                              radparameters: [5.25],
                              rcut: 7,
                              dcut: 0.01}}
            - functions: {"ALL": {nradmax_by_orders: [15, 3, 2],
                                  lmax_by_orders: [0, 2, 2]}}

    loss_dict: :class:`dict`
        A dictionnary of parameters for the loss function
        The default values are
            - kappa: 0.3
            - L1_coeffs: 1e-12
            - L2_coeffs: 1e-12
            - w0_rad: 1e-12
            - w1_rad: 1e-12
            - w2_rad: 1e-12

    fitting_dict: :class:`dict`
        A dictionnary of parameters for the minimization
        The default values are
            - weighting: MBAR|Uniform otherwise
            - loss: loss_dict
            - maxiter: 100
            - fit_cycles: 1
            - repulsion: 'auto'
            - optimizer: 'BFGS'
            - optimizer_options: {disp: True, gtol: 0, xrtol: 0}

    backend_dict: :class:`dict`
        A dictionnary of parameters for the backend
        The default values are
            - evaluator: 'tensorpot'
            - parallel_mode: 'parallel'
            - n_workers: None
            - batch_size: 100
            - display_step: 50
    """
    def __init__(self, atoms, free_at_e, rcut=5.0, tol_e=5, tol_f=25,
                 bconf_dict=None, loss_dict=None, fitting_dict=None,
                 backend_dict=None, nworkers=None):

        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd
        self._verify_dependency()

        Descriptor.__init__(self, atoms, rcut)
        self.df_fn = Path("ACE.pckl.gzip").absolute()
        self.folder = ""

        self.rcut = rcut
        self.tol_e = tol_e
        self.tol_f = tol_f
        self.free_at_e = free_at_e
        bconf = def_bconf if bconf_dict is None else bconf_dict
        self.loss = def_loss if loss_dict is None else loss_dict
        self.fitting = def_fitting if fitting_dict is None else fitting_dict
        self.backend = def_backend if backend_dict is None else backend_dict
        self.fitting['loss'] = self.loss
        self.data = dict(filename=str(self.df_fn))

        if 'nworkers' not in self.backend and nworkers is not None:
            self.backend['parallel_mode'] = "process"
            self.backend['nworkers'] = nworkers
        if 'elements' not in self.loss:
            bconf['elements'] = np.unique(atoms.get_chemical_symbols())
        self.bconf = create_multispecies_basis_config(bconf)
        self.acefit = None

# ========================================================================== #
    def prepare_wf(self, wf, natoms):
        """
        Reshape wf from a flat array to a list of np.array where each
        np.array correspond to a conf i and is of size self.natoms[i]
        """
        new_wf = []
        curr_index = 0
        for nat in natoms:
            new_wf.append(wf[curr_index:nat+curr_index])
            curr_index += nat
        return new_wf

# ========================================================================== #
    @subfolder
    def fit(self, weights, atoms, name, natoms, energy, forces):
        """
        """
        # Data preparation
        nconfs=len(natoms)
        we = weights[:nconfs].tolist()
        wf = weights[nconfs:-(nconfs)*6]
        wf = self.prepare_wf(wf, natoms)
        atomic_env = self.compute_descriptor(atoms)

        # Dataframe preparation
        df = self.get_df()
        if not Path.exists(self.df_fn):
            df = create_dataframe()
        df = update_dataframe(
            df=df, name=name, atoms=atoms, atomic_env=atomic_env,
            energy=energy, forces=forces, we=we, wf=wf)
        df.to_pickle(self.df_fn, compression="gzip")

        # Do the fitting
        if self.acefit is None:
            self.create_acefit()
        else: 
            self.acefit.fit_config['fit_cycles'] += 1
        
        try:
            self.acefit.fit()
        except StopIteration as e:  # Scipy >= 1.11 works with StopIteration
            print("Warning : You should upgrade to Scipy>=1.11.0.")
            print("Everything should still work")
        finally:
            fn_yaml = "interim_potential_best_cycle.yaml"
            yace_cmd = f"pace_yaml2yace {fn_yaml} -o ACE.yace"
            run(shlex.split(yace_cmd))
            if not Path("ACE.yace").exists():
                msg = "The ACE fitting wasn't successful\n"
                msg += "If interim_potential_best_cycle.yaml doesn't exist "
                msg += f"in {Path().cwd()} then the ACEfit went wrong.\n"
                msg += f"Else, try this command '{yace_cmd}' inside "
                msg += f"{Path().cwd()}"
                raise RuntimeError(msg)
        return "ACE.yace", "interim_potential_best_cycle.yaml"

# ========================================================================== #
    @subfolder
    def create_acefit(self):
        """
        Creates the ACEFit Object. We need at least 1 conf.
        """
        def check_conv(last_fit_metric_data):
            """
            Function called after every fitting iteration to check convergence
            """
            iter_num = last_fit_metric_data["iter_num"]
            if iter_num == 0:
                MetricsAggregator.print_detailed_metrics(
                    last_fit_metric_data, title='Initial state:')
            elif iter_num % self.backend['display_step'] == 0:
                MetricsAggregator.print_extended_metrics(
                    last_fit_metric_data, title='INIT STATS:')
            e = last_fit_metric_data["rmse_epa"] * 1e3
            f = last_fit_metric_data["rmse_f_comp"] * 1e3
            print(f"Energy: {e} (meV/at), Tol:{self.tol_e} (meV/at)")
            print(f"Forces: {f} (meV/ang), Tol:{self.tol_f} (meV/ang)")
            if iter_num > 0 and self.tol_e >= e and self.tol_f >= f:
                s = "Convergence reached:\n"
                s += f"RMSE of energy: {e} < {self.tol_e} (meV/at)\n"
                s += f"RMSE of forces: {f} < {self.tol_f} (meV/ang)\n"
                print(s)
                raise StopIteration("My convergence reached")

        self.callback = lambda val: check_conv(val)

        self.acefit = GeneralACEFit(potential_config=self.bconf,
                                    fit_config=self.fitting,
                                    data_config=self.data,
                                    backend_config=self.backend)
        self.acefit.fit_backend.fit_metrics_callback = self.callback

# ========================================================================== #
    def calc_free_e(self, atoms):
        """
        Calculate the energy of free atom in the structure
        """
        e = 0
        for atom in atoms:
            e_tmp = self.free_at_e[atom.symbol]
            e += e_tmp
        return e

# ========================================================================== #
    def predict(self, atoms, bconf=None):
        raise NotImplementedError

# ========================================================================== #
    def _verify_dependency(self):
        """
        """
        s = ""
        if not ispandas:
            s += "Pandas package error.\n"
            s += "Can you import mlacs THEN pandas.\n\n"

        if not istf:
            s += "Tensorflow package error.\n"
            s += "Please install Tensorflow to minimize using CPU.\n\n"

        try:
            import tensorpotential  # noqa
        except ImportError:
            s += "Tensorpotential package error.\n"
            s += "Please install Tensorpotential from:\n"
            s += "https://github.com/ICAMS/TensorPotential.\n\n"

        if not ispyace:  # Github name is python-ace. Pip name is pyace.
            s += "pyace package error.\n"
            s += "Please install it from:\n"
            s += "https://github.com/ICAMS/python-ace\n\n"

        # Test that lammps-user-pace is installed by looking at lmp -help
        lmp_help = shlex.split(self.cmd + " -help")
        grep_pace = shlex.split("grep '\Wpace'")
        res = run(lmp_help, stdout=PIPE)
        out = run(grep_pace, input=res.stdout, stdout=PIPE)
        if out.returncode:  # There was no mention of pace in pair_style
            s += f"Pace style not found : {lmp_pace}\n"
            s += "Lammps exe is given by the environment variable :"
            s += "ASE_LAMMPSRUN_COMMAND\n"
            s += "You can install Pace for Lammps from:\n"
            s += "https://github.com/ICAMS/lammps-user-pace\n\n"

        if s:  # If any import error
            raise ImportError(s)

# ========================================================================== #
    def compute_descriptor(self, atoms, forces=True, stress=False):
        """
        Get the Pyace.catomic.AtomicEnvironemnt
        """
        if stress:
            raise ValueError("Stress are not implemented in ACE")
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        atomic_env = []
        pyacecalc = pyace.asecalc.PyACECalculator(self.bconf, fast_nl=True)
        for at in atoms:
            atomic_env.append(pyacecalc.get_atomic_env(at))
        return atomic_env

# ========================================================================== #
    def _write_lammps_input(self, masses, pbc):
        """
        This one is only used for getting the descriptor
        """
        raise NotImplementedError("yes")
        input_string = "# LAMMPS input file for extracting MLIP descriptors\n"
        input_string += "clear\n"
        input_string += "boundary         "
        for ppp in pbc:
            if ppp:
                input_string += "p "
            else:
                input_string += "f "
        input_string += "\n"
        input_string += "atom_style      atomic\n"
        input_string += "units            metal\n"
        input_string += "read_data        atoms.lmp\n"
        for n1 in range(len(self.masses)):
            input_string += f"mass             {n1+1} {self.masses[n1]}\n"

        input_string += f"pair_style       zero {2*self.rcut}\n"
        input_string += "pair_coeff       * *\n"

        input_string += "thermo         100\n"
        input_string += "timestep       0.005\n"
        input_string += "neighbor       1.0 bin\n"
        input_string += "neigh_modify   once no every 1 delay 0 check yes\n"

        input_string += f"compute      ml all pace {self._ace_opt_str()}\n"
        input_string += "fix          ml all ave/time 1 1 1 c_ml[*] " + \
                        "file descriptor.out mode vector\n"
        input_string += "run              0\n"

        with open(self.folder / "base.in", "w") as fd:
            fd.write(input_string)

# ========================================================================== #
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the descriptor and gradient values
        '''
        lammps_command = self.cmd + ' -in base.in -log none -sc lmp.out'
        lmp_handle = run(shlex.split(lammps_command),
                         stderr=PIPE)

        # There is a bug in LAMMPS that makes compute_mliap crashes at the end
        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    def cleanup(self):
        '''
        Function to cleanup the LAMMPS files used
        to extract the descriptor and gradient values
        '''
        pass

# ========================================================================== #
    def _write_mlip_params(self):
        """
        """
        raise NotImplementedError("No params for ACE... yet")

# ========================================================================== #
    def get_pair_style_coeff(self, folder):
        """
        """
        acefile = folder / "ACE.yace"
        pair_style = "pace"
        pair_coeff = [f"* * {acefile} " +
                      ''.join(self.elements)]
        return pair_style, pair_coeff

# ========================================================================== #
    def get_pair_style(self, folder=None):
        """
        """
        pair_style = "pace"
        return pair_style

# ========================================================================== #
    def get_pair_coeff(self, folder):
        """
        """
        acefile = folder / "ACE.yace"
        pair_coeff = [f"* * {acefile} " +
                      ''.join(self.elements)]
        return pair_coeff

# ========================================================================== #
    def _ace_opt_str(self):
        raise Warning("No options/params allowed for this version of ACE")
        return ""

# ========================================================================== #
    def get_df(self):
        if Path.exists(self.df_fn):
            df = pd.read_pickle(self.df_fn, compression="gzip")
        else:
            df = create_dataframe()
        return df

# ========================================================================== #
    def get_mlip_energy(self, atoms, coef):
        """
        Give the energy, forces, stress of atoms according to the ACE

        Note : GeneralAceFit use ase.calculate which leads to this function
               PyACEFit actually compute the value using atomic environment

        Parameters
        ----------
        coef: :class:`pathlib.Path`
              Path to the ACE potential file

        atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        """
        if not isinstance(atoms, list):
            atoms = [atoms]
        desc = self.compute_descriptor(atoms)

        name = [f"conf{i}" for i in range(len(atoms))]
        df = create_dataframe()
        df = update_dataframe(df=df, name=name, atoms=atoms,
                              atomic_env=desc)

        print(coef)
        scoef = str(coef)
        print(scoef)
   
        bconf = BBasisConfiguration(scoef)
        pyacefit = pyace.PyACEFit(bconf)
        pred = pyacefit.predict(df)
        e = pred['energy_pred'].values[0]
        pd_f = pred['forces_pred'].values
        s = np.zeros(6)

        # f is an array of nconf. Then each conf is a np.array
        f = []
        for force_conf in pd_f:
            f.append(force_conf)
        f = np.array(f)

        return e, f, s

# ========================================================================== #
    def __str__(self):
        txt = " ".join(self.elements)
        txt += " ACE descriptor,"
        txt += f" rcut = {self.rcut}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "ACE descriptor\n"
        txt += f"{(len(txt) - 1) * '-'}\n"
        txt += f"Free atom energy (eV/at): {self.free_at_e}\n"
        txt += f"Dataframe : {self.df_fn}\n"
        txt += f"Tolerance on e : {self.tol_e} (meV/at)\n"
        txt += f"Tolerance on f : {self.tol_f} (meV/ang)\n"
        txt += f"Fitting: {self.fitting}\n"
        txt += f"Backend: {self.backend}\n"
        txt += f"BBasisConfiguration: {self.bconf}\n"

        Warning("I should print the params of the ACE descriptor here")
        return txt
