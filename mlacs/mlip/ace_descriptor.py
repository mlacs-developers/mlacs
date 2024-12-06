"""
// Copyright (C) 2022-2024 MLACS group (AC, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import sys
import os
import shutil
import warnings
import logging

from pathlib import Path
from subprocess import run, PIPE
import shlex

import numpy as np
from ase import Atoms
from ..utilities import make_dataframe

from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.data import atomic_masses, chemical_symbols

from ..core.manager import Manager
from .descriptor import Descriptor
from ..utilities.io_lammps import (LammpsInput, LammpsBlockInput,
                                   get_lammps_command)

try:
    import pandas as pd
    ispandas = True
except ImportError:
    ispandas = False

# Tensorflow Warning if not using GPU
warnings.filterwarnings("ignore", category=Warning, module="tensorflow")
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove GPU warning for tf
    import tensorflow as tf  # noqa
    tf.get_logger().setLevel(logging.ERROR)  # Remove a warning
    istf = True
except ImportError:
    istf = False

try:
    import pyace
    from pyace.generalfit import GeneralACEFit
    from pyace import create_multispecies_basis_config
    from pyace.metrics_aggregator import MetricsAggregator
    ispyace = True
except ImportError:
    ispyace = False

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

def_backend = {'evaluator': 'tensorpot', 'parallel_mode': 'serial',
               'batch_size': 100, 'display_step': 50}

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
    atoms : :class:`ase.atoms` or list
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
            - n_workers : None
            - batch_size: 100
            - display_step: 50
    """
    def __init__(self, atoms, free_at_e, rcut=5.0, tol_e=5, tol_f=25,
                 bconf_dict=None, loss_dict=None, fitting_dict=None,
                 backend_dict=None, nworkers=None):

        if not ispyace:
            raise ValueError("Could not import pyace")

        self.cmd = get_lammps_command()
        self._verify_dependency()

        Descriptor.__init__(self, atoms, rcut)

        self.prefix = "ACE"
        self.desc_name = "ACE"
        self.n_fit_attempt = 3
        self.db_fn = "ACE.pckl.gzip"
        self.pot_fn = "ACE.yace"
        self.tol_e = tol_e
        self.tol_f = tol_f
        self.free_at_e = free_at_e
        bconf = def_bconf if bconf_dict is None else bconf_dict
        self.loss = def_loss if loss_dict is None else loss_dict
        self.fitting = def_fitting if fitting_dict is None else fitting_dict
        self.backend = def_backend if backend_dict is None else backend_dict

        # Kappa such that loss(tol_e) = loss(tol_f)
        self.loss['kappa'] = (tol_e)**2 / ((tol_e)**2 + 3*(tol_f)**2)
        self.fitting['loss'] = self.loss
        self.data = None

        if 'nworkers' not in self.backend and nworkers is not None:
            self.backend['parallel_mode'] = "process"
            self.backend['nworkers'] = nworkers
        if 'elements' not in self.loss:
            bconf['elements'] = self.elements

        # Set bconf['rcut'] according to rcut if rcut is given
        if rcut is not None:
            for bond_type, bond_info in bconf['bonds'].items():
                bond_info['rcut'] = rcut
        else:  # Set rcut according to bconf['rcut'] if rcut is not given
            rcut = 0
            for bond_type, bond_info in bconf['bonds'].items():
                if bond_info['rcut'] > rcut:
                    rcut = bond_info['rcut']
        self.rcut = rcut

        self.bconf = create_multispecies_basis_config(bconf)
        self.acefit = None
        self.log = None

# ========================================================================== #
    @Manager.exec_from_subdir
    def redirect_logger(self):
        """
        Redirect python used by pyace to pyace.log
        """
        file_handler = logging.FileHandler(self.subdir / 'pyace.log')
        # A lot of loggers are involved with pyace
        loggers = [
            __name__,
            'pyace.generalfit',
            'pyace.preparedata',
            'pyace.fitadapter',
            'pyace.metrics_aggregator',
            'numexpr.utils',
            'tensorpotential.fit',
        ]
        logging.getLogger('numexpr.utils').setLevel(logging.CRITICAL)

        # Loop through the loggers and reset the file handler for each
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            while logger.handlers:
                logger.removeHandler(logger.handlers[0])
            logger.addHandler(file_handler)
            logger.propagate = False
        self.log = logging.getLogger(__name__)

# ========================================================================== #
    def get_mlip_params(self):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    @Manager.exec_from_path
    def get_mlip_file(self, folder):
        """
        Read MLIP coefficients from a file.
        """
        filename = Path(folder) / self.pot_fn

        if not filename.is_file():
            filename = filename.absolute()
            raise FileNotFoundError(f"File {filename} does not exist")
        return str(filename)

# ========================================================================== #
    def prepare_wf(self, wf, forces):
        """
        Reshape wf from a flat array to a list of np.array where each
        np.array correspond to a conf i and is of size self.natoms[i]
        """
        new_wf = []
        curr_index = 0
        for f in forces:
            new_wf.append(wf[curr_index:curr_index+len(f)*3])
            curr_index += len(f)*3
        new_wf = [np.sum(np.reshape(new_wf[i], [-1, 3]), axis=1)
                  for i in range(len(new_wf))]
        return new_wf

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def fit(self, atoms, weights, name=None):
        """
        """
        natoms = [len(at) for at in atoms]
        energy = [at.get_potential_energy() - self.calc_free_e(at)
                  for at in atoms]
        forces = [at.get_forces() for at in atoms]
        if name is None:
            name = [f"config{i}" for i in range(len(atoms))]

        # Data preparation
        nconfs = len(natoms)
        we = weights[:nconfs].tolist()
        wf = weights[nconfs:-(nconfs)*6]
        wf = self.prepare_wf(wf, forces)

        atomic_env = self.compute_descriptors(atoms)

        # Dataframe preparation
        df = self.get_df()
        df = make_dataframe(
             df=df, name=name, atoms=atoms, atomic_env=atomic_env,
             energy=energy, forces=forces, we=we, wf=wf)
        df.to_pickle(self.workdir / self.subdir / self.db_fn,
                     compression="gzip")

        # Do the fitting
        if self.acefit is None:
            self.create_acefit()
        else:
            # Dubious implementation of GeneralACEFit,
            # No function to update dataframe once created ??
            self.create_acefit()

        # Note that self.fitting is the same object as self.acefit.fit_config
        real_maxiter = self.fitting['maxiter']
        nattempt = 0
        retry = True
        while retry:
            retry, niter_done = self.actual_fit()
            nattempt += 1
            self.fitting['maxiter'] -= niter_done
            if nattempt > self.n_fit_attempt or self.fitting['maxiter'] < 1:
                retry = False
            if retry:
                self.acefit.fit_config['fit_cycles'] += 1

        self.fitting['maxiter'] = real_maxiter

        fn_yaml = "interim_potential_best_cycle.yaml"
        yace_cmd = f"pace_yaml2yace {fn_yaml} -o {self.pot_fn}"
        self.mlip_model = Path.cwd() / self.pot_fn
        run(shlex.split(yace_cmd), stdout=sys.stdout, stderr=sys.stdout)

        if not Path(self.pot_fn).exists():
            msg = "The ACE fitting wasn't successful\n"
            msg += "If interim_potential_best_cycle.yaml doesn't exist "
            msg += f"in {Path().cwd()} then the ACEfit went wrong.\n"
            msg += f"Else, try this command '{yace_cmd}' inside "
            msg += f"{Path().cwd()}"
            raise RuntimeError(msg)

        return self.pot_fn

# ========================================================================== #
    def actual_fit(self):
        try:
            self.acefit.fit()
        except StopIteration:  # For Scipy < 1.11
            pass
        fit_res = self.acefit.fit_backend.fitter.res_opt

        # Fit failed due to precision loss
        retry = fit_res.status == 2 and not fit_res.success
        niter_done = self.acefit.current_fit_iteration
        return retry, niter_done

# ========================================================================== #
    @Manager.exec_from_subdir
    def set_restart_coefficient(self, mlip_subfolder):
        """
        Restart the calculation with the coefficient from this folder.
        This is because we cannot get back the coefficients from yace
        """
        if str(mlip_subfolder).endswith(".yace"):
            path = Path(mlip_subfolder).with_suffix(".yaml")
            if not path.exists():
                new_fn = "interim_potential_best_cycle.yaml"
                path = Path(mlip_subfolder).parent / new_fn
            if not path.exists():
                raise ValueError("Cannot find yaml file to get coefficients")
        fn = str(path)
        self.bconf.load(fn)
        fc = int(pyace.BBasisConfiguration(fn).metadata["_fit_cycles"])
        self.fitting['fit_cycles'] = fc+1

# ========================================================================== #
    @Manager.exec_from_subdir
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
            msg = f"Step: {last_fit_metric_data['iter_num']}   "
            msg += f"RMSE Energy: {e:.4f} (meV/at), "
            msg += f"RMSE Forces: {f:.4f} (meV/ang)"
            self.log.info(msg)
            if iter_num > 0 and self.tol_e >= e and self.tol_f >= f:
                s = "Convergence reached:\n"
                s += f"RMSE of energy: {e:.8f} < {self.tol_e} (meV/at)\n"
                s += f"RMSE of forces: {f:.8f} < {self.tol_f} (meV/ang)\n"
                self.log.info(s)
                raise StopIteration("My convergence reached")

        self.callback = lambda val: check_conv(val)
        self.data = dict(filename=str(self.db_fn))
        self.acefit = GeneralACEFit(potential_config=self.bconf,
                                    fit_config=self.fitting,
                                    data_config=self.data,
                                    backend_config=self.backend)
        self.acefit.fit_backend.fit_metrics_callback = self.callback

# ========================================================================== #
    def calc_free_e(self, atoms):
        """
        Calculate the contribution which isn't the binding energy
        for this particular configuration
        """
        e = 0
        for atom in atoms:
            e_tmp = self.free_at_e[atom.symbol]
            e += e_tmp
        return e

# ========================================================================== #
    @Manager.exec_from_path
    def predict(self, atoms, coef, folder):
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if Path("Atoms").exists():
            shutil.rmtree("Atoms")
        Path("Atoms").mkdir()

        clear_loop = False  # If we use clear or delete_atoms in lammps input
        for i, at in enumerate(atoms):
            if np.any(at.get_pbc() != atoms[0].get_pbc()):
                raise ValueError("PBC cannot change between states")
            if np.any(at.cell.array != atoms[0].cell.array):
                clear_loop = True
            lmp_atfname = f"Atoms/atoms{i+1}.lmp"

            write_lammps_data(lmp_atfname,
                              at,
                              specorder=self.elements.tolist())

        self._write_lammps_input(atoms, coef, is_clear=clear_loop)

        self._run_lammps(lmp_atfname)
        e, f, s = self._read_lammps_output(atoms)
        self.cleanup(len(atoms))

        if len(e) == 1:
            return e[0], f[0], s[0]

        return e, f, s

# ========================================================================== #
    @Manager.exec_from_path
    def _read_lammps_output(self, atoms):
        with open("lammps.out", "r") as f:
            lines = f.readlines()

        toparse = []
        tofind = "Step          Temp          E_pair         Press"
        for idx, line in enumerate(lines):
            if tofind in line:
                # "WARNING:" may appear between Step and value
                while "WARNING" in lines[idx+1]:
                    idx += 1
                toparse.append(lines[idx+1].split())
        if len(toparse) != len(atoms):
            raise ValueError("lammps.out a thermo for each atoms")

        e, f, s = [], [], []
        for i in range(len(toparse)):
            thermo, at = toparse[i], atoms[i]
            e_correction = self.calc_free_e(at)  # To only get cohesion energy
            bar2GPa = 1e-4
            e.append((float(thermo[2]) + e_correction)/len(at))
            f.append(np.array(read(f'forces{i+1}.out',
                              format='lammps-dump-text').get_forces()))
            s.append(np.loadtxt(f'stress{i+1}.out', skiprows=4)[:, 1]*bar2GPa)
        return e, f, s

# ========================================================================== #
    @Manager.exec_from_path
    def cleanup(self, natoms):
        '''
        Function to cleanup the LAMMPS files used
        to extract the descriptor and gradient values
        '''
        for i in range(natoms):
            Path(f"forces{i+1}.out").unlink()
            Path(f"stress{i+1}.out").unlink()
        Path("lammps.out").unlink()
        Path("lammps.in").unlink()
        shutil.rmtree("Atoms")

# ========================================================================== #
    @Manager.exec_from_path
    def _write_lammps_input(self, atoms, coef, is_clear=True):
        """
        Create one lammps input to evaluate E,F,S of multiple atoms
        Note : We need to use clear if the cell changes between atoms
               Else, we can use delete_atoms which doesn't reload potential
        """
        masses = [atomic_masses[chemical_symbols.index(el)]
                  for el in self.elements]

        # Write the input file
        txt = "LAMMPS input file for extracting MLIP E, F, S"
        lmp_in = LammpsInput(txt)

        block_loopstart = LammpsBlockInput("loopstart", "Loop starts")
        block_loopstart("loop", f"variable a loop {len(atoms)}")
        block_loopstart("label", "label loopstart")
        if is_clear:
            # loopstart, init, atoms_data, pot, compute1, compute2, loopback
            block_loopstart("clear data", "clear")
            lmp_in("loop", block_loopstart)
            read_data = "read_data Atoms/atoms${a}.lmp"
        else:
            # init, atoms_data, pot, compute1, loopstart, compute2, loopback
            read_data = "read_data Atoms/atoms1.lmp"

        block = LammpsBlockInput("init", "Initialization")
        pbc_txt = "{0} {1} {2}".format(
                *tuple("sp"[int(x)] for x in atoms[0].get_pbc()))
        block("boundary", f"boundary {pbc_txt}")
        block("atom_style", "atom_style  atomic")
        block("units", "units metal")
        block("read_data", read_data)
        for i, m in enumerate(masses):
            block(f"mass{i}", f"mass   {i+1} {m}")
        lmp_in("init", block)

        block = LammpsBlockInput("interaction", "Interactions")
        block("pair_style", "pair_style pace")
        pc = f"pair_coeff * * {coef} {' '.join(self.elements)}"
        block("pair_coeff", pc)
        lmp_in("interaction", block)

        block = LammpsBlockInput("compute1", "Compute")
        block("thermo_style", "thermo_style custom step temp epair press")
        block("cp", "compute svirial all pressure NULL virial")
        lmp_in("compute1", block)

        if not is_clear:
            block_loopstart("delat", "delete_atoms group all")
            block_loopstart("rd", "read_data Atoms/atoms${a}.lmp add append")
            lmp_in("loop", block_loopstart)

        block = LammpsBlockInput("compute2", "Compute")
        block("fs", "fix svirial${a} all ave/time 1 1 1 c_svirial" +
              " file stress${a}.out mode vector")
        block("dump", "dump dump_force${a} all custom 1 forces${a}.out" +
              " id type x y z fx fy fz")
        block("run", "run 0")
        lmp_in("compute2", block)

        block = LammpsBlockInput("loopback", "Loop Back")
        block("iterate", "next a")
        block("loopend", "jump SELF loopstart")
        lmp_in("loopback", block)

        with open("lammps.in", "w") as fd:
            fd.write(str(lmp_in))

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
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(gpus) == 0:
                # ON: Not great as MLACS logger hasn't been initialized
                mlacs_log = logging.getLogger('mlacs')
                mlacs_log.warn("No GPUs found, will fit on CPUs\n")
        except ImportError:
            s += "Tensorpotential package error.\n"
            s += "Please install Tensorpotential from:\n"
            s += "https://github.com/ICAMS/TensorPotential.\n\n"

        # Test that lammps-user-pace is installed by looking at lmp -help
        lmp_help = shlex.split(self.cmd + " -help")
        grep_pace = shlex.split(r"grep '\Wpace'")
        res = run(lmp_help, stdout=PIPE)
        out = run(grep_pace, input=res.stdout, stdout=PIPE)
        if out.returncode:  # There was no mention of pace in pair_style
            s += "Pace style not found.\n"
            s += "You can install Pace for Lammps from:\n"
            s += "https://github.com/ICAMS/lammps-user-pace\n\n"

        if s:  # If any import error
            raise ImportError(s)

# ========================================================================== #
    def compute_descriptor(self, atoms, forces=True, stress=False):
        """
        Get the Pyace.catomic.AtomicEnvironemnt
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        atomic_env = []
        pyacecalc = pyace.asecalc.PyACECalculator(self.bconf, fast_nl=True)
        for at in atoms:
            atomic_env.append(pyacecalc.get_atomic_env(at))
        return atomic_env

# ========================================================================== #
    @Manager.exec_from_path
    def _run_lammps(self, lmp_atoms_fname):
        '''
        Function that call LAMMPS to extract the descriptor and gradient values
        '''
        lammps_command = self.cmd + ' -in lammps.in -log none -sc lammps.out'
        lmp_handle = run(shlex.split(lammps_command),
                         stderr=PIPE)

        # There is a bug in LAMMPS that makes compute_mliap crashes at the end
        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    @Manager.exec_from_path
    def _write_mlip_params(self):
        """
        """
        raise NotImplementedError("No params for ACE... yet")

# ========================================================================== #
    def get_pair_style_coeff(self):
        """
        """
        pair_style = self.get_pair_style()
        pair_coeff = self.get_pair_coeff()
        return pair_style, pair_coeff

# ========================================================================== #
    def get_pair_style(self):
        """
        """
        pair_style = "pace"
        return pair_style

# ========================================================================== #
    def get_pair_coeff(self):
        """
        """
        acefile = self.get_filepath('.yace')
        pair_coeff = [f"* * {acefile} " +
                      ' '.join(self.elements)]
        return pair_coeff

# ========================================================================== #
    def _ace_opt_str(self):
        raise Warning("No options/params allowed for this version of ACE")
        return ""

# ========================================================================== #
    @Manager.exec_from_path
    def get_df(self):
        """
        Return the pandas.DataFrame object associated to this AceDescriptor
        """
        db_fn = Path(self.get_filepath(".pckl.gzip"))
        if db_fn.exists():
            df = pd.read_pickle(self.db_fn, compression="gzip")
        else:
            df = None
        return df

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
        txt += f"Rcut: {self.rcut} ang\n"
        txt += f"Free atom energy (eV/at): {self.free_at_e}\n"
        txt += f"Dataframe : {self.db_fn}\n"
        txt += f"Tolerance on e : {self.tol_e} (meV/at)\n"
        txt += f"Tolerance on f : {self.tol_f} (meV/ang)\n"
        txt += f"Fitting: {self.fitting}\n"
        txt += f"Backend: {self.backend}\n"
        txt += f"BBasisConfiguration: {self.bconf}\n"
        return txt
