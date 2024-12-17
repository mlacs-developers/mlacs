"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ase.atoms import Atoms
from ase.io import read, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from .core import Manager
from .mlip import LinearPotential, MliapDescriptor
from .calc import CalcManager
from .state import StateManager
from .utilities.log import MlacsLog
from .utilities import create_random_structures, save_cwd
from .utilities.io_abinit import OtfMlacsHist
from .properties import PropertyManager, RoutinePropertyManager


# ========================================================================== #
# ========================================================================== #
class Mlas(Manager):
    r"""
    A Learn on-the-fly simulation constructed in order to sample approximate
    distribution

    Parameters
    ----------

    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        the atom object on which the simulation is run.

    state: :class:`StateManager` or :class:`list` of :class:`StateManager`
        Object determining the state to be sampled

    calc: :class:`ase.calculators` or :class:`CalcManager`
        Class controlling the potential energy of the system
        to be approximated.
        If a :class:`ase.calculators` is attached, the :class:`CalcManager`
        is automatically created.

    mlip: :class:`MlipManager` (optional)
        Object managing the MLIP to approximate the real distribution
        Default is a LammpsMlip object with a snap descriptor,
        ``5.0`` angstrom rcut with ``8`` twojmax.

    prop: :class:`PropertyManager` or :class:`list` or :class:`CalcProperty`
    (optional)
        Object managing the MLIP to approximate the real distribution
        Default is a LammpsMlip object with a snap descriptor,
        ``5.0`` angstrom rcut with ``8`` twojmax.

    neq: :class:`int` (optional)
        The number of equilibration iteration. Default ``10``.

    confs_init: :class:`int` or :class:`list` of :class:`ase.Atoms` (optional)
        If :class:`int`, Number of configurations used to train a preliminary
        MLIP. The configurations are created by rattling the first structure.
        If :class:`list` of :class:`ase.Atoms`, The atoms that are to be
        computed in order to create the initial training configurations.
        Default ``None``.

    std_init: :class:`float` (optional)
        Variance (in :math:`\mathring{a}^2`) of the displacement
        when creating initial configurations.
        Default :math:`0.05 \mathring{a}^2`

    keep_tmp_mlip: :class:`Bool` (optional)
        Keep every generated MLIP. If True and using MBAR, a restart will
        recalculate every previous MLIP.weight using the old coefficients.
        Default ``True``.

    workdir: :class:`str` (optional)
        The directory in which to run the calculation.

    prefix: :class:`str` (optional)
        The prefix to prepend the name of the States files.

    ncprefix: :class:`str` (optional)
        The prefix to prepend the name of the *HIST.nc file.
        Script name format: ncprefix + scriptname + '_HIST.nc'.
        Default `''`.

    ncformat: :class:`str` (optional)
        The format of the *HIST.nc file. One of the five flavors of netCDF
        files format available in netCDF4 python package: 'NETCDF3_CLASSIC',
        'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA','NETCDF4_CLASSIC',
        'NETCDF4'.
        Default ``NETCDF3_CLASSIC``.
    """

    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 prop=None,
                 neq=10,
                 confs_init=None,
                 std_init=0.05,
                 keep_tmp_mlip=True,
                 workdir='',
                 prefix='',
                 ncprefix='',
                 ncformat='NETCDF3_CLASSIC'):

        Manager.__init__(self, workdir=workdir, prefix=prefix)

        # Initialize working directory
        self.workdir.mkdir(exist_ok=True, parents=True)
        self.ncfile = None

        ##############
        # Check inputs
        ##############
        self.keep_tmp_mlip = keep_tmp_mlip
        self._initialize_state(state, atoms, neq, prefix)
        self._initialize_calc(calc)
        self._initialize_mlip(mlip)
        # self._initialize_properties(prop)

        # Miscellanous initialization
        self.rng = np.random.default_rng()

        # Check if trajectory files already exists
        self.launched = self._check_if_launched()

        # Create Abinit-style *HIST.nc file of netcdf format
        self.ncfile = OtfMlacsHist(ncprefix=ncprefix,
                                   workdir=workdir,
                                   ncformat=ncformat,
                                   launched=self.launched,
                                   atoms=self.atoms)
        if self.ncfile.unique_atoms_type:
            self._initialize_routine_properties()

        self._initialize_logger()

        # Initialize momenta and parameters for initial/training configs
        if not self.launched:
            self._initialize_momenta()
            self.traj = []
            self.confs_init = confs_init or []
            self.std_init = std_init
            self.nconfs = [0] * self.nstate
            self.uniq_at = None
            self.idx_computed = None
            self.nb_distinct_conf = None

        # Reinitialize everything from the trajectories
        # Compute fitting data - get trajectories - get current configurations
        else:
            self.restart_from_traj()

        self.step = 0
        self.log._delimiter()
        self._write("Starting the simulation", True)

# ========================================================================== #
    @Manager.exec_from_workdir
    def run(self, nsteps=100):
        """
        Run the algorithm for nsteps
        """
        isearlystop = False
        while self.step < nsteps:
            if self._check_early_stop():
                isearlystop = True
                break
            self.log.init_new_step(self.step)
            if not self.launched:
                self._run_initial_step()
                self.step += 1
            else:
                step_done = self._run_step()
                if not step_done:
                    pass
                else:
                    self.step += 1

        self.log.write_end(isearlystop)
        # Here we have to add some info about the end of the simulation
        self.log.write_footer()

# ========================================================================== #
    def _run_step(self):
        """
        Run a regular step of the algorithm.

        A regular step consists in:
           - Train (fit) the MLIP,
           - Run MLMD with trained MLIP during nsteps,
           - Compute the true potential energy/forces/stresses,
           - Update database and save new configs.

        Returns
        ----------

        `partial_success`: :class:`bool`
            `True` if at least one state has been successfully computed by the
            true potential after MLMD, `False` otherwise.
            See also _handle_potential_error() method.
        """

        eq = self._is_eq_step()

        # Train MLIP
        self._write("Training new MLIP")
        self._set_mlip_subfolder()
        self.mlip.train_mlip()

        # Run MLMD with as many threads as there are states
        self._write("Running MLMD")
        atoms_mlip = self._create_mlip_atoms(eq)
        futures = []
        with save_cwd(), ThreadPoolExecutor() as executor:
            for istate in range(self.nstate):
                exe = executor.submit(self.state[istate].run_dynamics,
                                      *(atoms_mlip[istate],
                                        self.mlip.pair_style,
                                        self.mlip.pair_coeff,
                                        self.mlip.model_post,
                                        self.mlip.atom_style,
                                        eq[istate],
                                        self.mlip.get_elements()))
                futures.append(exe)
                msg = f"State {istate+1}/{self.nstate} has been launched"
                self._write(msg)
            for istate, exe in enumerate(futures):
                atoms_mlip[istate] = exe.result()
                if self.keep_tmp_mlip:
                    mm = self.mlip.descriptor.subsubdir
                    atoms_mlip[istate].info['parent_mlip'] = str(mm)
            executor.shutdown(wait=True)

        # Computing energy with true potential
        self._write("Computing energy with the True potential")
        # TODO GA: Might be better to do the threading at this level,
        #          up from calc.compute_true_potential.
        subfolder_l = [s.subfolder for s in self.state]
        step_l = [max(self.nconfs)] * self.nstate
        atoms_true = self.calc.compute_true_potential(atoms_mlip,
                                                      subfolder_l,
                                                      step=step_l)
        partial_success = self._handle_potential_error(atoms_true)

        # SinglePointCalculator to bypass the calc attach to atoms thing of ase
        for at in atoms_mlip:
            at.calc = self.mlip.get_calculator()
            at.calc = SinglePointCalculator(at.copy(),
                                            energy=at.get_potential_energy(),
                                            forces=at.get_forces(),
                                            stress=at.get_stress())

        # Update database / Save new configs in trajectory files
        attrue = self.add_traj_descriptors(atoms_true)
        for i, (attrue, atmlip) in enumerate(zip(atoms_true, atoms_mlip)):
            if attrue is not None:
                self.mlip.update_matrices(attrue)
                self.traj[i].write(attrue)
                self.atoms[i] = attrue

                prefix = self.state[i].prefix
                filepath = self.workdir / (prefix + "_potential.dat")
                with open(filepath, "a") as f:
                    f.write("{:20.15f}   {:20.15f}\n".format(
                             attrue.get_potential_energy(),
                             atmlip.get_potential_energy()))
                self.nconfs[i] += 1

        # TODO: CD: Implement groups in netcdf for Atoms with difft chem. form.
        if self.ncfile.unique_atoms_type:
            self._compute_properties()
            self._compute_routine_properties()
        self._execute_post_step()

        return partial_success

# ========================================================================== #
    def _run_initial_step(self):
        """
        Run the initial step.

        Notes
        -----

        This routine unfolds in three stages.

            (i) Compute initial configurations.
            `Initial` refers to the atomic configurations `atoms` given
            as a parameter to the `Mlas` object. These confs are assigned to
            self.atoms and self.uniq_at, and get stored in `Trajectory.traj` at
            the end of the routine.

                NB: `Trajectory.traj` becomes `Trajectory_i.traj` if there are
                several states, or more generally `prefix.traj` as defined in
                _initialize_state().

            (ii) [Conditionnal] Compute training configurations.
            `Training` refers to configurations that serve as complementary
            configurations ensuring the database is large enough to fit an
            MLIP. These confs are assigned to self.confs_init and are saved in
            `Training_configurations.traj`.

                NB: This stage is only executed if the MLIP is empty. This is
                intentional, as MLACS enables a non-empty MLIP to be loaded at
                the start of the calculation (independently of restarts). Note
                how, in that case, the `initial` configs are still computed.

                NB: There are several options to set these training confs. One
                of them is to load them from an existing *.traj file.

            (iii) Update MLIP object with new confs from step (i) and (ii).
        """
        self._write("\nRunning initial step\n")

        # Compute initial configurations
        self._get_unique_atoms()
        self._compute_initial_confs()
        self._distribute_atoms()
        self._create_initial_traj()

        # Load/compute training configurations (if empty MLIP)
        if self.mlip.nconfs == 0:
            conf_fname = str(self.workdir / "Training_configurations.traj")
            if self._is_traj_file(conf_fname):
                self._load_training_confs(conf_fname)
            else:
                self._initialize_training_confs()
                self._compute_training_confs()
                self._create_training_traj()

        # Add initial/training configurations to database
        for at in self.confs_init + self.uniq_at:
            self.mlip.update_matrices(at)

        self.launched = True

# ========================================================================== #
    def _initialize_calc(self, calc):
        """Create calculator object"""
        if isinstance(calc, Calculator):
            self.calc = CalcManager(calc)
        elif isinstance(calc, CalcManager):
            self.calc = calc
        else:
            msg = "calc should be a ase Calculator object or " + \
                  "a CalcManager object"
            raise TypeError(msg)
        self.calc.workdir = self.workdir

# ========================================================================== #
    def _initialize_mlip(self, mlip):
        """Create mlip object"""
        if mlip is None:
            descriptor = MliapDescriptor(self.atoms[0], 5.0)
            self.mlip = LinearPotential(descriptor)
        else:
            self.mlip = mlip

        self.mlip.workdir = self.workdir
        if not self.mlip.folder:
            self.mlip.folder = 'MLIP'
        self.mlip.descriptor.workdir = self.workdir
        self.mlip.descriptor.folder = self.mlip.folder
        self.mlip.weight.workdir = self.workdir
        self.mlip.weight.folder = self.mlip.folder
        self.mlip.subdir.mkdir(exist_ok=True, parents=True)

# ========================================================================== #
    def _initialize_properties(self, prop):
        """Create property object"""
        self.prop = PropertyManager(prop)

# ========================================================================== #
    def _initialize_routine_properties(self):
        """Create routine property object"""
        self.routine_prop = RoutinePropertyManager(self.ncfile,
                                                   self.launched)

# ========================================================================== #
    def _initialize_momenta(self):
        """Create property object"""
        for i in range(self._nmax):
            self.state[i].initialize_momenta(self.atoms[i])
            prefix = self.state[i].prefix
            pot_fname = self.workdir / (prefix + "_potential.dat")
            with open(pot_fname, "w") as f:
                f.write("# True epot [eV]          MLIP epot [eV]\n")
        self.prefix = ''

# ========================================================================== #
    def _initialize_state(self, state, atoms, neq, prefix='Trajectory'):
        """
        Function to initialize the state
        """
        # Put the state(s) as a list
        if isinstance(state, StateManager):
            self.state = [state]
        if isinstance(state, list):
            self.state = state
        self.nstate = len(self.state)

        for s in self.state:
            s.workdir = self.workdir
            s.folder = 'MolecularDynamics'
            if not s.subfolder:
                s.subfolder = prefix
            if not s.prefix:
                s.prefix = prefix

        if self.nstate > 1:
            for i, s in enumerate(self.state):
                s.subfolder = s.subfolder + f"_{i+1}"
                s.prefix = s.prefix + f"_{i+1}"

        self.atoms = []
        # Create list of atoms
        if isinstance(atoms, Atoms):
            for _ in range(self.nstate):
                self.atoms.append(atoms.copy())
        elif isinstance(atoms, list):
            e = "You should have 1 atoms per state"
            assert len(atoms) == self.nstate, e
            self.atoms = [at.copy() for at in atoms]
        else:
            msg = "atoms should be a ASE Atoms object or " + \
                  "a list of ASE atoms objects"
            raise TypeError(msg)
        self.atoms_start = [at.copy() for at in self.atoms]

        # Create list of neq -> number of equilibration
        # mlmd runs for each state
        if isinstance(neq, int):
            self.neq = [neq] * self.nstate
        elif isinstance(neq, list):
            assert len(neq) == self.nstate
            self.neq = self.nstate
        else:
            msg = "neq should be an integer or a list of integers"
            raise TypeError(msg)

# ========================================================================== #
    def _initialize_logger(self):
        """Initialize logger."""
        self.log = MlacsLog(str(self.workdir / "MLACS.log"), self.launched)
        # XXX: self.logger: useless line?
        # self.logger = self.log.logger_log
        self._write()
        self.log._delimiter()
        self._write("Recap of the simulation parameters", True)
        self._write()
        self._write("Recap of the states", False, True)
        for i in range(self.nstate):
            self._write(f"State {i+1}/{self.nstate} :")
            self._write(repr(self.state[i]))
            self._write()
        self._write()
        self._write("Recap of the calculator", False, True)
        msg = self.calc.log_recap_state()
        self._write(msg)
        self._write()
        self._write("Recap of the MLIP", False, True)
        self._write(repr(self.mlip))
        self._write()

        # Share logger instance to mlip and calc objects.
        self.mlip.logger = self.log
        self.calc.logger = self.log

# ========================================================================== #
    def _check_if_launched(self):
        """
        Function to check simulation restarts:
         - Check if trajectory files exist and are not empty.
         - Check if the number of configuration found is at least two.

        Returns
        ----------

        `bool`
            Boolean indicating if the conditions are met.
        """
        # Build a list of all *.traj files to test
        all_traj_files = [str(self.workdir / "Training_configurations.traj")]
        for i in range(self._nmax):
            traj_fname = str(self.workdir / (self.state[i].prefix + ".traj"))
            all_traj_files.append(traj_fname)

        # Check all possible *.traj files
        _nat_init = 0
        for traj_fname in all_traj_files:
            if self._is_traj_file(traj_fname):
                _nat_init += len(read(traj_fname, index=':'))
            else:
                return False

        return _nat_init >= 2

# ========================================================================== #
    def _is_eq_step(self):
        """
        Distinguish between equilibration and production steps for the MLMD.

        Returns
        ----------

        `eq`: :class:`list` of :class:`bool`
            Element in `eq` is True if this is an equilibration step.
        """
        self._write()
        eq = []
        for istate in range(self.nstate):
            trajstep = self.nconfs[istate]
            if self.nconfs[istate] < self.neq[istate]:
                eq.append(True)
                msg = f"Equilibration step for state {istate+1}, "
            else:
                eq.append(False)
                msg = f"Production step for state {istate+1}, "
            msg += f"{trajstep} configurations for this state"
            self._write(msg)
        self._write()
        return eq

# ========================================================================== #
    def _set_mlip_subfolder(self):
        """
        Set mlip subfolder.

        Notes
        ----------

        The branching here results from two situations:

        (1) One wants to keep all MLIP data (i.e., for each step),

        (2) One is only interested in the last MLIP data, and therefore
            routinely overwritting MLIP data is fine.
        """
        if self.keep_tmp_mlip:
            self.mlip.subfolder = f"Coef{max(self.nconfs)}"
        else:
            self.mlip.subfolder = ''

# ========================================================================== #
    def _create_mlip_atoms(self, eq):
        """
        Create MLIP atoms object.

        Returns
        ----------

        `atoms_mlip` :class:`list` of :class:`ase.Atoms`
            The list of Atoms objects on which MLMD will be performed.
            Can be either
                - the initial configs (if restart or equilibration steps)
                - the last `self.atoms` configs (reinstantiated for precaution)
        """
        atoms_mlip = [None]*self.nstate
        for istate in range(self.nstate):
            if self.state[istate].isrestart or eq[istate]:
                self._write(" -> Starting from first atomic configuration")
                atoms_mlip[istate] = self.atoms_start[istate].copy()
                self.state[istate].initialize_momenta(atoms_mlip[istate])
            else:
                atoms_mlip[istate] = self.atoms[istate].copy()
        return atoms_mlip

# ========================================================================== #
    def _handle_potential_error(self, atoms_true):
        """
        True potential error handling.

        Returns
        ----------

        `bool`
            `True` if at least one state has been successfully computed by the
            true potential after MLMD, `False` otherwise.
        """
        partial_success = True
        nerror = 0
        for i, at in enumerate(atoms_true):
            if at is None:
                msg = f"For state {i+1}/{self._nmax} calculation with " + \
                       "the true potential resulted in error " + \
                       "or didn't converge"
                self._write(msg)
                nerror += 1
        if nerror == self.nstate:
            msg = "All true potential calculations failed, " + \
                  "restarting the step"
            self._write(msg)
            partial_success = False
        return partial_success

# ========================================================================== #
    def _is_traj_file(self, trajname):
        """
        Check if Trajectory file exists and is readable.

        Parameters
        ----------

        trajname: :class:`str`
            The *.traj file being tested.

        Returns
        ----------

        `bool`
            Boolean indicating if the conditions are met.
        """
        if not os.path.isfile(trajname):
            return False
        try:
            read(trajname, index=":")
            return True
        except UnknownFileTypeError:
            return False

# ========================================================================== #
    def add_traj_descriptors(self, atoms):
        """
        Add descriptors in trajectory files.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        desc = self.mlip.descriptor.compute_descriptors(atoms)
        for iat, at in enumerate(atoms):
            at.info['descriptor'] = desc[iat]
        return atoms

# ========================================================================== #
    @Manager.exec_from_workdir
    def restart_from_traj(self):
        """
        Restart a calculation from previous trajectory files
        """
        train_traj, prev_traj = self.read_traj()

        #for i in range(self._nmax):
        #    self.state[i].subsubdir.mkdir(exist_ok=True, parents=True)

        # Add the Configuration without a MLIP generating them
        if train_traj is not None:
            for i, conf in enumerate(train_traj):
                self._write(f"Configuration {i+1} / {len(train_traj)}")
                self.mlip.update_matrices(conf)  # We add training conf

        # Add all the configuration of trajectories traj
        self._write("Adding previous configuration iteratively")
        # GA: TODO We dont actually need parent_list. Remove this variable.
        parent_list, mlip_coef = self.mlip.read_parent_mlip(prev_traj)

        # Directly adding initial conf to have it once even if multistate
        atoms_by_mlip = [[] for _ in range(len(parent_list))]
        no_parent_atoms = [prev_traj[0][0]]

        for istate in range(self.nstate):
            for iconf in range(1, len(prev_traj[istate])):
                if "parent_mlip" in prev_traj[istate][iconf].info:
                    pm = prev_traj[istate][iconf].info['parent_mlip']
                    idx = parent_list.index(pm)
                    atoms_by_mlip[idx].append(prev_traj[istate][iconf])
                else:
                    no_parent_atoms.append(prev_traj[istate][iconf])

        for conf in no_parent_atoms:
            self.mlip.update_matrices(conf)

        # If the last simulation was with keep_tmp_mlip=False,
        # we put the old MLIP.model and weight in a Coef folder
        can_use_weight = self.mlip.can_use_weight
        if len(no_parent_atoms) > 1 and self.keep_tmp_mlip and can_use_weight:
            self._write("Some configuration in Trajectory have no parent_mlip")
            self._write("You should rerun this simulation with DatabaseCalc")

            fm = self.mlip.subdir / "MLIP.model"
            fw = self.mlip.subdir / "MLIP.weight"

            last_coef = max(self.nconfs)-1
            self.mlip.subfolder = f"Coef{last_coef}"

            if os.path.isfile(fm):
                if not os.path.exists(self.mlip.subsubdir):
                    self.mlip.subsubdir.mkdir(exist_ok=True, parents=True)
                    os.rename(fm, self.mlip.subsubdir / "MLIP.model")
                    os.rename(fw, self.mlip.subsubdir / "MLIP.weight")

        curr_step = 0
        for i in range(len(atoms_by_mlip)):
            curr_step += 1

            # GA: Since we don't read
#           self.mlip.subsubdir = Path(atoms_by_mlip[i][0].info['parent_mlip'])
            self.mlip.next_coefs(mlip_coef[i])
            for at in atoms_by_mlip[i]:
                self.mlip.update_matrices(at)
        self.mlip.subfolder = ''

        # Update this simulation traj
        self.traj = []
        self.atoms = []

        for i in range(self._nmax):
            traj_fname = str(self.workdir / (self.state[i].prefix + ".traj"))
            self.traj.append(Trajectory(traj_fname, mode="a"))
            self.atoms.append(prev_traj[i][-1])
        del prev_traj

# ========================================================================== #
    def read_traj(self):
        """
        Read Trajectory files from previous simulations
        """
        self._write("Adding previous configurations to the training data")

        conf_fname = str(self.workdir / "Training_configurations.traj")
        if os.path.isfile(conf_fname):
            train_traj = Trajectory(conf_fname, mode="r")
            self._write(f"{len(train_traj)} training configurations\n")
        else:
            train_traj = None

        prev_traj = []
        lgth = []
        for i in range(self._nmax):
            traj_fname = str(self.workdir / (self.state[i].prefix + ".traj"))
            prev_traj.append(Trajectory(traj_fname, mode="r"))
            lgth.append(len(prev_traj[i]))
        self.nconfs = lgth
        self._write(f"{np.sum(lgth)} configuration from trajectories")
        return train_traj, prev_traj

# ========================================================================== #
    @property
    def _nmax(self):
        return self.nstate

# ========================================================================== #
    def _check_early_stop(self):
        """
        Break the self consistent procedure.
        """
        return self.prop.check_criterion

# ========================================================================== #
    def _execute_post_step(self):
        """
        Function to execute some things that might be needed for a specific
        mlas object.
        For example, computing properties
        """
        pass

# ========================================================================== #
    def _compute_properties(self):
        """
        Function to execute and converge on specific Properties.
        For example, CalcRdf, CalcTI ...
        """
        pass

# ========================================================================== #
    def _compute_routine_properties(self):
        """Compute routine properties"""
        self.routine_prop.calc_initialize(atoms=self.atoms)
        msg = self.routine_prop.run(self.step)
        self.log.logger_log.info(msg)
        self.routine_prop.save_prop(self.step)
        self.routine_prop.save_weighted_prop(self.step, self.mlip.weight)
        self.routine_prop.save_weights(self.step,
                                       self.mlip.weight,
                                       self.ncfile.ncformat)

# ========================================================================== #
    def _get_unique_atoms(self):
        """
        Compute list of unique atoms and corresponding indices.

        Notes
        -----

        The initial `atoms` list, self.atoms, may be redundant (e.g., if state-
        averaging is sought during MD). To avoid calculating the same true
        potential several times, a list of `unique` configurations (two-by-two
        distinct `Atoms` objects) is obtained in this function.
        """
        uniq_at, idx_computed = [], []
        for istate, at in enumerate(self.atoms):
            for i, unique_at in enumerate(uniq_at):
                if at == unique_at:
                    idx_computed[i].append(istate)
                    break  # Found identical conf, exit loop
            else:  # No identical conf. found, so add to list of unique confs
                uniq_at.append(at)
                idx_computed.append([istate])
        self.idx_computed = idx_computed
        self.nb_distinct_conf = len(uniq_at)
        self.uniq_at = uniq_at

        self._write(f"Number of distinct config.: {self.nb_distinct_conf}")

# ========================================================================== #
    def _distribute_atoms(self):
        """
        Update `self.atoms` with computed properties gathered in `uniq_at`, by
        distributing unique atoms properties to all atoms.
        """
        for iun, at in enumerate(self.uniq_at):
            for icop in self.idx_computed[iun]:
                # All atoms/calc are reinstantiated
                newat = at.copy()
                calc = SinglePointCalculator(newat,
                                             energy=at.get_potential_energy(),
                                             forces=at.get_forces(),
                                             stress=at.get_stress())
                newat.calc = calc
                self.atoms[icop] = newat
                # XXX: [CD] does this reinstantiation serve any purpose at all?
                # It seems that the fate of self.atoms[*] is to be written in
                # *.traj files, but self.atoms[*] is never modified itself.
                # As a matter of fact, the instruction
                # self.atoms[icop] = at
                # where unique atoms share the same atom/calc instances, pass
                # all the tests.

# ========================================================================== #
    def _create_initial_traj(self):
        """
        Create the .traj files.
        Write the initial configurations (stored in self.atoms) in .traj files.
        """
        self._write("Creating trajectories")
        for istate in range(self.nstate):
            prefix = self.state[istate].prefix
            self.traj.append(Trajectory(prefix + ".traj", mode="w"))  # Create
            self.traj[istate].write(self.atoms[istate])  # Update
            self.nconfs[istate] += 1

# ========================================================================== #
    def _compute_initial_confs(self):
        """
        Compute true potential properties of (unique) initial configurations.
        """
        subfolder_l = ["Initial"] * self.nb_distinct_conf
        istep = np.arange(self.nb_distinct_conf, dtype=int)
        self._write("Computing true potential energy [initial configs]")
        self.uniq_at = self.calc.compute_true_potential(self.uniq_at,
                                                        subfolder_l,
                                                        istep)
        if any(at is None for at in self.uniq_at):
            msg = "True potential calculation failed or didn't converge"
            raise TruePotentialError(msg)
        self._write("Computation done")

# ========================================================================== #
    def _initialize_training_confs(self):
        """Create training configurations"""
        if self.confs_init == []:
            # By default, only one training configuration is created
            self.confs_init = create_random_structures(self.uniq_at,
                                                       self.std_init,
                                                       1)
        elif isinstance(self.confs_init, (int, float)):
            # XXX: why should it be self.atoms[0] that is passed as an
            # arg here to create_random_structures, and not uniq_at?
            self.confs_init = create_random_structures(self.atoms[0],
                                                       self.std_init,
                                                       self.confs_init)

# ========================================================================== #
    def _load_training_confs(self, conf_fname):
        """Load and log training configurations."""

        self.confs_init = read(conf_fname, index=":")
        nb_confs_found = len(self.confs_init)

        self._write(f"{nb_confs_found} training configurations found")
        self._write("Adding them to the training data")

        if nb_confs_found >= 2:
            self._write("\tNo need to start new training computations\n")
        self._write()

# ========================================================================== #
    def _compute_training_confs(self):
        """
        Compute true potential properties of training configurations.
        """
        nstate = len(self.confs_init)
        subfolder_l = ["Training"] * nstate
        istep = np.arange(nstate, dtype=int)
        self._write("\nComputing true potential energy [training configs]")
        self.confs_init = self.calc.compute_true_potential(self.confs_init,
                                                           subfolder_l,
                                                           istep)
        if any(at is None for at in self.confs_init):
            msg = "True potential calculation failed or didn't converge"
            raise TruePotentialError(msg)
        self._write("Computation done")

# ========================================================================== #
    def _create_training_traj(self):
        """
        Create the training *.traj file.
        Write the initial configurations (stored in self.confs_init) in *.traj.
        """
        conf_fname = str(self.workdir / "Training_configurations.traj")
        init_traj = Trajectory(conf_fname, mode="w")
        for conf in self.confs_init:
            init_traj.write(conf)
        self._write()

# ========================================================================== #
    def _write(self, msg="", center=False, underline=False):
        self.log.write(msg, center, underline)


class TruePotentialError(Exception):
    """
    To be raised if there is a problem with the true potential
    """
    pass
