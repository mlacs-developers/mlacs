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
from .properties import PropertyManager
from .state import StateManager
from .utilities.log import MlacsLog
from .utilities import create_random_structures, save_cwd


# ========================================================================== #
# ========================================================================== #
class Mlas(Manager):
    """
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

    neq: :class:`int` (optional)
        The number of equilibration iteration. Default ``10``.

    nbeads: :class:`int` (optional)
        The number of beads to use from Path-Integral simulations.
        This value has to be lower than the number of beads used
        in the State object, or equal to it.
        If it is lower, this number indicates the number of beads
        for which a trajectory will be created and computed
        with the reference potential.
        Default ``1``, ignored for non-path integral States

    workdir: :class:`str` (optional)
        The directory in which to run the calculation.

    confs_init: :class:`int` or :class:`list` of :class:`ase.Atoms` (optional)
        If :class:`int`, Number of configurations used to train a preliminary
        MLIP. The configurations are created by rattling the first structure.
        If :class:`list` of :class:`ase.Atoms`, The atoms that are to be
        computed in order to create the initial training configurations.
        Default ``1``.

    std_init: :class:`float` (optional)
        Variance (in :math:`Ang^2`) of the displacement
        when creating initial configurations.
        Default :math:`0.05 Ang^2`

    keep_tmp_mlip: :class:`bool` (optional)
        Keep every generated MLIP. If True and using MBAR, a restart will
        recalculate every previous MLIP.weight using the old coefficients.
        Default ``False``.
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
                 workdir=''):

        Manager.__init__(self, workdir=workdir)

        # Initialize working directory
        self.workdir.mkdir(exist_ok=True, parents=True)

        ##############
        # Check inputs
        ##############
        self.keep_tmp_mlip = keep_tmp_mlip
        self._initialize_state(state, atoms, neq)
        self._initialize_calc(calc)
        self._initialize_mlip(mlip)
        self._initialize_property(prop)

        # Miscellanous initialization
        self.rng = np.random.default_rng()

        #######################
        # Initialize everything
        #######################

        # Check if trajectory files already exists
        self.launched = self._check_if_launched()

        self.log = MlacsLog(str(self.workdir / "MLACS.log"), self.launched)
        self.logger = self.log.logger_log
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

        # We initialize momenta and parameters for training configurations
        if not self.launched:
            self._initialize_momenta()
            self.confs_init = confs_init
            self.std_init = std_init
            self.nconfs = [0] * self.nstate
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
        Run one step of the algorithm

        One step consist in:
           fit of the MLIP
           nsteps of MLMD
           true potential computation
        """
        # Check if this is an equilibration or normal step for the mlmd
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
            msg += f"configurations {trajstep} for this state"
            self._write(msg)
        self._write()

        # Training MLIP
        self._write("Training new MLIP")

        if self.keep_tmp_mlip:
            self.mlip.subfolder = f"Coef{max(self.nconfs)}"
        else:
            self.mlip.subfolder = ''

        # TODO GA: mlip object should be logging instead
        msg = self.mlip.train_mlip()
        self._write(msg)

        # Create MLIP atoms object
        atoms_mlip = []
        for i in range(self._nmax):
            at = self.atoms[i].copy()
            atoms_mlip.append(at)

        # SinglePointCalculator to bypass the calc attach to atoms thing of ase
        sp_calc_mlip = []

        # Run the actual MLMD
        self._write("Running MLMD")

        for istate in range(self.nstate):
            if self.state[istate].isrestart or eq[istate]:
                self._write(" -> Starting from first atomic configuration")
                atoms_mlip[istate] = self.atoms_start[istate].copy()
                self.state[istate].initialize_momenta(atoms_mlip[istate])

        # With those thread, we can execute all the states in parallell
        futures = []
        with save_cwd(), ThreadPoolExecutor() as executor:
            for istate in range(self.nstate):
                exe = executor.submit(self.state[istate].run_dynamics,
                                      *(atoms_mlip[istate],
                                        self.mlip.pair_style,
                                        self.mlip.pair_coeff,
                                        self.mlip.model_post,
                                        self.mlip.atom_style,
                                        eq[istate]))
                futures.append(exe)
                self._write(f"State {istate+1}/{self.nstate} has been launched")  # noqa
            for istate, exe in enumerate(futures):
                atoms_mlip[istate] = exe.result()
                if self.keep_tmp_mlip:
                    mm = self.mlip.descriptor.subsubdir
                    atoms_mlip[istate].info['parent_mlip'] = str(mm)
            executor.shutdown(wait=True)

        # Computing energy with true potential
        self._write("Computing energy with the True potential")
        atoms_true = []
        nerror = 0  # Handling of calculator error / non-convergence

        # TODO GA: Might be better to do the threading at this level,
        #          up from calc.compute_true_potential.
        subfolder_l = [s.subfolder for s in self.state]
        step_l = [self.step] * self.nstate
        atoms_true = self.calc.compute_true_potential(atoms_mlip,
                                                      subfolder_l,
                                                      step=step_l)

        for i, at in enumerate(atoms_mlip):
            at.calc = self.mlip.get_calculator()
            sp_calc_mlip.append(SinglePointCalculator(
                                at,
                                energy=at.get_potential_energy(),
                                forces=at.get_forces(),
                                stress=at.get_stress()))
            at.calc = sp_calc_mlip[i]

        for i, at in enumerate(atoms_true):
            if at is None:
                msg = f"For state {i+1}/{self._nmax} calculation with " + \
                       "the true potential resulted in error " + \
                       "or didn't converge"
                self._write(msg)
                nerror += 1

        # True potential error handling
        if nerror == self.nstate:
            msg = "All true potential calculations failed, " + \
                  "restarting the step"
            self._write(msg)
            return False

        # And now we can write the configurations in the trajectory files
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

        self._compute_properties()
        self._execute_post_step()

        return True

# ========================================================================== #
    def _run_initial_step(self):
        """
        Run the initial step, where no MLIP or configurations are available

        consist in
            Compute potential energy for the initial positions
            Compute potential for nconfs_init training configurations
        """
        # Compute potential energy, update fitting matrices
        # and write the configuration to the trajectory
        self.traj = []  # To initialize the trajectories for each state

        self._write("Running initial step")
        # Once each computation is done, we need to correctly assign each atom
        # to the right state, this is done using the idx_computed list of list
        uniq_at = []
        idx_computed = []
        for istate in range(self.nstate):
            if len(uniq_at) == 0:  # We always have to add the first atoms
                uniq_at.append(self.atoms[istate])
                idx_computed.append([istate])
            else:
                isin_list = False
                for icop, at in enumerate(uniq_at):
                    if self.atoms[istate] == at:
                        isin_list = True
                        idx_computed[icop].append(istate)
                if not isin_list:
                    uniq_at.append(self.atoms[istate])
                    idx_computed.append([istate])

        self._write("There are {len(uniq_at)} unique configuration in the states ")  # noqa

        # And finally we compute the properties for each unique atoms
        nstate = len(uniq_at)
        subfolder_l = ["Initial"] * nstate
        istep = np.arange(nstate, dtype=int)
        uniq_at = self.calc.compute_true_potential(uniq_at, subfolder_l, istep)

        self._write("Computation done, creating trajectories")

        # And now, we dispatch each atoms to the right trajectory
        for iun, at in enumerate(uniq_at):
            if at is None:
                msg = "True potential calculation failed or " + \
                      "didn't converge"
                raise TruePotentialError(msg)
            for icop in idx_computed[iun]:
                newat = at.copy()
                epot = at.get_potential_energy()
                forces = at.get_forces()
                stress = at.get_stress()
                calc = SinglePointCalculator(newat,
                                             energy=epot,
                                             forces=forces,
                                             stress=stress)
                newat.calc = calc
                self.atoms[icop] = newat

        self.atoms = self.add_traj_descriptors(self.atoms)
        for istate in range(self.nstate):
            prefix = self.state[istate].prefix
            self.traj.append(Trajectory(prefix + ".traj", mode="w"))
            self.traj[istate].write(self.atoms[istate])
            self.nconfs[istate] += 1

        # If there is no configurations in the database,
        # we need to create some and run the true potential
        if self.mlip.nconfs == 0:
            msg = "\nComputing energy with true potential " + \
                  "on training configurations"
            self._write(msg)
            # Check number of training configurations and create them if needed
            if self.confs_init is None:
                confs_init = create_random_structures(uniq_at,
                                                      self.std_init,
                                                      1)
            elif isinstance(self.confs_init, (int, float)):
                confs_init = create_random_structures(self.atoms[0],
                                                      self.std_init,
                                                      self.confs_init)
            elif isinstance(self.confs_init, list):
                confs_init = self.confs_init

            conf_fname = str(self.workdir / "Training_configurations.traj")
            checkisfile = False
            if os.path.isfile(conf_fname):
                try:
                    read(conf_fname)
                    checkisfile = True
                except UnknownFileTypeError:
                    checkisfile = False
            else:
                checkisfile = False

            if checkisfile:
                self.log._delimiter()
                self._write("Training configurations found")
                self._write("Adding them to the training data")

                confs_init = read(conf_fname, index=":")
                confs_init = self.add_traj_descriptors(confs_init)
                for conf in confs_init:
                    self.mlip.update_matrices(conf)
            else:

                # Distribute state training
                nstate = len(confs_init)
                subfolder_l = ["Training"] * nstate
                istep = np.arange(nstate, dtype=int)
                confs_init = self.calc.compute_true_potential(
                    confs_init,
                    subfolder_l,
                    istep)
                confs_init = self.add_traj_descriptors(confs_init)
                init_traj = Trajectory(conf_fname, mode="w")
                for i, conf in enumerate(confs_init):
                    if conf is None:
                        msg = "True potential calculation failed or " + \
                              "didn't converge"
                        raise TruePotentialError(msg)
                    self.mlip.update_matrices(conf)
                    init_traj.write(conf)
                # We dont need the initial configurations anymore
                del self.confs_init
            self._write()
        else:
            msg = f"There are already {self.mlip.nconfs} configurations " + \
                  "in the database, no need to start training computations\n"
            self._write(msg)
        # And now we add the starting configurations in the fit matrices
        for at in uniq_at:
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
    def _initialize_property(self, prop):
        """Create property object"""
        self.prop = PropertyManager(prop)

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
            for istate in range(self.nstate):
                self.atoms.append(atoms.copy())
        elif isinstance(atoms, list):
            assert len(atoms) == self.nstate
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
    def _check_if_launched(self):
        """
        Function to check simulation restarts:
         - Check if trajectory files exist and are not empty.
         - Check if the number of configuration found is at least two.
        """
        _nat_init = 0
        for i in range(self._nmax):
            traj_fname = str(self.workdir / (self.state[i].prefix + ".traj"))
            if os.path.isfile(traj_fname):
                try:
                    _nat_init += len(read(traj_fname, index=':'))
                except UnknownFileTypeError:
                    return False
            else:
                return False
        traj_fname = str(self.workdir / "Training_configurations.traj")
        if os.path.isfile(traj_fname):
            try:
                _nat_init += len(read(traj_fname, index=':'))
            except UnknownFileTypeError:
                return False
        if 1 < _nat_init:
            return True
        else:
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

        # for i in range(self._nmax):
        #     self.state[i].subsubdir.mkdir(exist_ok=True, parents=True)

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
    def _write(self, msg="", center=False, underline=False):
        self.log.write(msg, center, underline)


class TruePotentialError(Exception):
    """
    To be raised if there is a problem with the true potential
    """
    pass
