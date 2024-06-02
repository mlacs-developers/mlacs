"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import numpy as np

from ase.io import read, Trajectory
from ase.io.formats import UnknownFileTypeError
from ase.calculators.singlepoint import SinglePointCalculator

from .mlas import Mlas
from .core import Manager
from .state import StateManager
from .utilities.log import MlacsLog
from .utilities import create_random_structures
from .utilities.path_integral import compute_centroid_atoms


# ========================================================================== #
# ========================================================================== #
class PiMlacs(Mlas, Manager):
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
        Variance (in :math:`\mathring{a}^2`) of the displacement
        when creating initial configurations.
        Default :math:`0.05 \mathring{a}^2`

    keep_tmp_mlip: :class:`bool` (optional)
        Keep every generated MLIP. If True and using MBAR, a restart will
        recalculate every previous MLIP.weight using the old coefficients.
        Default ``False``.

    ntrymax: :class:`int` (optional)
        The maximum number of tentative to retry a step if
        the reference potential raises an error or didn't converge.
        Default ``0``.
    """
    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 prop=None,
                 neq=10,
                 nbeads=1,
                 confs_init=None,
                 std_init=0.05,
                 keep_tmp_mlip=True,
                 ntrymax=0,
                 workdir=''):

        Manager.__init__(self, workdir=workdir)

        # Initialize working directory
        self.workdir.mkdir(exist_ok=True, parents=True)

        ##############
        # Check inputs
        ##############
        self.keep_tmp_mlip = keep_tmp_mlip
        Mlas.__init__(self, workdir=workdir)
        self.traj_centroid = Trajectory(self.get_filepath(".traj"), mode="a")

        # Miscellanous initialization
        self.rng = np.random.default_rng()
        self.ntrymax = ntrymax

        # Check if trajectory files already exists
        self.launched = self._check_if_launched()

        self.log = MlacsLog(str(self.workdir / "MLACS.log"), self.launched)
        self.logger = self.log.logger_log
        msg = ""
        for i in range(self.nstate):
            msg += f"State {i+1}/{self.nstate} :\n"
            msg += repr(self.state[i])
        self.logger.info(msg)
        msg = self.calc.log_recap_state()
        self.logger.info(msg)
        self.logger.info(repr(self.mlip))

# ========================================================================== #
    def _initialize_momenta(self):
        """Create property object"""
        for i in range(self._nmax):
            self.state[0].initialize_momenta(self.atoms[i])
            prefix = self.state[i].prefix
            pot_fname = self.workdir / (prefix + "_potential.dat")
            with open(pot_fname, "w") as f:
                f.write("# True epot [eV]          MLIP epot [eV]\n")
        self.prefix = ''
        pot_fname = self.get_filepath("_potential.dat")
        with open(pot_fname, "w") as f:
            f.write("# True epot [eV]           True ekin [eV]   " +
                    "   MLIP epot [eV]            MLIP ekin [eV]\n")

# ========================================================================== #
    @property
    def _nmax(self):
        return self.nbeads

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
        self.logger.info("")
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
            self.logger.info(msg)
        self.logger.info("\n")

        # Training MLIP
        msg = "Training new MLIP\n"
        self.logger.info(msg)

        if self.keep_tmp_mlip:
            self.mlip.subfolder = f"Coef{max(self.nconfs)}"
        else:
            self.mlip.subfolder = ''

        # TODO GA: mlip object should be logging instead
        msg = self.mlip.train_mlip()
        self.logger.info(msg)

        # Create MLIP atoms object
        atoms_mlip = []
        for i in range(self._nmax):
            at = self.atoms[i].copy()
            at.set_masses(self.masses)
            atoms_mlip.append(at)

        # SinglePointCalculator to bypass the calc attach to atoms thing of ase
        sp_calc_mlip = []

        # Run the actual MLMD
        msg = "Running MLMD"
        self.logger.info(msg)

        # For PIMD, i-pi state manager is handling the parallel stuff
        atoms_mlip = self.state[istate].run_dynamics(
                           atoms_mlip[istate],
                           self.mlip.pair_style,
                           self.mlip.pair_coeff,
                           self.mlip.model_post,
                           self.mlip.atom_style,
                           eq[istate],
                           self.nbeads)

        # Computing energy with true potential
        msg = "Computing energy with the True potential\n"
        self.logger.info(msg)
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
                msg = "One of the true potential calculation failed, " + \
                      "restarting the step\n"
                self.logger.info(msg)
                return False

        # True potential error handling
        if nerror == self.nstate:
            msg = "All true potential calculations failed, " + \
                  "restarting the step\n"
            self.logger.info(msg)
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

        atoms_centroid = compute_centroid_atoms(atoms_true,
                                                self.temperature)
        atoms_centroid_mlip = compute_centroid_atoms(atoms_mlip,
                                                     self.temperature)
        self.traj_centroid.write(atoms_centroid)
        epot = atoms_centroid.get_potential_energy()
        ekin = atoms_centroid.get_kinetic_energy()
        epot_mlip = atoms_centroid_mlip.get_potential_energy()
        ekin_mlip = atoms_centroid_mlip.get_kinetic_energy()

        with open(self.prefix_centroid + "_potential.dat", "a") as f:
            f.write(f"{epot:20.15f}   " +
                    f"{ekin:20.15f}   " +
                    f"{epot_mlip:20.15f}   " +
                    f"{ekin_mlip:20.15f}\n")
        self.nconfs[0] += 1

        # Computing properties with ML potential.
        # Computing "on the fly" properties.
        if self.prop.manager is not None:
            self.prop.calc_initialize(atoms=self.atoms)
            self.prop.subdir = f"Step{self.step}"
            msg = self.prop.run(self.step)
            self.logger.info(msg)
            if self.prop.check_criterion:
                msg = "All property calculations are converged, " + \
                      "stopping MLACS ...\n"
                self.logger.info(msg)
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

        msg = "Running initial step"
        self.logger.info(msg)
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

        msg = f"There are {len(uniq_at)} unique configuration in the states "
        self.logger.info(msg)

        # And finally we compute the properties for each unique atoms
        nstate = len(uniq_at)
        subfolder_l = ["Initial"] * nstate
        istep = np.arange(nstate, dtype=int)
        uniq_at = self.calc.compute_true_potential(uniq_at, subfolder_l, istep)
        uniq_at = self.add_traj_descriptors(uniq_at)

        msg = "Computation done, creating trajectories"
        self.logger.info(msg)

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

        for istate in range(self.nstate):
            atoms = uniq_at[0]
            for ibead in range(self.nbeads):
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                stress = atoms.get_stress()
                calc = SinglePointCalculator(self.atoms[ibead],
                                             energy=energy,
                                             forces=forces,
                                             stress=stress)
                at = self.atoms[ibead].copy()
                at.set_masses(self.masses)
                at.calc = calc
                prefix = self.state[ibead].prefix
                self.traj.append(Trajectory(prefix + ".traj", mode="w"))
                self.traj[ibead].write(at)

            # Create centroid traj
            self.traj_centroid = Trajectory(self.prefix_centroid + ".traj",
                                            mode="w")
            self.traj_centroid.write(compute_centroid_atoms(
                                     [at] * self.nbeads,
                                     self.temperature))
            self.nconfs[istate] += 1

        # If there is no configurations in the database,
        # we need to create some and run the true potential
        if self.mlip.nconfs == 0:
            msg = "\nComputing energy with true potential " + \
                  "on training configurations"
            self.logger.info(msg)
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

            confs_init = self.add_traj_descriptors(confs_init)
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
                msg = "Training configurations found\n"
                msg += "Adding them to the training data"
                self.logger.info(msg)

                confs_init = read(conf_fname, index=":")
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
            self.logger.info("")
        else:
            msg = f"There are already {self.mlip.nconfs} configurations " + \
                  "in the database, no need to start training computations\n"
            self.logger.info(msg)
        # And now we add the starting configurations in the fit matrices
        for at in uniq_at:
            self.mlip.update_matrices(at)

        self.launched = True

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

        npimd = 0
        for s in self.state:
            if s.ispimd and self.nbeads > 1:
                npimd += 1
        if not npimd == 0 or not npimd == 1:
            msg = "PIMD simulation is available only for one state at a time"
            raise ValueError(msg)

        # We get the number of beads here
        assert self.nstate >= self.nbeads, (
            'nbeads should be smaller than number of states')

        # We need to store the masses for isotope purposes
        self.masses = atoms.get_masses()
        # We need the temperature for centroid computation purposes
        self.temperature = self.state[0].get_temperature()
        # Now we add the atoms
        for ibead in range(self.nbeads):
            at = atoms.copy()
            at.set_masses(self.masses)
            self.atoms.append(at)
        # Create list of neq
        self.neq = [neq]

        # Create prefix of output files
        self.prefix = prefix + "_centroid"

# ========================================================================== #
    def read_traj(self):
        """
        Read Trajectory files from previous simulations
        """
        msg = "Adding previous configurations to the training data"
        self.logger.info(msg)

        conf_fname = str(self.workdir / "Training_configurations.traj")
        if os.path.isfile(conf_fname):
            train_traj = Trajectory(conf_fname, mode="r")
            msg = "{0} training configurations\n".format(len(train_traj))
            self.logger.info(msg)
        else:
            train_traj = None

        prev_traj = []
        lgth = []
        for i in range(self._nmax):
            traj_fname = str(self.workdir / (self.state[i].prefix + ".traj"))
            prev_traj.append(Trajectory(traj_fname, mode="r"))
            lgth.append(len(prev_traj[i]))
        self.nconfs = [lgth[0]]
        if not np.all([a == lgth[0] for a in lgth]):
            msg = "Not all trajectories have the same number " + \
                  "of configurations"
            raise ValueError(msg)
        else:
            self.nconfs = lgth
        msg = f"{np.sum(lgth)} configuration from trajectories\n"
        self.logger.info(msg)
        return train_traj, prev_traj


class TruePotentialError(Exception):
    """
    To be raised if there is a problem with the true potential
    """
    pass
