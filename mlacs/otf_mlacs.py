"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read, Trajectory
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from mlacs.mlip import LammpsMlip
from mlacs.calc import CalcManager
from mlacs.state import StateManager
from mlacs.utilities.log import MlacsLog
from mlacs.utilities import create_random_structures
from mlacs.utilities.path_integral import compute_centroid_atoms


# ========================================================================== #
# ========================================================================== #
class OtfMlacs:
    """
    A Learn on-the-fly simulation constructed in order to sample approximate
    distribution

    Parameters
    ----------

    atoms: :class:`ase.Atoms` or :list: of `ase.Atoms`
        the atom object on which the simulation is run. The atoms has to have
        a calculator attached
    state: :class:`StateManager` or :list: of :class: `StateManager`
        Object determining the state to be sampled
    calc: :class:`ase.calculators` or :class:`CalcManager`
        Class controlling the potential energy of the system
        to be approximated.
        If a :class:`ase.calculators` is attached, the :class:`CalcManager`
        is automatically created.
    mlip: :class:`MlipManager` (optional)
        Object managing the MLIP to approximate the real distribution
        Default is a LammpsMlip object with a snap descriptor,
        5.0 angstrom rcut with 8 twojmax.
    neq: :class:`int` (optional)
        The number of equilibration iteration. Default ``10``.
    prefix_output: :class:`str` (optional)
        Prefix for the output files of the simulation.
        Default ``\"Trajectory\"``.
    confs_init: :class:`int` or :class:`list` of :class:`ase.Atoms`  (optional)
        if :class:`int`: Number of configuirations used
        to train a preliminary MLIP
        The configurations are created by rattling the first structure
        if :class:`list` of :class:`ase.Atoms`: The atoms that are to be
        computed in order to create the initial training configurations
        Default ``1``.
    std_init: :class:`float` (optional)
        Variance (in angs^2) of the displacement when creating
        initial configurations. Default ``0.05`` angs^2
    """
    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 neq=10,
                 nbeads=1,
                 prefix_output="Trajectory",
                 confs_init=None,
                 std_init=0.05,
                 ntrymax=0):
        ##############
        # Check inputs
        ##############
        self._initialize_state(state, atoms, neq, prefix_output, nbeads)

        # Create calculator object
        if isinstance(calc, Calculator):
            self.calc = CalcManager(calc)
        elif isinstance(calc, CalcManager):
            self.calc = calc
        else:
            msg = "calc should be a ase Calculator object or " + \
                  "a CalcManager object"
            raise TypeError(msg)

        # Create mlip object
        if mlip is None:
            self.mlip = LammpsMlip(self.atoms[0])  # Default MLIP Manager
        else:
            self.mlip = mlip

        # Miscellanous initialization
        self.rng = np.random.default_rng()
        self.ntrymax = ntrymax

        #######################
        # Initialize everything
        #######################
        # Check if trajectory file already exists
        if os.path.isfile(self.prefix_output[0] + ".traj"):
            self.launched = True
        else:
            self.launched = False

        if self.pimd:
            nmax = self.nbeads
            val = "bead"
        else:
            nmax = self.nstate
            val = "state"

        self.log = MlacsLog("MLACS.log", self.launched)
        msg = ""
        for i in range(self.nstate):
            msg += "State {0}/{1} :\n".format(i+1, self.nstate)
            msg += self.state[i].log_recap_state()
        self.log.logger_log.info(msg)
        msg = self.calc.log_recap_state()
        self.log.logger_log.info(msg)
        self.log.recap_mlip(self.mlip.get_mlip_dict())

        # We initialize momenta and parameters for training configurations
        if not self.launched:
            for i in range(nmax):
                if self.pimd:
                    self.state[0].initialize_momenta(self.atoms[i])
                else:
                    self.state[i].initialize_momenta(self.atoms[i])
                with open(self.prefix_output[i] + "_potential.dat", "w") as f:
                    f.write("# True epot [eV]          MLIP epot [eV]\n")
            if self.pimd:
                with open(self.prefix_centroid + "_potential.dat", "w") as f:
                    f.write("# True epot [eV]           True ekin [eV]   " +
                            "   MLIP epot [eV]            MLIP ekin [eV]\n")
            self.confs_init = confs_init
            self.std_init = std_init
            self.nconfs = [0] * self.nstate

        # Reinitialize everything from the trajectories
        # Compute fitting data - get trajectories - get current configurations
        else:
            msg = "Adding previous configurations to the training data"
            self.log.logger_log.info(msg)
            if os.path.isfile("Training_configurations.traj"):
                train_traj = Trajectory("Training_configurations.traj",
                                        mode="r")
                msg = "{0} training configurations".format(len(train_traj))
                self.log.logger_log.info(msg)
                for i, conf in enumerate(train_traj):
                    msg = f"Configuration {i+1} / {len(train_traj)}"
                    self.log.logger_log.info(msg)
                    self.mlip.update_matrices(conf)
                del train_traj
                self.log.logger_log.info("\n")

            prev_traj = []
            lgth = []
            for i in range(nmax):
                prev_traj.append(Trajectory(self.prefix_output[i] + ".traj",
                                            mode="r"))
                lgth.append(len(prev_traj[i]))
            if self.pimd:
                self.nconfs = [lgth[0]]
                if not np.all([a == lgth[0] for a in lgth]):
                    msg = "Not all trajectories have the same number " + \
                          "of configurations"
                    raise ValueError(msg)
            else:
                self.nconfs = lgth
            msg = f"{np.sum(lgth)} configuration from trajectories"
            self.log.logger_log.info(msg)
            lgth = np.max(lgth)
            for iconf in range(lgth):
                for i in range(nmax):
                    self.mlip.update_matrices(prev_traj[i][iconf])
                    msg = f"Configuration {iconf} of {val} " + \
                          f"{i+1}/{nmax}"
                    self.log.logger_log.info(msg)

            self.traj = []
            self.atoms = []
            for i in range(nmax):
                self.traj.append(Trajectory(self.prefix_output[i]+".traj",
                                            mode="a"))
                self.atoms.append(prev_traj[i][-1])
            if self.pimd:
                self.traj_centroid = Trajectory(self.prefix_centroid + ".traj",
                                                mode="a")
            del prev_traj

        self.step = 0
        self.ntrymax = ntrymax

# ========================================================================== #
    def run(self, nsteps=100):
        """
        Run the algorithm for nsteps
        """
        # Initialize ntry in case of failing computation
        self.ntry = 0
        while self.step < nsteps:
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
        if self.pimd:
            nmax = self.nbeads
            val = "Beads"
        else:
            nmax = self.nstate
            val = "State"

        self.log.logger_log.info("")
        eq = []
        for istate in range(self.nstate):
            if self.nconfs[istate] < self.neq[istate]:
                eq.append(True)
                msg = "Equilibration step for state {0}".format(istate+1)
                self.log.logger_log.info(msg)
            else:
                eq.append(False)
                msg = "Production step for state {0}".format(istate+1)
                self.log.logger_log.info(msg)
        self.log.logger_log.info("\n")

        # Training MLIP
        msg = "Training new MLIP\n"
        self.log.logger_log.info(msg)
        msg = self.mlip.train_mlip()
        self.log.logger_log.info(msg)

        # Create MLIP atoms object
        atoms_mlip = []
        for i in range(nmax):
            at = self.atoms[i].copy()
            if self.pimd:
                at.set_masses(self.masses)
            atoms_mlip.append(at)

        # SinglePointCalculator to bypass the calc attach to atoms thing of ase
        sp_calc_mlip = []

        # Run the actual MLMD
        msg = "Running MLMD"
        self.log.logger_log.info(msg)
        for istate in range(self.nstate):
            msg = "State {0}/{1}".format(istate+1, self.nstate)
            self.log.logger_log.info(msg)
            if self.pimd:
                atoms_mlip = self.state[istate].run_dynamics(
                                   atoms_mlip[istate],
                                   self.mlip.pair_style,
                                   self.mlip.pair_coeff,
                                   self.mlip.model_post,
                                   self.mlip.atom_style,
                                   self.mlip.bonds,
                                   self.mlip.angles,
                                   self.mlip.bond_style,
                                   self.mlip.bond_coeff,
                                   self.mlip.angle_style,
                                   self.mlip.angle_coeff,
                                   eq[istate],
                                   self.nbeads)
            else:
                atoms_mlip[istate] = self.state[istate].run_dynamics(
                                           atoms_mlip[istate],
                                           self.mlip.pair_style,
                                           self.mlip.pair_coeff,
                                           self.mlip.model_post,
                                           self.mlip.atom_style,
                                           self.mlip.bonds,
                                           self.mlip.angles,
                                           self.mlip.bond_style,
                                           self.mlip.bond_coeff,
                                           self.mlip.angle_style,
                                           self.mlip.angle_coeff,
                                           eq[istate])
        for i, at in enumerate(atoms_mlip):
            at.calc = self.mlip.calc
            sp_calc_mlip.append(SinglePointCalculator(
                                at,
                                energy=at.get_potential_energy(),
                                forces=at.get_forces(),
                                stress=at.get_stress()))
            at.calc = sp_calc_mlip[i]

        # Computing energy with true potential
        msg = "Computing energy with the True potential\n"
        self.log.logger_log.info(msg)
        atoms_true = []
        nerror = 0  # Handling of calculator error / non-convergence
        for i in range(nmax):
            msg = "{0} {1}/{2}".format(val, i+1, nmax)
            self.log.logger_log.info(msg)
            at = self.calc.compute_true_potential(atoms_mlip[i].copy())
            atoms_true.append(at)
            if self.pimd:
                if atoms_true[i] is None:
                    msg = "One of the true potential calculation failed, " + \
                          "restarting the step\n"
                    self.log.logger_log.info(msg)
                    return False
                else:
                    atoms_true[i].set_masses(self.masses)
            else:
                if atoms_true[i] is None:
                    msg = f"For state {i+1}/{nmax} calculation with " + \
                           "the true potential resulted in error " + \
                           "or didn't converge"
                    self.log.logger_log.info(msg)
                    nerror += 1

        # True potential error handling
        if nerror == self.nstate:
            msg = "All true potential calculations failed, " + \
                  "restarting the step\n"
            self.log.logger_log.info(msg)
            return False

        # We need to write atoms after computation,
        # in case the simulation stops before all beads are computed
        # or one of the true calc computation fails
        for i, at in enumerate(atoms_true):
            if at is not None:
                self.mlip.update_matrices(at)
                self.traj[i].write(at)
                self.atoms[i] = at
                with open(self.prefix_output[i] + "_potential.dat", "a") as f:
                    f.write("{:20.15f}   {:20.15f}\n".format(
                             at.get_potential_energy(),
                             at.get_potential_energy()))
                if not self.pimd:
                    self.nconfs[i] += 1
        if self.pimd:
            atoms_centroid = compute_centroid_atoms(atoms_true,
                                                    self.temperature)
            atoms_centroid_mlip = compute_centroid_atoms(atoms_mlip,
                                                         self.temperature)
            self.traj_centroid.write(atoms_centroid)
            epot = atoms_centroid.get_potential_energy()
            ekin = atoms_centroid.get_kinetic_energy()
            epot_mlip = atoms_centroid_mlip.get_potential_energy()
            ekin_mlip = atoms_centroid_mlip.get_potential_energy()
            with open(self.prefix_centroid + "_potential.dat", "a") as f:
                f.write(f"{epot:20.15f}   " +
                        f"{ekin:20.15f}   " +
                        f"{epot_mlip:20.15f}   " +
                        f"{ekin_mlip:20.15f}\n")
            self.nconfs[0] += 1
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
        computed_atoms = []  # To have a list of the already computed atoms
        for istate in range(self.nstate):
            if len(computed_atoms) == 0:
                msg = "Initial configuration for state " + \
                      f"{istate+1}/{self.nstate}"
                self.log.logger_log.info(msg)
                atoms = self.calc.compute_true_potential(
                                  self.atoms[istate].copy())
                if atoms is None:
                    msg = "True potential calculation failed or " + \
                          "didn't converge"
                    raise TruePotentialError(msg)
                computed_atoms.append(atoms)
                self.mlip.update_matrices(atoms)
            else:  # This part is to avoid making the same calculation twice
                as_prev = False
                for at in computed_atoms:
                    if self.atoms[istate] == at:
                        msg = "Initial configuration for state " + \
                              f"{istate+1}/{self.nstate} is identical " + \
                              "to a previously computed configuration"
                        self.log.logger_log.info(msg)
                        epot = at.get_potential_energy()
                        forces = at.get_forces()
                        stress = at.get_stress()
                        calc = SinglePointCalculator(self.atoms[istate],
                                                     energy=epot,
                                                     forces=forces,
                                                     stress=stress)
                        atoms = self.atoms[istate].copy()
                        atoms.calc = calc
                        as_prev = True
                if not as_prev:
                    msg = "Initial configuration for state + " + \
                           f"{istate+1}/{self.nstate}"
                    self.log.logger_log.info(msg)
                    atoms = self.calc.compute_true_potential(
                                      self.atoms[istate].copy())
                    if atoms is None:
                        msg = "True potential calculation " + \
                              "failed or didn't converge"
                        raise TruePotentialError(msg)
                    computed_atoms.append(atoms)
                    self.mlip.update_matrices(atoms)
            if self.pimd:
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
                    self.traj.append(Trajectory(self.prefix_output[ibead] +
                                                ".traj", mode="w"))
                    self.traj[ibead].write(at)

                # Create centroid traj
                self.traj_centroid = Trajectory(self.prefix_centroid +
                                                ".traj", mode="w")
                self.traj_centroid.write(compute_centroid_atoms(
                                         [at] * self.nbeads,
                                         self.temperature))
            else:
                self.traj.append(Trajectory(self.prefix_output[istate] +
                                            ".traj", mode="w"))
                self.traj[istate].write(atoms)

        # Training configurations
        msg = "\nComputing energy with true potential " + \
              "on training configurations"
        self.log.logger_log.info(msg)
        # Check number of training configurations and create them if needed
        if self.confs_init is None:
            confs_init = create_random_structures(computed_atoms,
                                                  self.std_init,
                                                  1)
        elif isinstance(self.confs_init, (int, float)):
            confs_init = create_random_structures(self.atoms[0],
                                                  self.std_init,
                                                  self.confs_init)
        elif isinstance(self.confs_init, list):
            confs_init = self.confs_init
        nconfs_init = len(confs_init)

        if os.path.isfile("Training_configurations.traj"):
            msg = "Training configurations found\n"
            msg += "Adding them to the training data"
            self.log.logger_log.info(msg)

            confs_init = read("Training_configurations.traj", index=":")
            for conf in confs_init:
                self.mlip.update_matrices(conf)
        else:
            init_traj = Trajectory("Training_configurations.traj", mode="w")
            for i, conf in enumerate(confs_init):
                msg = "Configuration {:} / {:}".format(i+1, nconfs_init)
                self.log.logger_log.info(msg)

                conf = self.calc.compute_true_potential(conf)
                if conf is None:
                    msg = "True potential calculation failed or " + \
                          "didn't converge"
                    raise TruePotentialError(msg)

                self.mlip.update_matrices(conf)
                init_traj.write(conf)
            # We dont need the initial configurations anymore
            del self.confs_init
        self.log.logger_log.info("")
        self.launched = True

# ========================================================================== #
    def _initialize_state(self, state, atoms, neq, prefix_output, nbeads):
        """
        Function to initialize the state
        """
        # Put the state(s) as a list
        if isinstance(state, StateManager):
            self.state = [state]
        if isinstance(state, list):
            self.state = state

        print(nbeads)
        npimd = 0
        for s in self.state:
            if s.ispimd and nbeads > 1:
                npimd += 1
        if npimd == 0:
            self.pimd = False
        elif npimd == 1:
            self.pimd = True
        else:
            msg = "PIMD simulation is available only for one state at a time"
            raise ValueError(msg)

        self.atoms = []
        self.nstate = len(self.state)
        if self.pimd:
            # We get the number of beads here
            self.nbeads = nbeads  # self.state[0].get_nbeads()
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
            self.prefix_output = []
            for i in range(self.nbeads):
                self.prefix_output.append(prefix_output + "_{0}".format(i+1))
            self.prefix_centroid = prefix_output + "_centroid"

        else:
            # Create list of atoms
            if isinstance(atoms, Atoms):
                for istate in range(self.nstate):
                    self.atoms.append(atoms.copy())
            elif isinstance(atoms, list):
                assert len(atoms) == self.nstate
                self.atoms = atoms
            else:
                msg = "atoms should be a ASE Atoms object or " + \
                      "a list of ASE atoms objects"
                raise TypeError(msg)

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

            # Create prefix of output files
            if isinstance(prefix_output, str):
                self.prefix_output = []
                if self.nstate > 1:
                    for i in range(self.nstate):
                        self.prefix_output.append(prefix_output +
                                                  f"_{i+1}")
                else:
                    self.prefix_output.append(prefix_output)
            elif isinstance(prefix_output, list):
                assert len(prefix_output) == self.nstate
                self.prefix_output = prefix_output
            else:
                msg = "prefix_output should be a string or a list of strings"
                raise TypeError(msg)


class TruePotentialError(Exception):
    """
    To be raised if there is a problem with the true potential
    """
    pass
