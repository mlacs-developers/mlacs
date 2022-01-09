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

from mlacs.mlip import LammpsSnap
from mlacs.calc import CalcManager
from mlacs.state import StateManager
from mlacs.utilities.log import MlacsLog
from mlacs.utilities import create_random_structures


class OtfMlacs:
    """
    A Learn on-the-fly Molecular Dynamics constructed in order to sample an approximate distribution

    Parameters
    ----------

    atoms: :class:`ase.Atoms` or :list: of `ase.Atoms`
        the atom object on which the simulation is run. The atoms has to have a calculator attached
    state: :class:`StateManager` or :list: of :class: `StateManager`
        Object determining the state to be sampled
    calc: :class:`ase.calculators` or :class:`CalcManager`
        Class controlling the potential energy of the system to be approximated.
        If a :class:`ase.Calculators` is attached, the :class:`CalcManager` is 
        automatically created.
    mlip: :class:`MlipManager` (optional)
        Object managing the MLIP to approximate the real distribution
        Default is a :class:`LammpsSnap` object with a ``5.0`` angstrom rcut
        with ``8`` twojmax
    neq: :class:`int` (optional) or :class`list` of :class:`int`
        The number of equilibration iterations. Default ``10``.
    prefix_output: :class:`str` (optional) or list of :str:
        Prefix for the output files of the simulation. Default ``\"Trajectory\"``.
    confs_init: :class:`int` or :class:`list` of :class:`ase.Atoms`  (optional)
        if :class:`int`: Number of configuirations used to train a preliminary MLIP
        The configurations are created by rattling the first structure
        if :class:`list` of :class:`ase.Atoms`: The atoms that are to be computed in order to create the initial training configurations
        Default ``1``.
    std_init: :class:`float` (optional)
        Variance (in angs^2) of the displacement when creating initial configurations. Default ``0.05`` angs^2
    """
    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 neq=10,
                 prefix_output="Trajectory",
                 confs_init=None,
                 std_init=0.05,
                 ntrymax=0
                ):
        ##############
        # Check inputs
        ##############
        # Create list of states
        if isinstance(state, StateManager):
            self.state = [state]
        elif isinstance(state, list):
            self.state  = state
        else:
            msg = "state should be a StateManager object or a list of StateManager objects"
            raise TypeError(msg)
        # We get the number of simulated state here
        self.nstate = len(self.state)

        # Create list of atoms
        if isinstance(atoms, Atoms):
            self.atoms = []
            for istate in range(self.nstate):
                self.atoms.append(atoms.copy())
        elif isinstance(atoms, list):
            assert len(atoms) == self.nstate
            self.atoms = atoms
        else:
            msg = "atoms should be a ASE Atoms object or a list of ASE atoms objects"
            raise TypeError(msg)

        # Create calculator object 
        if isinstance(calc, Calculator):
            self.calc = CalcManager(calc)
        elif isinstance(calc, CalcManager):
            self.calc = calc
        else:
            msg = "calc should be a ase Calculator object or a CalcManager object"
            raise TypeError(msg)
    
        # Create mlip object
        if mlip is None:
            #self.mlip = LammpsMlip(self.atoms[0]) # Default MLIP Manager
            self.mlip = LammpsSnap(self.atoms[0]) # Default MLIP Manager
        else:
            self.mlip = mlip

        # Create list of neq -> number of equilibration mlmd runs for each state
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
                    self.prefix_output.append(prefix_output + "_{0}".format(i+1))
            else: 
                self.prefix_output.append(prefix_output)
        elif isinstance(prefix_output, list):
            assert len(prefix_output) == self.nstate
            self.prefix_output = prefix_output
        else:
            msg = "prefix_output should be a string or a list of strings"
            raise TypeError(msg)

        # Miscellanous initialization
        self.rng      = np.random.default_rng()
        self.ntrymax  = ntrymax

        #######################
        # Initialize everything
        #######################
        # Check if trajectory file already exists
        if os.path.isfile(self.prefix_output[0] + ".traj"):
            self.launched = True
        else:
            self.launched = False

        self.log = MlacsLog("MLACS.log", self.launched)
        msg = ""
        for istate in range(self.nstate):
            msg += "State {0}/{1} :\n".format(istate+1, self.nstate)
            msg += self.state[istate].log_recap_state()
        self.log.logger_log.info(msg)
        msg = self.calc.log_recap_state()
        self.log.logger_log.info(msg)
        self.log.recap_mlip(self.mlip.get_mlip_dict())
            
        # We initialize momenta and parameters for training configurations
        if not self.launched:
            for istate in range(self.nstate):
                self.state[istate].initialize_momenta(self.atoms[istate])
                with open(self.prefix_output[istate] + "_potential.dat", "w") as f:
                    f.write("# True epot [eV]          MLIP epot [eV]\n")
            self.confs_init  = confs_init
            self.std_init    = std_init
            self.nconfs = [0] * self.nstate
        
        # Reinitialize everything from the trajectories
        # Compute fitting data - get trajectories - get current configurations
        #if self.launched:
        else:
            msg = "Adding previous configurations to the training data"
            self.log.logger_log.info(msg)
            if os.path.isfile("Training_configurations.traj"):
                train_traj = Trajectory("Training_configurations.traj", mode="r")
                msg = "{0} training configurations".format(len(train_traj))
                self.log.logger_log.info(msg)
                for i, conf in enumerate(train_traj):
                    msg = "Configuration {:} / {:}".format(i+1, len(train_traj))
                    self.log.logger_log.info(msg)
                    self.mlip.update_matrices(conf)
                del train_traj
                self.log.logger_log.info("\n")

            prev_traj = []
            lgth      = []
            for istate in range(self.nstate):
                prev_traj.append(Trajectory(self.prefix_output[istate] + ".traj", mode="r"))
                lgth.append(len(prev_traj[istate]))
            msg = "{0} configuration from trajectories".format(np.sum(lgth))
            self.log.logger_log.info(msg)
            lgth = np.max(lgth)
            for iconf in range(lgth):
                for istate in range(self.nstate):
                    try:
                        self.mlip.update_matrices(prev_traj[istate][iconf])
                        msg = "Configuration {0} of state {1}/{2}".format(iconf, istate+1, self.nstate)
                        self.log.logger_log.info(msg)
                    except:
                        pass

            self.traj   = []
            self.atoms  = []
            self.nconfs = []
            for istate in range(self.nstate):
                self.traj.append(Trajectory(self.prefix_output[istate]+".traj", mode="a"))
                self.atoms.append(prev_traj[istate][-1])
                self.nconfs.append(len(prev_traj[istate]))
            del prev_traj
                
        self.step = 0
        self.ntrymax = ntrymax


#===================================================================================================================================================#
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


#===================================================================================================================================================#
    def _run_step(self):
        """
        Run one step of the algorithm

        One step consist in:
           fit of the MLIP
           nsteps of MLMD
           true potential computation
        """
        # Check if this is an equilibration or normal step for the mlmd
        
        self.log.logger_log.info("")
        eq = []
        for istate in range(self.nstate):
            if self.nconfs[istate] < self.neq[istate]:
                eq.append(True)
                msg   = "Equilibration step for state {0}".format(istate+1)
                self.log.logger_log.info(msg)
            else:
                eq.append(False)
                msg   = "Production step for state {0}".format(istate+1)
                self.log.logger_log.info(msg)
        self.log.logger_log.info("\n")

        # Training MLIP
        msg = "Training new MLIP\n"
        self.log.logger_log.info(msg)
        msg = self.mlip.train_mlip()
        self.log.logger_log.info(msg)

        # Create MLIP atoms object
        atoms_mlip = []
        for istate in range(self.nstate):
            atoms_mlip.append(self.atoms[istate].copy())

        # Run the actual MLMD
        msg = "Running MLMD"
        self.log.logger_log.info(msg)
        for istate in range(self.nstate):
            msg = "State {0}/{1}".format(istate+1, self.nstate)
            self.log.logger_log.info(msg)
            if self.state[istate].islammps:
                atoms_mlip[istate] = self.state[istate].run_dynamics(atoms_mlip[istate], self.mlip.pair_style, self.mlip.pair_coeff, eq[istate])
                atoms_mlip[istate].calc = self.mlip.calc
            else:
                atoms_mlip[istate] = self.state[istate].run_dynamics(atoms_mlip[istate], self.mlip.calc, eq[istate])
        self.log.logger_log.info("")
                
        # Computing energy with true potential
        msg  = "Computing energy with the True potential\n"
        self.log.logger_log.info(msg)
        atoms_true = []
        nerror = 0 # Handling of calculator error / non-convergence
        for istate in range(self.nstate):
            atoms_true.append(self.calc.compute_true_potential(atoms_mlip[istate].copy()))
            if atoms_true[istate] is not None:
                self.mlip.update_matrices(atoms_true[istate])
                self.traj[istate].write(atoms_true[istate])
                self.atoms[istate]   = atoms_true[istate]
                self.nconfs[istate] += 1
                with open(self.prefix_output[istate] + "_potential.dat", "a") as f:
                    f.write("{:20.15f}   {:20.15f}\n".format(atoms_true[istate].get_potential_energy(), atoms_mlip[istate].get_potential_energy()))
            if atoms_true[istate] is None:
                msg  = "For state {0}/{1} calculation with the true potential resulted in error or didn't converge".format(istate+1, self.nstate)
                self.log.logger_log.info(msg)
                nerror += 1

        if nerror == self.nstate:
            msg  = "All true potential calculations failed, restarting the step\n"
            self.log.logger_log.info(msg)
            return False
        else:
            return True


#===================================================================================================================================================#
    def _run_initial_step(self):
        """
        Run the initial step, where no MLIP or configurations are available

        consist in
            Compute potential energy for the initial positions
            Compute potential for nconfs_init training configurations
        """
        msg = "\nComputing energy with true potential on initial configuration"
        self.log.logger_log.info(msg)

        # Compute potential energy, update fitting matrices and write the configuration to the trajectory
        self.traj      = [] # To initialize the trajectories for each state
        computed_atoms = [] # To have a list of the already computed atoms in order to avoid making the same calculation twice
        for istate in range(self.nstate):
            if len(computed_atoms) == 0:
                msg  = "Initial configuration for state {0}/{1}".format(istate+1, self.nstate)
                self.log.logger_log.info(msg)
                atoms = self.calc.compute_true_potential(self.atoms[istate].copy())
                if atoms is None:
                    msg = "True potential calculation failed or didn't converge"
                    raise TruePotentialError(msg)
                computed_atoms.append(atoms)
                self.mlip.update_matrices(atoms)
            else: # This part is to avoid making the same calculation twice
                as_prev = False
                for at in computed_atoms:
                    if self.atoms[istate] == at:
                        msg  = "Initial configuration for state {0}/{1} is identical to a previously computed configuration".format(istate+1, self.nstate)
                        self.log.logger_log.info(msg)
                        calc       = SinglePointCalculator(self.atoms[istate], energy=at.get_potential_energy(), forces=at.get_forces(), stress=at.get_stress())
                        #calc       = SinglePointCalculator(self.atoms[istate], **at.calc.results)
                        atoms      = self.atoms[istate].copy()
                        atoms.calc = calc
                        as_prev = True
                if not as_prev:
                    msg  = "Initial configuration for state {0}/{1}".format(istate+1, self.nstate)
                    self.log.logger_log.info(msg)
                    atoms = self.calc.compute_true_potential(self.atoms[istate].copy())
                    if atoms is None:
                        msg = "True potential calculation failed or didn't converge"
                        raise TruePotentialError(msg)
                    computed_atoms.append(atoms)
                    self.mlip.update_matrices(atoms)
            self.traj.append(Trajectory(self.prefix_output[istate] + ".traj", mode="w"))
            self.traj[istate].write(atoms)

        msg = "\nComputing energy with true potential on training configurations"
        self.log.logger_log.info(msg)
        # Check number of training configurations and create them if needed
        if self.confs_init is None:
            confs_init = []
            for at in computed_atoms:
                confs_init.extend(create_random_structures(at, self.std_init, 1))
        elif isinstance(self.confs_init, (int, float)):
            confs_init = create_random_structures(self.atoms[0], self.std_init, self.confs_init)
        elif isinstance(self.confs_init, list):
            confs_init = self.confs_init
        nconfs_init = len(confs_init)

        if os.path.isfile("Training_configurations.traj"):
            msg  = "Training configurations found\n"
            msg += "Adding them to the training data"
            self.log.logger_log.info(msg)

            confs_init = read("Training_configurations.traj",index=":")
            for conf in confs_init:
                self.mlip.update_matrices(conf)
        else:
            init_traj = Trajectory("Training_configurations.traj", mode="w")
            for i, conf in enumerate(confs_init):
                msg = "Configuration {:} / {:}".format(i+1, nconfs_init)
                self.log.logger_log.info(msg)

                conf.rattle(0.1, rng=self.rng)
                conf = self.calc.compute_true_potential(conf)
                if conf is None:
                    msg = "True potential calculation failed or didn't converge"
                    raise TruePotentialError(msg)
             
                self.mlip.update_matrices(conf)
                init_traj.write(conf)
            # We dont need the initial configurations anymore
            del self.confs_init
        self.log.logger_log.info("")
        self.launched = True
