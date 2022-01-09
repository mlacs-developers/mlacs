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
from mlacs.path_integral import compute_centroid_atoms


class PiMlacs:
    """
    A Learn on-the-fly Path-Integral Molecular Dynamics constructed in order to sample an approximate distribution

    Parameters
    ----------

    atoms: :class:`ase.Atoms` or :list: of `ase.Atoms`
        the atom object on which the simulation is run. The atoms has to have a calculator attached
    state: :class:`StateManager` or :list: of :class: `StateManager`
        Object determining the state to be sampled
    calc: :class:`ase.calculators` or :class:`CalcManager`
        Class controlling the potential energy of the system to be approximated.
        If a :class:`ase.calculators` is attached, the :class:`CalcManager` is 
        automatically created.
    mlip: :class:`MlipManager` (optional)
        Object managing the MLIP to approximate the real distribution
        Default is a LammpsSnap object with a 5.0 angstrom rcut
        with 8 twojmax.
    neq: :class:`int` (optional)
        The number of equilibration iteration. Default ``10``.
    prefix_output: :class:`str` (optional)
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
        self.state  = state
        # We get the number of beads here
        self.nbeads = self.state.get_nbeads()
        # We need to store the masses for isotope purposes
        self.masses = atoms.get_masses()
        # We need the temperature for centroid computation purposes
        self.temperature = self.state.get_temperature()

        # Create list of atoms
        self.atoms = []
        for ibead in range(self.nbeads):
            at = atoms.copy()
            at.set_masses(self.masses)
            self.atoms.append(at)

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
            self.mlip = LammpsSnap(self.atoms[0]) # Default MLIP Manager
        else:
            self.mlip = mlip

        # Get number of equilibration steps
        self.neq = neq

        # Create prefix of output files
        self.prefix_output = []
        for i in range(self.nbeads):
            self.prefix_output.append(prefix_output + "_{0}".format(i+1))
        self.prefix_centroid = prefix_output + "_centroid"

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

        self.log = MlacsLog("PIMLACS.log", self.launched)
        msg = self.state.log_recap_state()
        self.log.logger_log.info(msg)
        msg = self.calc.log_recap_state()
        self.log.logger_log.info(msg)
        self.log.recap_mlip(self.mlip.get_mlip_dict())
            
        # We initialize momenta and parameters for training configurations
        if not self.launched:
            for ibead in range(self.nbeads):
                self.state.initialize_momenta(self.atoms[ibead])
                with open(self.prefix_output[ibead] + "_potential.dat", "w") as f:
                    f.write("# True epot [eV]          MLIP epot [eV]\n")
                with open(self.prefix_centroid + "_potential.dat", "w") as f:
                    f.write("# True epot [eV]           True ekin [eV]       MLIP epot [eV]            MLIP ekin [eV]\n")
            self.confs_init  = confs_init
            self.std_init    = std_init
            self.nconfs = 0
        
        # Reinitialize everything from the trajectories
        # Compute fitting data - get trajectories - get current configurations
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
            for ibead in range(self.nbeads):
                prev_traj.append(Trajectory(self.prefix_output[ibead] + ".traj", mode="r"))
                lgth.append(len(prev_traj[ibead]))
            if not np.all([a == lgth[0] for a in lgth]):
                msg = "Not all trajectories have the same number of configurations"
                raise ValueError(msg)
            else:
                self.nconfs = lgth[0]
            msg = "{0} configuration from trajectories".format(np.sum(lgth))
            self.log.logger_log.info(msg)
            lgth = np.max(lgth)
            for iconf in range(lgth):
                for ibead in range(self.nbeads):
                    try:
                        self.mlip.update_matrices(prev_traj[ibead][iconf])
                        msg = "Configuration {0} of bead {1}/{2}".format(iconf, ibead+1, self.nbead)
                        self.log.logger_log.info(msg)
                    except:
                        pass

            self.traj   = []
            self.atoms  = []
            for ibead in range(self.nbeads):
                self.traj.append(Trajectory(self.prefix_output[ibead]+".traj", mode="a"))
                self.atoms.append(prev_traj[ibead][-1])
            self.traj_centroid = Trajectory(self.prefix_centroid + ".traj", mode="a")
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
        if self.nconfs < self.neq:
            eq    = True
            msg   = "Equilibration step"
        else:
            eq    = False
            msg   = "Production step"
        self.log.logger_log.info(msg)
        self.log.logger_log.info("\n")

        # Training MLIP
        msg = "Training new MLIP\n"
        self.log.logger_log.info(msg)
        msg = self.mlip.train_mlip()
        self.log.logger_log.info(msg)

        # Create MLIP atoms object
        atoms_mlip = []
        for ibead in range(self.nbeads):
            at = self.atoms[ibead].copy()
            at.set_masses(self.masses)
            atoms_mlip.append(at)


        # SinglePointCalculator to bypass the calc attach to atoms thing of ase
        sp_calc_true = []
        sp_calc_mlip = []

        # Run the actual MLMD
        msg = "Running MLMD"
        self.log.logger_log.info(msg)
        atoms_mlip = self.state.run_dynamics(atoms_mlip, self.mlip.pair_style, self.mlip.pair_coeff, eq)
        for ibead, at in enumerate(atoms_mlip):
            at.calc = self.mlip.calc
            sp_calc_mlip.append(SinglePointCalculator(at, energy=at.get_potential_energy(), forces=at.get_forces(), stress=at.get_stress()))
            at.calc = sp_calc_mlip[ibead]
                
        # Computing energy with true potential
        msg  = "Computing energy with the True potential\n"
        self.log.logger_log.info(msg)
        atoms_true = []
        for ibead in range(self.nbeads):
            msg = "Bead {0}/{1}".format(ibead+1, self.nbeads)
            self.log.logger_log.info(msg)
            at = self.calc.compute_true_potential(atoms_mlip[ibead].copy())
            sp_calc_true.append(SinglePointCalculator(at, energy=at.get_potential_energy(), forces=at.get_forces(), stress=at.get_stress()))
            at.calc = sp_calc_true[ibead]
            atoms_true.append(at)
            #atoms_true.append(self.calc.compute_true_potential(atoms_mlip[ibead].copy()))
            if atoms_true[ibead] is None:
                msg  = "One of the true potential calculation failed, restarting the step\n"
                self.log.logger_log.info(msg)
                return False
            else:
                atoms_true[ibead].set_masses(self.masses)

        # We need to write atoms after computation, in case the simulation stops before all beads are computed
        # or one of the true calc computation fails
        for ibead, at in enumerate(atoms_true):
            self.mlip.update_matrices(at)
            self.traj[ibead].write(at)
            self.atoms[ibead] = at
            with open(self.prefix_output[ibead] + "_potential.dat", "a") as f:
                f.write("{:20.15f}   {:20.15f}\n".format(at.get_potential_energy(), at.get_potential_energy()))
        atoms_centroid      = compute_centroid_atoms(atoms_true, self.temperature)
        atoms_centroid_mlip = compute_centroid_atoms(atoms_mlip, self.temperature)
        self.traj_centroid.write(atoms_centroid)
        with open(self.prefix_centroid + "_potential.dat", "a") as f:
            f.write("{:20.15f}   {:20.15f}   {:20.15f}   {:20.15f}\n".format(atoms_centroid.get_potential_energy(), 
                                                                             atoms_centroid.get_kinetic_energy(),
                                                                             atoms_centroid_mlip.get_potential_energy(),
                                                                             atoms_centroid_mlip.get_kinetic_energy()))
        self.nconfs += 1
        return True


#===================================================================================================================================================#
    def _run_initial_step(self):
        """
        Run the initial step, where no MLIP or configurations are available

        consist in
            Compute potential energy for the initial positions
            Compute potential for nconfs_init training configurations
        """
        # Compute potential energy, update fitting matrices and write the configuration to the trajectory
        msg = "Computing energy with the true potential on the initial configuration"
        self.log.logger_log.info(msg)
        at = self.calc.compute_true_potential(self.atoms[0].copy())
        if at is None:
            msg = "True potential calculation failed or didn't converge"
            raise TruePotentialError(msg)
        else:
            at.set_masses(self.masses)
        self.mlip.update_matrices(at)

        # Create trajectory for each bead
        self.traj = []
        for ibead in range(self.nbeads):
            calc = SinglePointCalculator(self.atoms[ibead], energy=at.get_potential_energy(), forces=at.get_forces(), stress=at.get_stress())
            atoms = self.atoms[ibead].copy()
            atoms.set_masses(self.masses)
            atoms.calc = calc
            self.traj.append(Trajectory(self.prefix_output[ibead] + ".traj", mode="w"))
            self.traj[ibead].write(atoms)
        # Create centroid traj
        self.traj_centroid = Trajectory(self.prefix_centroid + ".traj", mode="w")
        self.traj_centroid.write(compute_centroid_atoms([at] * self.nbeads, self.temperature))

        msg = "\nComputing energy with true potential on training configurations"
        self.log.logger_log.info(msg)
        # Check number of training configurations and create them if needed
        if self.confs_init is None:
            confs_init = create_random_structures(at, self.std_init, 1)
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
