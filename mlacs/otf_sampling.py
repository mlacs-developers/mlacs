"""
Class for On-The-Fly Machine-Learning Assisted Sampling
"""
import os

import numpy as np
from ase.io import read as ase_read, Trajectory

from mlacs.mlip import LammpsMlip
from mlacs.utilities.log import MLACS_Log
from mlacs.utilities import create_random_structures


class OtfMLACS:
    """
    On-the-fly Machine-Learning Assisted Sampling

    A Learn on-the-fly Molecular Dynamics constructed in order to sample an approximate distribution

    Parameters:
    -----------
    atoms: ase atoms object
         the atom object on which the simulation is run. The atoms has to have a calculator attached
    state: StateManager object
    calc: ase calculator
    mlip: MLIPManager object
    neq: int
        The number of step equilibration steps
    confs_init: int or list of atoms
        if int: Number of configuirations used to train a preliminary MLIP
                The configurations are created by rattling the first structure
        else: The atoms that are to be computed in order to create the initial training configurations
    prefix_output: str
        Prefix for the output files of the simulation
    """
    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 neq=10,
                 prefix_output="Trajectory",
                 confs_init=None,
                 std_init=0.05
                ):
        
        self.atoms     = atoms
        self.true_calc = calc
        if mlip is None:
            self.mlip = LammpsMlip(atoms) # Default MLIP Manager
        else:
            self.mlip = mlip
        self.neq       = neq

        self.state = state

        self.prefix_output = prefix_output
        self.rng           = np.random.default_rng()

        # Initialize everything
        if os.path.isfile(self.prefix_output + ".traj"):
            # Previous simulation found, need to reinitialize everything in memory
            restart = True
            self.log = MLACS_Log(self.prefix_output + ".log", restart)
            self.log.recap_mlip(self.mlip.get_mlip_dict())
            msg = self.state.log_recap_state()
            self.log.logger_log.info(msg)

            # Get trajectory and create the matrices for the fitting
            self.traj = Trajectory(self.prefix_output + ".traj", mode="a")
            if os.path.isfile("Training_configurations.traj"):
                train_traj = Trajectory("Training_configurations.traj", mode="r")
                for conf in train_traj:
                    self.mlip.update_matrices(conf)
                del train_traj
            prev_traj = Trajectory(self.prefix_output + ".traj", mode="r")
            for conf in prev_traj:
                self.mlip.update_matrices(conf)

            # Update current atoms and step
            self.atoms = prev_traj[-1]
            self.step  = len(prev_traj)
            del prev_traj
            
            # Update the potential file to compare predicted and true potential
            if os.path.isfile(self.prefix_output + "_potential.dat"):
                potentials    = np.loadtxt(self.prefix_output + "_potential.dat")
                self.vtrue    = np.atleast_2d(potentials)[:,1]
                self.vmlip    = np.atleast_2d(potentials)[:,2]
        else:
            # No previous step
            restart = False
            self.log = MLACS_Log(self.prefix_output + ".log", restart)
            self.log.recap_mlip(self.mlip.get_mlip_dict())
            msg = self.state.log_recap_state()
            self.log.logger_log.info(msg)

            # Everything at 0
            self.step = 0
            self.traj = Trajectory(self.prefix_output + ".traj", mode="w")
            self.vtrue       = np.array([])
            self.vmlip       = np.array([])
            self.state.initialize_momenta(self.atoms)


        # If initial step, initialize everything in order to train an initial MLIP
        if self.step == 0:
            self.confs_init = confs_init
            self.nconfs_init = 1
            self.std_init = std_init

        # If first launch of the simulation, print the starting atoms in the log
        if not restart:
            self.log.write_input_atoms(self.atoms)
        

#===================================================================================================================================================#
    def run(self, nsteps=100):
        """
        Run the algorithm until nsteps
        """
        while self.step < nsteps:
            self.log.init_new_step(self.step)
            if self.step == 0:
                self.run_initial_step()
                self.step += 1
            else:
                self.run_step()
                self.step += 1


#===================================================================================================================================================#
    def run_step(self):
        """
        Run one step of the algorithm

        One step consist in:
           fit of the MLIP
           nsteps of MLMD
           true potential computation
        """
        # Check if this is an equilibration or normal step
        if self.step < self.neq:
            eq   = True
            msg  = "Equilibration step\n"
            msg += "\n"
            self.log.logger_log.info(msg)
        else:
            eq = False

        msg = "Training new MLIP potential\n"
        self.log.logger_log.info(msg)
        msg = self.mlip.train_mlip()
        self.log.logger_log.info(msg)

        # Copy atoms to have a MLIP one
        atoms_mlip = self.atoms.copy()

        # Ensure the momenta is right before the MLMD
        momenta = self.atoms.get_momenta()
        atoms_mlip.set_momenta(momenta)

        # Run the actual MLMD
        if self.state.islammps:
            atoms_mlip = self.state.run_dynamics(atoms_mlip, self.mlip.pair_style, self.mlip.pair_coeff, eq)
            atoms_mlip.calc = self.mlip.calc
        else:
            atoms_mlip = self.state.run_dynamics(atoms_mlip, self.mlip.calc, eq)

        # Clean the MLIP to liberate procs
        #self.mlip.calc.clean()

        # Prepare atoms object to compute the energy with the true potential
        atoms_true      = atoms_mlip.copy() # copy to avoid disasters
        atoms_true.calc = self.true_calc

        msg  = "Computing energy with the True potential\n"
        msg += "\n"
        self.log.logger_log.info(msg)
        # Compute the potential energy for the true potential
        Vn_true = atoms_true.get_potential_energy()
        # Compute the potential energy for the MLIP
        Vn_mlip = atoms_mlip.get_potential_energy()

        # Update the matrices for the MLIP fit
        self.mlip.update_matrices(atoms_true)

        # Update the true and mlip potential energy arrays
        self.vtrue = np.append(self.vtrue, Vn_true)
        self.vmlip = np.append(self.vmlip, Vn_mlip)

        # Add new configuration to trajectory and save potential arrays to have a measure of accuracy
        self.traj.write(atoms_true)
        idx           = np.arange(1, len(self.vmlip)+1)
        all_potential = np.vstack((idx, self.vtrue, self.vmlip)).T
        np.savetxt(self.prefix_output + "_potential.dat", all_potential, fmt="%d " + 2 * " %15.20f", header="Step - Vtrue - Vmlip")

        # Update atoms
        self.atoms = atoms_true


#===================================================================================================================================================#
    def run_initial_step(self):
        """
        Run the initial step, where no MLIP or configurations are available

        consist in
            Compute potential energy for the initial positions
            Compute potential for nconfs_init training configurations
        """
        msg = "Computing energy with true potential on initial configuration"
        self.log.logger_log.info(msg)

        # Compute potential energy, update fitting matrices and write the configuration to the trajectory
        self.atoms.calc = self.true_calc
        v_init     = self.atoms.get_potential_energy()
        self.mlip.update_matrices(self.atoms)
        self.traj.write(self.atoms)

        msg = "Computing energy with true potential on initial training configurations"
        self.log.logger_log.info(msg)

        # Check number of training configurations and create them if needed
        if self.confs_init is None:
            confs_init = create_random_structures(self.atoms, self.std_init, 1)
        elif isinstance(self.confs_init, (int, float)):
            confs_init = create_random_structures(self.atoms, self.std_init, self.confs_init)
        elif isinstance(self.confs_init, list):
            confs_init = self.confs_init
        nconfs_init = len(confs_init)

        init_traj = Trajectory("Training_configurations.traj", mode="w")
        for i, conf in enumerate(confs_init):
            msg = "Configuration {:} / {:}".format(i+1, nconfs_init)
            self.log.logger_log.info(msg)

            conf.calc = self.true_calc
            conf.rattle(0.1, rng=self.rng)
            conf.calc = self.atoms.calc
            conf.get_potential_energy()
         
            self.mlip.update_matrices(conf)
            init_traj.write(conf)
        # We dont need the initial configurations anymore
        del self.confs_init
