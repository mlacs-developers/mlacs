"""
Class for On-The-Fly Machine-Learning Assisted Sampling
"""
import os
import shutil

import numpy as np
from ase.io import read as ase_read, write as ase_write, Trajectory

from otf_mlacs.mlip.mlip_manager import MLIPManager
from otf_mlacs.utilities.log import MLACS_Log
from otf_mlacs.utilities import create_random_structures


class OtfMLAS:
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
    nt: int
        Number of steps for the NVT Molecular Dynamics steps with the MLIP
    nt_eq: int
        Number of steps for the NVT Molecular Dynamics steps with the MLIP during equilibration
    nconfs_init: int
        Number of configuirations used to train a preliminary MLIP
        The configurations are created by rattling the first structure
    prefix_output: str
        Prefix of the output of the simulation
    """
    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 neq=10,
                 prefix_output="Trajectory",
                 confs_init=None,
                 std_init=0.1
                ):
        
        self.atoms     = atoms
        self.true_calc = calc
        if mlip is None:
            self.mlip = MLIPManager(atoms)
        else:
            self.mlip = mlip
        self.neq       = neq

        self.state = state

        self.prefix_output = prefix_output
        self.rng           = np.random.default_rng()

        if os.path.isfile(self.prefix_output + ".traj"):
            # Previous simulation found, need to reinitialize everything in memory
            restart = True
            self.traj = Trajectory(self.prefix_output + ".traj", mode="a")
            if os.path.isfile("Training_configurations.traj"):
                train_traj = Trajectory("Training_configurations.traj", mode="r")
                for conf in train_traj:
                    self.mlip.update_matrices(conf)
                del train_traj
            prev_traj = Trajectory(self.prefix_output + ".traj", mode="r")
            for conf in prev_traj:
                self.mlip.update_matrices(conf)
            self.atoms = prev_traj[-1]
            self.step  = len(prev_traj)
            del prev_traj
            
            if os.path.isfile("potential.dat"):
                potentials    = np.loadtxt("potential.dat")
                self.vtrue    = np.atleast_2d(potentials)[:,1]
                self.vmlip    = np.atleast_2d(potentials)[:,2]
        else:
            # No previous step
            restart = False
            self.step = 0
            self.traj = Trajectory(self.prefix_output + ".traj", mode="w")
            self.vtrue       = np.array([])
            self.vmlip       = np.array([])
            self.state.initialize_momenta(self.atoms)


        if self.step == 0:
            self.confs_init = confs_init
            self.nconfs_init = 1
            self.std_init = std_init

        # Start the log
        self.log = MLACS_Log(self.prefix_output + ".log", restart)
        self.log.recap_mlip(self.mlip.get_mlip_dict())
        #self.log.recap_params()
        if not restart:
            self.log.write_input_atoms(self.atoms)
        

#===================================================================================================================================================#
    def run(self, nsteps=100):
        """
        Run the algorithm
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
           nsteps of MLMD for each state
           true potential computation for each state
        """
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

        atoms_mlip = self.atoms.copy()

        momenta = self.atoms.get_momenta()
        atoms_mlip.set_momenta(momenta)

        # Run the actual MLMD
        atoms_mlip = self.state.run_dynamics(atoms_mlip, self.mlip.calc, eq)

        # Clean the MLIP to liberate procs
        self.mlip.calc.clean()

        # Prepare atoms object to compute the energy with the true potential
        atoms_true      = atoms_mlip.copy()
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

        self.vtrue = np.append(self.vtrue, Vn_true)
        self.vmlip = np.append(self.vmlip, Vn_mlip)

        # Add new configuration to trajectory and update potential to have a measure of accuracy
        self.traj.write(atoms_true)
        idx           = np.arange(1, len(self.vmlip)+1)
        all_potential = np.vstack((idx, self.vtrue, self.vmlip)).T
        np.savetxt("potential.dat", all_potential, fmt="%d " + 2 * " %15.20f", header="Step - Vtrue - Vmlip")

        # Update atoms
        self.atoms = atoms_true


#===================================================================================================================================================#
    def run_initial_step(self):
        """
        """
        msg = "Computing energy with true potential on initial configuration"
        self.log.logger_log.info(msg)

        self.atoms.calc = self.true_calc
        v_init     = self.atoms.get_potential_energy()
        self.mlip.update_matrices(self.atoms)

        self.traj.write(self.atoms)

        msg = "Computing energy with true potential on initial training configurations"
        self.log.logger_log.info(msg)

        if self.confs_init is None:
            confs_init = create_random_structures(supercell, self.std_init, 1)
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
