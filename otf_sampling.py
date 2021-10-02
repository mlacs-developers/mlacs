"""
Class for On-The-Fly Machine-Learning Assisted Sampling
"""
import os
import shutil

import numpy as np
from ase.io import read as ase_read, write as ase_write, Trajectory

from otf_mlacs.mlip.mlip_manager import MLIPManager
from otf_mlacs.utilities.log import MLACS_Log


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
                 nt=1000,
                 nt_eq=250,
                 prefix_output="Trajectory",
                 confs_init=None,
                ):
        
        self.atoms     = atoms
        self.true_calc = calc
        if mlip is None:
            self.mlip = MLIPManager(atoms)
        else:
            self.mlip = mlip
        self.nt        = nt
        self.nt_eq     = nt_eq
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
                self.vtrue    = potentials[:,1]
                self.vmlip    = potentials[:,2]
        else:
            # No previous step
            restart = False
            self.step = 0
            self.traj = Trajectory(self.prefix_output + ".traj", mode="w")
            self.vtrue       = np.array([])
            self.vmlip       = np.array([])


        if self.step == 0:
            self.confs_init = confs_init
            self.nconfs_init = 1

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
            nsteps = self.nt_eq
            msg  = "Equilibration step\n"
            msg += "\n"
            self.log.logger_log.info(msg)
        else:
            nsteps = self.nt

        msg = "Training new MLIP potential\n"
        self.log.logger_log.info(msg)
        msg = self.mlip.train_mlip()
        self.log.logger_log.info(msg)

        atoms_mlip = self.atoms.copy()

        if self.step == 1:
            #momenta = self.state.initialize_momenta()
            momenta = np.zeros((len(atoms_mlip),3))
        else:
            momenta = self.atoms.get_momenta()
        atoms_mlip.set_momenta(momenta)

        # Run the actual MLMD
        atoms_mlip = self.state.run_dynamics(atoms_mlip, self.mlip.calc, nsteps)

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

        v_init     = self.atoms.get_potential_energy()
        self.mlip.update_matrices(self.atoms)

        self.traj.write(self.atoms)

        msg = "Computing energy with true potential on initial training configurations"
        self.log.logger_log.info(msg)

        init_traj = Trajectory("Training_configurations.traj", mode="w")
        for iconf in range(self.nconfs_init):
            msg = "Configuration {:} / {:}".format(iconf+1, self.nconfs_init)
            self.log.logger_log.info(msg)

            atoms_rattled = self.atoms.copy()
            atoms_rattled.rattle(0.1, rng=self.rng)
            atoms_rattled.calc = self.atoms.calc
            atoms_rattled.get_potential_energy()
         
            self.mlip.update_matrices(atoms_rattled)
            init_traj.write(atoms_rattled)
