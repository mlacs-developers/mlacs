import os
import logging
import datetime
from mlacs.version import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s')

#===================================================================================================================================================#
#===================================================================================================================================================#
class MLACS_Log:
    '''
    Logging class
    '''
    def __init__(self, logfile, restart=False):
        if not restart:
            if os.path.isfile(logfile):
                prev_step = 1
                while os.path.isfile(logfile + '{:04d}'.format(prev_step)):
                    prev_step += 1
                os.rename(logfile, logfile + "{:04d}".format(prev_step))

        self.logger_log = logging.getLogger('output')
        self.logger_log.addHandler(logging.FileHandler(logfile, 'a'))

        if not restart:
            self.write_header()
        else:
            self.write_restart()


#===================================================================================================================================================#
    def write_header(self):
        msg  = '===============================================================\n' 
        msg += '    On-the-fly Machine-Learning Assisted Canonical Sampling\n'
        #msg += '===============================================================\n' 
        msg += '======================= version  ' + str(__version__) + ' ========================\n'
        now = datetime.datetime.now()
        msg += 'date: ' + now.strftime('%d-%m-%Y  %H:%M:%S')
        msg += '\n'
        msg += '\n'
        self.logger_log.info(msg)


#===================================================================================================================================================#
    def write_restart(self):
        msg  = '\n'
        msg += '\n'
        msg += '===============================================================\n' 
        msg += '                 Restarting simulation\n'
        msg += '===============================================================\n' 
        now = datetime.datetime.now()
        msg += 'date: ' + now.strftime('%d-%m-%Y  %H:%M:%S')
        msg += '\n'
        msg += '\n'
        self.logger_log.info(msg)


#===================================================================================================================================================#
    def recap_params(self, temperature, nsteps, nt, nt_eq, neq, nconfs_init, mlip_params, friction, timestep):
        msg  = '===============================================================\n' 
        msg += "Recap of the simulation parameters\n"
        msg += "----------------------------------\n"
        msg += "\n"
        msg += "General parameters:\n"
        msg += "Temperature:                                 {:} K\n".format(temperature)
        msg += "Total number of steps:                       {:}\n".format(nsteps)
        msg += "Number of equilibration steps:               {:}\n".format(neq)
        msg += "Number of MLMD steps during equilibration    {:}\n".format(nt_eq)
        msg += "Number of MLMD steps after equilibration     {:}\n".format(nt)
        msg += "Number of initial configurations:            {:}\n".format(nconfs_init)
        msg += "\n"
        msg += "Machine-Learning Molecular Dynamics parameters:\n"
        msg += "Langevin thermostat\n"
        msg += "Timestep:                                    {:} fs\n".format(timestep)
        msg += "Friction parameters:                         {:}\n".format(friction)
        msg += "\n"
        msg += "Machine-Learning Interatomic Potential parameters:\n"
        msg += "The model used is {:}\n".format(mlip_params["model"])
        msg += "Cutoff radius:                               {:}\n".format(mlip_params["rcutfac"])
        msg += "Descriptor\n"
        if mlip_params["style"] == "snap":
            msg += "Spectral Neighbor Analysis Potential\n"
            msg += "2J_max:                                      {:}\n".format(mlip_params["twojmax"])
            if mlip_params["chemflag"] == 1:
                msg += "Multi-element version\n"
        elif mlip_params["style"] == "so3":
            msg += "Smooth SO(3) Power Spectrum\n"
            msg += "nmax                                         {:}\n".format(mlip_params["nmax"])
            msg += "lmax                                         {:}\n".format(mlip_params["lmax"])
            msg += "alpha                                        {:}\n".format(mlip_params["alpha"])
        msg += "Total number of coefficient to fit           {:}\n".format(mlip_params["ncoef"])
        msg += "\n"
        msg += "\n"
        msg += '===============================================================\n' 
        self.logger_log.info(msg)
        

#===================================================================================================================================================#
    def recap_mlip(self, mlip_params):
        msg  = "Machine-Learning Interatomic Potential parameters\n"
        msg += "The model used is {:}\n".format(mlip_params["model"])
        msg += "Cutoff radius:                               {:}\n".format(mlip_params["rcut"])
        msg += "Descriptor\n"
        if mlip_params["style"] == "snap":
            msg += "Spectral Neighbor Analysis Potential\n"
            msg += "2Jmax:                                      {:}\n".format(mlip_params["twojmax"])
            if mlip_params["chemflag"] == 1:
                msg += "Multi-element version\n"
        elif mlip_params["style"] == "so3":
            msg += "Smooth SO(3) Power Spectrum\n"
            msg += "nmax                                         {:}\n".format(mlip_params["nmax"])
            msg += "lmax                                         {:}\n".format(mlip_params["lmax"])
            msg += "alpha                                        {:}\n".format(mlip_params["alpha"])
        msg += "Energy coefficient  {:}\n".format(mlip_params["energy_coefficient"])
        msg += "Forces coefficient  {:}\n".format(mlip_params["forces_coefficient"])
        msg += "Stress coefficient  {:}\n".format(mlip_params["stress_coefficient"])
        msg += "\n"
        self.logger_log.info(msg)
        

#===================================================================================================================================================#
    def write_input_atoms(self, atoms):
        """
        """
        pos  = atoms.get_scaled_positions(wrap=False)
        cell = atoms.get_cell()

        msg  = "Initial configuration:\n"
        msg += "----------------------\n"
        msg += "Number of atoms:         {:}\n".format(len(atoms))
        msg += "Elements:\n"
        i = 0
        for symb in atoms.get_chemical_symbols():
            msg += symb + '  '
            i += 1
            if i == 10:
                i = 0
                msg += '\n'
        msg += "\n"
        msg += "\n"
        msg += "Supercell vectors in angstrom:\n"
        for alpha in range(3):
            msg += '{:18.16f}  {:18.16f}  {:18.16f}\n'.format(cell[alpha,0], cell[alpha,1], cell[alpha,2])
        msg += '\n'
        msg += 'Reduced positions:\n'
        for iat in range(len(atoms)):
            msg += '{:16.16f}  {:20.16f}  {:20.16f}\n'.format(pos[iat,0], pos[iat,1], pos[iat,2])
        msg += '\n'
        msg += '\n'
        msg += '===============================================================\n' 
        self.logger_log.info(msg)
        

#===================================================================================================================================================#
    def init_new_step(self, step):
        msg  = '===============================================================\n' 
        msg += 'Step {:}'.format(step)
        self.logger_log.info(msg)
