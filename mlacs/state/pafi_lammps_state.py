import os
from subprocess import call, run

import numpy as np

from ase import Atoms
from ase.units import fs, kB
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory, read, write
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsdata import write_lammps_data
from ase.calculators.singlepoint import SinglePointCalculator as SPC

from mlacs.state import LammpsState
from mlacs.utilities import get_elements_Z_and_masses, write_lammps_NEB_ASCIIfile

#========================================================================================================================#
#========================================================================================================================#
class PafiLammpsState(LammpsState):
    """
    """
    def __init__(self,
                 temperature,
                 configurations,
                 reaction_coordinate=0.5,
                 Kspring=1.0,
                 maxjump=0.4,
                 dt=1.5,
                 damp=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 brownian=True,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 NEBworkdir=None,
                 prt=True,
                 workdir=None
                ):

        LammpsState.__init__(self,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval,
                             rng,
                             init_momenta,
                             workdir
                            )
                     

        self.isrestart     = False
        self.isappend      = False
        self.temperature   = temperature
        self.nsteps        = nsteps
        self.NEBcoord      = reaction_coordinate
        self.print         = prt
        self.Kspring       = Kspring
        self.maxjump       = maxjump
        self.damp          = damp
        self.brownian      = brownian
        self.NEBworkdir    = NEBworkdir
        self.confNEB       = configurations
        if self.NEBworkdir is None:
            self.NEBworkdir = os.getcwd() + "/LammpsMLNEB/"
        if self.NEBworkdir[-1] != "/":
            self.NEBworkdir[-1] += "/"
        if not os.path.exists(self.NEBworkdir):
            os.makedirs(self.NEBworkdir)
        if len(self.confNEB) != 2:
            raise TypeError('First and last configurations are not defined')
        if self.print:
            self.mfepstep = 0
        self.lammpsNEBfname = self.NEBworkdir + "lammps_input.in" 
        self._get_lammps_command_replica()
        self._Finit = 0

#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps, rep=''):
        """
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)

        if self.damp is None:
            damp = "$(10*dt)"

        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units        metal\n"

        pbc = atoms.get_pbc()
        input_string += "atom_style   atomic\n"
        input_string += "atom_modify  map array sort 0 0.0\n"
        input_string += "neigh_modify every 2 delay 10 check yes page 1000000 one 100000\n\n"
        input_string += "fix 1 all property/atom d_nx d_ny d_nz d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz\n"
        input_string += "read_data {0} fix 1 NULL PafiPath\n\n".format(self.atomsfname+rep)
        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"
        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"
        input_string += "# Compute relevant field for PAFI simulation\n"
        input_string += "timestep  {0}\n".format(self.dt/ 1000)
        input_string += "thermo    1\n"
        input_string += "min_style fire\n"
        input_string += "compute   1 all property/atom d_nx d_ny d_nz d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz\n"
        input_string += "run 0\n"
        input_string += "\n"

        input_string += "# Set up PAFI Langevin/Brownian integration in hyperplane\n"
        if self.brownian:
            input_string += "fix       pafihp all pafi 1 {0} {1} {2} overdamped {3} com yes\n".format(self.temperature, damp, self.rng.integers(99999), "yes")
        else:
            input_string += "fix       pafihp all pafi 1 {0} {1} {2} overdamped {3} com yes\n".format(self.temperature, damp, self.rng.integers(99999), "no")

        input_string += "\n"
        input_string += "run 0\n"
        input_string += "\n"
        input_string += "minimize 0 0 250 250\n"
        input_string += "reset_timestep  0\n\n"


        if self.logfile is not None:
            input_string += self.get_log_in()
        if self.trajfile is not None:
            input_string += self.get_traj_in(elem)
        input_string += self._log_pafi(rep)


        input_string += "# Dump last step\n"
        input_string += "dump last all custom {0} ".format(nsteps) + self.workdir + "configurations.out  id type xu yu zu vx vy vz fx fy fz element\n"
        input_string += "dump_modify last element "
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "dump_modify last delay {0}\n".format(nsteps)
        input_string += "\n"

        input_string += "run  {0}".format(nsteps)


        with open(self.lammpsfname+rep, "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def write_lammps_input_NEB(self, atoms, pair_style, pair_coeff):
        """
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)
        
        input_string  = "# LAMMPS input file to run a MLNEB simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "atom_modify  map array sort 0 0.0\n"
        input_string += "neigh_modify every 2 delay 10 check yes page 1000000 one 100000\n\n"
        input_string += "read_data atoms-0.data\n\n"
        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"
        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"
        input_string += "# Compute relevant field for NEB simulation\n"
        input_string += "timestep    {0}\n".format(self.dt/ (fs * 1000))
        input_string += "thermo      1\n"
        input_string += "fix         neb all neb {0} parallel ideal\n".format(self.Kspring)
        input_string += "run 100\n"
        input_string += "reset_timestep  0\n\n"
        input_string += "variable    i equal part\n"
        input_string += "min_style   quickmin\n"
        input_string += "neb         0.0 0.001 200 100 10 final atoms-1.data\n"
        input_string += "write_data  neb.$i\n"
        
        with open(self.lammpsNEBfname, "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def run_NEB(self, pair_style, pair_coeff):
        """
        Run a NEB calculation with lammps. Use replicas.
        """
        Z = self.confNEB[0].get_atomic_numbers()
        write_lammps_data(self.NEBworkdir+'atoms-0.data', self.confNEB[0])
        write_lammps_NEB_ASCIIfile(self.NEBworkdir+'atoms-1.data', self.confNEB[1])

        self.write_lammps_input_NEB(self.confNEB[0], pair_style, pair_coeff)
        lammps_command = self.cmdreplica + "-in " + self.lammpsNEBfname + "> log"
        call(lammps_command, shell=True, cwd=self.NEBworkdir)


#========================================================================================================================#
    def run_MFEP(self, pair_style, pair_coeff, restart=0, fstop=0.001, xi=None, nsteps=10000, interval=10, nthrow=2000, mpi=None):
        """
        Run a MFEP calculation with lammps. Use replicas.
        """
        if xi is None:
            xi = np.arange(0, 1.1, 0.1)
        if not hasattr(self, 'true_atoms'):
            self.run_NEB(pair_style, pair_coeff)
            self.extract_NEB_configurations()
        self.compute_spline(xi)
        nrep = len(self.spline_atoms)
        lmp = ''
        cnt = 0
        for rep in range(restart, nrep): 
            if mpi is None:
                lmp  = self._set_PAFI_MD_cmd('.{}'.format(rep), pair_style, pair_coeff,
                                             self.spline_atoms[rep],
                                             self.spline_coordinates[rep,:,:], 
                                             nsteps, nthrow)
                call(lmp, shell=True, cwd=self.workdir)
            else:
                lmp += self._set_PAFI_MD_cmd('.{}'.format(rep), pair_style, pair_coeff,
                                             self.spline_atoms[rep],
                                             self.spline_coordinates[rep,:,:], 
                                             nsteps, nthrow, mpi)
                cnt += 1
                if cnt < mpi and rep != nrep-1:
                    continue
                lmp += 'wait'
                cnt = 0
                #print(lmp)
                run(lmp, shell=True, cwd=self.workdir)
            lmp = ''
        self.pafi = []
        for rep in range(len(self.spline_atoms)): 
            self.pafi.append(np.loadtxt(self.workdir+'pafi.log.{}'.format(rep)).T[:,nthrow:].tolist())
        self.pafi = np.array(self.pafi)
        F = self.log_free_energy(xi)
        return F


#========================================================================================================================#
    def extract_NEB_configurations(self):
        """
        Step 1
        Extract the positions and energies of a NEB calculation for the N replicas.
        """
        true_atoms       = []
        true_coordinates = []
        Z = self.confNEB[0].get_atomic_numbers()
        N = len(self.confNEB[0])
        for rep in range(int(self.nreplica)):
            positions, cell = self._read_lammpsdata_truepositions(self.NEBworkdir + 'neb.' + str(rep))
            true_coordinates.append(positions)
            check = False
            with open(self.NEBworkdir + 'log.lammps.' + str(rep)) as r: 
                for l in r:
                    if check:
                        etotal = l.split()[2]
                        break
                    if 'Energy initial, next-to-last, final =' in l: check = True
            atoms = self._create_ASE_object(Z, positions, cell, etotal)
            true_atoms.append(atoms)
        self.true_atoms       = true_atoms
        self.path_coordinates = np.arange(self.nreplica)/(self.nreplica-1)
        self.true_coordinates = np.array(true_coordinates)
        self.true_energies    = np.array([float(true_atoms[i].get_potential_energy()) for i in range(self.nreplica)])
        if self.print: write(self.NEBworkdir + 'pos_neb_path_{}.xyz'.format(self.mfepstep), true_atoms, format='extxyz')


#========================================================================================================================#
    def compute_spline(self, xi=None):
        """
        Step 2
        Compute a 1D CubicSpline interpolation from a NEB calculation. 
        The function also set up the lammps data file for a constrained MD.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at reaction coordinate xi.
        """
        Z = self.confNEB[0].get_atomic_numbers()
        N = len(self.confNEB[0])

        from mlacs.utilities import interpolate_points as IP

        if xi is None:
            xi = self.NEBcoord

        self.spline_energies    = IP(self.path_coordinates, self.true_energies, xi, 0, border=1)
        spline_coordinates = []

        # Spline interpolation of the referent path and calculation of the path tangent and path tangent derivate
        for i in range(N):
            coord =      [IP(self.path_coordinates, self.true_coordinates[:,i,j], xi, 0) for j in range(3)] 
            coord.extend([IP(self.path_coordinates, self.true_coordinates[:,i,j], xi, 1) for j in range(3)]) 
            coord.extend([IP(self.path_coordinates, self.true_coordinates[:,i,j], xi, 2) for j in range(3)]) 
            spline_coordinates.append(coord)        
        spline_coordinates = np.array(spline_coordinates)

        self.spline_atoms = []
        self.spline_coordinates = []
        if isinstance(xi, float):
            self.spline_coordinates.append(np.array(self._COM_corrections(spline_coordinates.tolist())).round(8))
            self.spline_coordinates = np.array(self.spline_coordinates)
            self.spline_atoms.append(self._create_ASE_object(Z, 
                                     np.hsplit(self.spline_coordinates[0], 5)[0], 
                                     self.confNEB[0].get_cell(),
                                     self.spline_energies))
        else: 
            for rep in range(len(xi)):
                self.spline_coordinates.append(self._COM_corrections(spline_coordinates[:,:,rep].tolist()))
            self.spline_coordinates = np.array(self.spline_coordinates).round(8)
            for rep in range(len(xi)):
                self.spline_atoms.append(self._create_ASE_object(Z, 
                                         np.hsplit(self.spline_coordinates[rep,:,:], 5)[0], 
                                         self.confNEB[0].get_cell(),
                                         self.spline_energies[rep]))
        if isinstance(xi, float): self._write_PafiPath_atoms(self.atomsfname, self.spline_atoms[0], self.spline_coordinates[0])
        if self.print: write(self.NEBworkdir + 'pos_neb_spline_{}.xyz'.format(self.mfepstep), self.spline_atoms, format='extxyz')


#========================================================================================================================#
    def _set_PAFI_MD_cmd(self, rep, pair_style, pair_coeff, spatoms, sppositions, nsteps=10000, nthrow=0, mpi=None):
        """
        Post-MLACS
        Run a constrained MD.
        """
        
        self._write_PafiPath_atoms(self.atomsfname+rep, spatoms, sppositions)
        self.write_lammps_input(spatoms, pair_style, pair_coeff, nsteps, rep)
        lammps_command = "lmp_serial < " + self.lammpsfname + rep + "> log" + rep 
        if mpi is not None:
#            lammps_command = "srun -n 1 " + self.exe + " -in " + self.lammpsfname + rep + " -log log" + rep +' &'
            lammps_command = "ccc_mprun -E'--exclusive' -n 1 " + self.exe + " -in " + self.lammpsfname + rep + " -log log" + rep +' &'
        return lammps_command


#========================================================================================================================#
    def _create_ASE_object(self, Z, positions, cell, energy):
        """
        Create ASE Atoms object.  
        """
        atoms = Atoms(numbers=Z,
                      positions=positions,
                      cell=cell)
        calc  = SPC(atoms  = atoms,
                    energy = energy)
        atoms.set_calculator(calc)
        return atoms

#========================================================================================================================#
    def _COM_corrections(self, spline):
        """
        Correction of the path tangent to have zero center of mass
        """
        N = len(self.confNEB[0])
        pos, der, der2 = np.hsplit(np.array(spline), 3)
        com  = np.array([np.average(der[:,i]) for i in range(3)])
        norm = 0
        for i in range(N):
            norm += np.sum([ (der[i,j]-com[j])**2 for j in range(3)])
        norm = np.sqrt(norm)
        for i in range(N):
            spline[i].extend([(der[i,j] - com[j])/norm for j in range(3)]) 
            spline[i].extend([der2[i,j]/norm/norm for j in range(3)]) 
        return spline


#========================================================================================================================#
    def _write_PafiPath_atoms(self, filename, atoms, spline):
        """
        Write the lammps data file for a constrained MD, from an Atoms object.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at reaction coordinate xi.
        """
        symbol  = atoms.get_chemical_symbols()
        species = sorted(set(symbol))
        N       = len(symbol)
        cell    = atoms.get_cell()
        instr  = '#{0} (written by MLACS)\n\n'.format(filename)
        instr += '{0} atoms\n'.format(N)
        instr += '{0} atom types\n'.format(len(species))
        instr += '0 {0} xlo xhi\n'.format(cell[0,0])
        instr += '0 {0} ylo yhi\n'.format(cell[1,1])
        instr += '0 {0} zlo zhi\n'.format(cell[2,2])
        instr += '\nAtoms\n\n'
        for i in range(N): 
            strformat = '{:>6} '+ '{:>3} ' + ('{:12.8f} ' *3) + '\n'
            instr += strformat.format(i+1, species.index(symbol[i]) + 1, *spline[i,:3])
        instr += '\nPafiPath\n\n'
        for i in range(N): 
            strformat = '{:>6} '+ ('{:12.8f} ' *9) + '\n'
            instr += strformat.format(i+1, *spline[i,:3], *spline[i,9:])
        with open(filename, 'w') as w:
            w.write(instr)

#========================================================================================================================#
    def _read_lammpsdata_truepositions(self, filename, wrap=True):
        """
        Extract positions from lammpsdata files with memory of periodicity.
        Inspired from ASE.
        """
        
        (xy, xz, yz) = None, None, None
        (section, style) = None, None
        pos_in = {}
        travel_in = {}

        with open(filename, 'r') as r:
            for l in r:
                if 'atoms' in l: N = int(l.split()[0])
                if 'Atoms' in l: 
                    (section, _, style) = l.split()
                    continue
                if 'Velocities' in l: 
                    (section) = l.split()
                    continue
                if 'xlo xhi' in l:
                    (xlo, xhi) = [float(x) for x in l.split()[0:2]]
                if 'ylo yhi' in l:
                    (ylo, yhi) = [float(x) for x in l.split()[0:2]]
                if 'zlo zhi' in l:
                    (zlo, zhi) = [float(x) for x in l.split()[0:2]]
                if 'xy xz yz' in l:
                    (xy, xz, yz) = [float(x) for x in l.split()[0:3]]
                if section == 'Atoms':
                    fields = l.split()
                    if len(fields)==0: continue
                    id = int(fields[0])
                    if style == "atomic" and (len(fields) == 5 or len(fields) == 8):
                        # id type x y z [tx ty tz]
                        pos_in[id] = (
                            int(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                        )
                        if len(fields) == 8:
                            travel_in[id] = (
                                int(fields[5]),
                                int(fields[6]),
                                int(fields[7]),
                            )
                    else:
                        msg = "Style '{}' not supported or invalid number of fields {}".format(style, len(fields))
                        raise RuntimeError(msg)

        # set cell
        cell = np.zeros((3, 3))
        cell[0, 0] = xhi - xlo
        cell[1, 1] = yhi - ylo
        cell[2, 2] = zhi - zlo
        if xy is not None:
            cell[1, 0] = xy
        if xz is not None:
            cell[2, 0] = xz
        if yz is not None:
            cell[2, 1] = yz
        positions = np.zeros((N, 3))
        for id in pos_in.keys(): 
            ind = id - 1
            positions[ind, :] = [pos_in[id][1]+cell[0, 0]*travel_in[id][0],
                                 pos_in[id][2]+cell[1, 1]*travel_in[id][1], 
                                 pos_in[id][3]+cell[2, 2]*travel_in[id][2]]            

        return positions, cell

#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta is None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature, rng=self.rng)
        else:
            atoms.set_momenta(self.init_momenta)

#========================================================================================================================#
    def log_free_energy(self, xi):
        """
        """

        from mlacs.utilities.miscellanous import integrate_points as IntP
        from mlacs.utilities.miscellanous import interpolate_points as IP

        dF  = []
        psi = []
        cor = []
        maxjump = []
        for rep in range(len(xi)):
            dF.append(np.average(self.pafi[rep, 0]))
            psi.append(np.average(self.pafi[rep, 2]))
            cor.append(np.average(np.log(np.abs(self.pafi[rep, 2]/self.pafi[0,2]))))
            maxjump.append([x for x in self.pafi[rep,4].tolist() if x >= self.maxjump])
        dF   = np.array(dF)
        cor  = np.array(cor)
        psi  = np.array(psi)
        maxjump = np.array(maxjump)
        F    = -np.array(IntP(xi, dF, xi))
        Fcor = -np.array(IntP(xi, dF+kB*self.temperature*cor, xi))
        Ipsi = np.array(IntP(xi, psi, xi))
        if self.print:
            with open('free_energy.dat', 'w') as w:
                w.write('##  Free energy barier: {} eV  ##  xi  <dF/dxi>  <F(xi)>  <psi>  cor  Fcor(xi)  Nmaxjump  ##\n'.format(max(F) - min(F)))
                strformat = ('{:12.8f} ' * 7) + '\n'
                for i in range(len(xi)):
                    w.write(strformat.format(xi[i], dF[i], F[i], psi[i], kB*self.temperature*cor[i], Fcor[i], len(maxjump[i])))
        return Fcor


#========================================================================================================================#
    def _log_pafi(self, rep=''):
        """
        """
        input_string  = "# Logging\n"
        input_string += "variable    dU    equal f_pafihp[1]\n"
        input_string += "variable    dUerr equal f_pafihp[2]\n"
        input_string += "variable    psi   equal f_pafihp[3]\n"
        input_string += "variable    err   equal f_pafihp[4]\n"
        input_string += "compute     disp    all displace/atom\n"
        input_string += "compute     maxdisp all reduce max c_disp[4]\n"
        input_string += "variable    maxjump equal sqrt(c_maxdisp)\n"

        if self.isappend:
            input_string += 'fix logpafi all print 1 "${dU}  ${dUerr} ${psi} ${err} ${maxjump}" append pafi.log' + rep + ' title "# dU/dxi  (dU/dxi)^2  psi  err  maxjump"\n'.format(rep)
        else:
            input_string += 'fix logpafi all print 1 "${dU}  ${dUerr} ${psi} ${err} ${maxjump}" file pafi.log' + rep + ' title "# dU/dxi  (dU/dxi)^2  psi  err  maxjump"\n'.format(rep)
        input_string += "\n"
        return input_string

#========================================================================================================================#
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = self.damp
        if damp is None:
            damp = 100 * self.dt
        coord = self.NEBcoord
        if coord is None:
            coord = 0.5

        msg  = "NEB calculation as implemented in LAMMPS\n"
        msg += "Number of replicas :                     {0}\n".format(self.nreplica)
        msg += "String constant :                        {0}\n".format(self.Kspring)
        msg += "\n"
        msg += "Constrain dynamics as implemented in LAMMPS with fix PAFI\n"
        msg += "Temperature (in Kelvin) :                {0}\n".format(self.temperature)
        msg += "Number of MLMD equilibration steps :     {0}\n".format(self.nsteps_eq)
        msg += "Number of MLMD production steps :        {0}\n".format(self.nsteps)
        msg += "Timestep (in fs) :                       {0}\n".format(self.dt)
        msg += "Themostat damping parameter (in fs) :    {0}\n".format(damp)
        msg += "Reaction coordinate :                    {0}\n".format(coord)
        msg += "\n"
        return msg

#========================================================================================================================#
#    def compute_averaged_configurations(self, pair_style, pair_coeff, nsteps=10000, interval=10, nthrow=0):
#        """
#        Post-MLACS
#        CALL run_averageDM: Run a constrained MD and compute the average atomic positions and energies.
#        Construct Free energy derivative. 
#        """
#        self.isappend     = False
#        self.loginterval  = interval
#        self.trajinterval = interval
#
#        if nthrow is not 0:
#            nthrow = int(nthrow/interval)
#        
#        if self.logfile is None:
#            raise TypeError('You need to define the MD output file: \n - Invalid variable name logfile={}'.format(self.logfile))
#        if self.trajfile is None:
#            raise TypeError('You need to define the MD trajectory file: \n - Invalid variable name trajfile={}'.format(self.trajfile))
#        self.average_atoms     = []
#        self.average_energy    = []
#        self.average_positions = []
#        if self.print: write('pos_spline.xyz', self.spline_atoms, 'extxyz')
#        self.pafi = []
#        for rep in range(len(self.spline_atoms)):
#            atoms, energy, positions = self.run_averageDM(pair_style, pair_coeff, 
#                                                          self.spline_atoms[rep],
#                                                          self.spline_coordinates[rep,:,:], 
#                                                          nsteps, nthrow)
#            self.pafi.append(np.loadtxt(self.workdir+'pafi.log').T.tolist())
#            self.average_atoms.append(atoms)
#            self.average_energy.append(energy)
#            self.average_positions.append(positions)
#        if self.print: write('pos_averaged.xyz', self.average_atoms, 'extxyz')
#        self.pafi = np.array(self.pafi)
#        return np.array(self.average_atoms), np.array(self.average_energy), np.array(self.average_positions)
