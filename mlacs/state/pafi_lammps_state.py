import os
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ase.units import kB, J, kg, m
from ase.io import read

from .lammps_state import LammpsState
from .neb_lammps_state import NebLammpsState

from ..utilities import get_elements_Z_and_masses

from ..utilities.io_lammps import (get_general_input,
                                   get_log_input,
                                   get_traj_input,
                                   get_interaction_input,
                                   get_last_dump_input,
                                   get_pafi_input,
                                   get_pafi_log_input)

from ..utilities import integrate_points as IntP


# ========================================================================== #
# ========================================================================== #
class PafiLammpsState(LammpsState, NebLammpsState):
    """
    Class to manage constrained MD along a NEB reaction coordinate using
    the fix Pafi with LAMMPS.

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.
    reaction_coordinate: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        if ``None``, automatic search of the saddle point.
        Default ``None``
    Kspring: :class:`float`
        Spring constante for the NEB calculation.
        Default ``1.0``
    maxjump: :class:`float`
        Maximum atomic jump authorized for the free energy calculations.
        Configurations with an high `maxjump` will be removed.
        Default ``0.4``
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    damp: :class:`float` or ``None``
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    langevin: :class:`Bool`
        If ``True``, a Langevin thermostat is used.
        Else, a Brownian dynamic is used.
        Default ``True``
    linearmode: :class:`Bool`
        If ``True``, the reaction coordinate function is contructed using 
        a linear interpolation of the true 3N coordinates. 
        Else, the reaction coordinate function is determined using NEB.
        Default ``False``
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    prt : :class:`Bool` (optional)
        Printing options. Default ``True``
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
    """
    def __init__(self,
                 temperature,
                 configurations,
                 reaction_coordinate=None,
                 Kspring=1.0,
                 maxjump=0.4,
                 dt=1.5,
                 damp=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 langevin=True,
                 linearmode=False,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=49,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 prt=True,
                 workdir=None):
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
                             workdir)
        NebLammpsState.__init__(self,
                                configurations,
                                reaction_coordinate=None,
                                Kspring=1.0,
                                dt=dt)

        self.temperature = temperature
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq
        self.NEBcoord = reaction_coordinate
        self.finder = None
        self.xilinear = linearmode
        if self.NEBcoord is None:
            self.splprec = 1001
            self.finder = []
        self.print = prt
        self.Kspring = Kspring
        self.maxjump = maxjump
        self.dt = dt
        self.damp = damp
        self.langevin = langevin
        self.confNEB = configurations
        if len(self.confNEB) != 2:
            raise TypeError('First and last configurations are not defined')
        self._get_lammps_command_replica()

        self.ispimd = False
        self.isrestart = False
        self.isappend = False

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     bonds=None,
                     angles=None,
                     bond_style=None,
                     bond_coeff=None,
                     angle_style=None,
                     angle_coeff=None,
                     eq=False,
                     workdir=None):
        """
        Run state function.
        """
        self.run_NEB(pair_style,
                     pair_coeff,
                     model_post,
                     atom_style,
                     bonds,
                     angles,
                     bond_style,
                     bond_coeff,
                     angle_style,
                     angle_coeff,
                     workdir)
        self.extract_NEB_configurations()
        self.compute_spline()
        self.isrestart = False
        atoms = self.run_hpdynamics(supercell,
                                    pair_style,
                                    pair_coeff,
                                    model_post,
                                    atom_style,
                                    bonds,
                                    angles,
                                    bond_style,
                                    bond_coeff,
                                    angle_style,
                                    angle_coeff,
                                    eq)
        return atoms.copy()

# ========================================================================== #
    def run_hpdynamics(self,
                       supercell,
                       pair_style,
                       pair_coeff,
                       model_post=None,
                       atom_style="atomic",
                       bonds=None,
                       angles=None,
                       bond_style=None,
                       bond_coeff=None,
                       angle_style=None,
                       angle_coeff=None,
                       eq=False,
                       rep=None):
        """
        Function to run the PAFI dynamics
        """
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        if atom_style is None:
            atom_style = "atomic"

        atoms = supercell.copy()

        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        if rep is None:
            fname = self.workdir + self.lammpsfname
            atfname = self.workdir + self.atomsfname
            spatoms = self.spline_atoms[0]
            spcoord = self.spline_coordinates[0]
        else:
            fname = self.MFEPworkdir + self.lammpsfname + f'.{rep}'
            atfname = self.MFEPworkdir + self.atomsfname + f'.{rep}'
            spatoms = atoms
            spcoord = self.spline_coordinates[rep]

        self._write_PafiPath_atoms(atfname,
                                   spatoms,
                                   spcoord)
        self.write_lammps_input_pafi(atoms,
                                     atom_style,
                                     bond_style,
                                     bond_coeff,
                                     angle_style,
                                     angle_coeff,
                                     pair_style,
                                     pair_coeff,
                                     model_post,
                                     nsteps,
                                     fname,
                                     atfname,
                                     rep)

        lammps_command = self.cmd + " -in " + fname + \
            " -sc out.lmp"
        cwd = self.workdir
        if rep is not None:
            lammps_command += f'.{rep}'
            cwd = self.MFEPworkdir
        lmp_handle = run(lammps_command,
                         shell=True,
                         cwd=cwd,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

        if charges is not None:
            init_charges = atoms.get_initial_charges()
        atoms = read(self.workdir + "configurations.out")
        if charges is not None:
            atoms.set_initial_charges(init_charges)

        return atoms.copy()

# ========================================================================== #
    def run_MFEP(self,
                 pair_style,
                 pair_coeff,
                 model_post=None,
                 atom_style="atomic",
                 bonds=None,
                 angles=None,
                 bond_style=None,
                 bond_coeff=None,
                 angle_style=None,
                 angle_coeff=None,
                 workdir=None,
                 ncpus=1,
                 restart=0,
                 xi=None,
                 nsteps=10000,
                 interval=10,
                 nthrow=2000):
        """
        Run a MFEP calculation with lammps. Use replicas.
        """
        self.nsteps = nsteps
        if workdir is not None:
            self.workdir = workdir
        self.MFEPworkdir = self.workdir + "MFEP/"
        if not os.path.exists(self.MFEPworkdir):
            os.makedirs(self.MFEPworkdir)
        if xi is None:
            xi = np.arange(0, 1.01, 0.01)
        self.run_NEB(pair_style,
                     pair_coeff,
                     model_post,
                     atom_style,
                     bonds,
                     angles,
                     bond_style,
                     bond_coeff,
                     angle_style,
                     angle_coeff,
                     workdir)
        self.extract_NEB_configurations()
        self.compute_spline(xi)
        nrep = len(self.spline_atoms)
        with ThreadPoolExecutor(max_workers=ncpus) as executor:
            for rep in range(restart, nrep):
                atoms = self.spline_atoms[rep].copy()
                atoms.set_pbc([1, 1, 1])
                executor.submit(self.run_hpdynamics,
                                *(atoms,
                                  pair_style,
                                  pair_coeff,
                                  model_post,
                                  atom_style,
                                  bonds,
                                  angles,
                                  bond_style,
                                  bond_coeff,
                                  angle_style,
                                  angle_coeff,
                                  False,
                                  rep))
        F = self.log_free_energy(xi,
                                 self.MFEPworkdir,
                                 nthrow)
        return F

# ========================================================================== #
    def write_lammps_input_pafi(self,
                                atoms,
                                atom_style,
                                bond_style,
                                bond_coeff,
                                angle_style,
                                angle_coeff,
                                pair_style,
                                pair_coeff,
                                model_post,
                                nsteps,
                                fname,
                                atfname,
                                rep=None):
        """
        Write the LAMMPS input for the constrained MD simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms)
        pbc = atoms.get_pbc()

        custom = "atom_modify  map array sort 0 0.0\n"
        custom += "neigh_modify every 2 delay 10" + \
                  " check yes page 1000000 one 100000\n\n"
        custom += "fix 1 all property/atom d_nx d_ny d_nz" + \
                  " d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz\n"
        filename = atfname + " fix 1 NULL PafiPath"
        input_string = ""
        input_string += get_general_input(pbc,
                                          masses,
                                          charges,
                                          atom_style,
                                          filename,
                                          custom)
        input_string += get_interaction_input(bond_style,
                                              bond_coeff,
                                              angle_style,
                                              angle_coeff,
                                              pair_style,
                                              pair_coeff,
                                              model_post)
        input_string += get_pafi_input(self.dt / 1000,
                                       self.temperature,
                                       self.rng.integers(99999),
                                       self.damp,
                                       self.langevin)
        if self.logfile is not None:
            input_string += get_log_input(self.loginterval, self.logfile)
        if self.trajfile is not None:
            input_string += get_traj_input(self.loginterval,
                                           self.trajfile,
                                           elem)
        if rep is None:
            rep = 0
        input_string += get_pafi_log_input(rep,
                                           self.isappend)
        input_string += get_last_dump_input(self.workdir,
                                            elem,
                                            nsteps)
        input_string += f"run  {nsteps}"

        with open(fname, "w") as f:
            f.write(input_string)

# ========================================================================== #
    def _write_PafiPath_atoms(self, filename, atoms, spline):
        """
        Write the lammps data file for a constrained MD, from an Atoms object.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at
                reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at
                reaction coordinate xi.
        """
        from ase.calculators.lammps import Prism, convert
        symbol = atoms.get_chemical_symbols()
        species = sorted(set(symbol))
        N = len(symbol)
        p = Prism(atoms.get_cell())
        xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(),
                                            'distance', 'ASE', 'metal')
        instr = f'#{filename} (written by MLACS)\n\n'
        instr += f'{N} atoms\n'
        instr += f'{len(species)} atom types\n'
        instr += f'0 {xhi} xlo xhi\n'
        instr += f'0 {yhi} ylo yhi\n'
        instr += f'0 {zhi} zlo zhi\n'
        if p.is_skewed():
            instr += f'{xy} {xz} {yz}  xy xz yz\n'
        instr += '\nAtoms\n\n'
        for i in range(N):
            strformat = '{:>6} ' + '{:>3} ' + ('{:12.8f} ' * 3) + '\n'
            instr += strformat.format(i+1, species.index(symbol[i]) + 1,
                                      *spline[i, :3])
        instr += '\nPafiPath\n\n'
        for i in range(N):
            strformat = '{:>6} ' + ('{:12.8f} ' * 9) + '\n'
            instr += strformat.format(i+1, *spline[i, :3], *spline[i, 9:])
        with open(filename, 'w') as w:
            w.write(instr)

# ========================================================================== #
    def log_free_energy(self, xi, workdir, nthrow=2000, _ref=0):
        """
        Extract the MFEP gradient from log files.
        Integrate the MFEP and compute the Free energy barier.
        """

        self.pafi = []
        for rep in range(len(xi)):
            logfile = workdir + f'pafi.log.{rep}'
            data = np.loadtxt(logfile).T[:, nthrow:].tolist()
            self.pafi.append(data)
        self.pafi = np.array(self.pafi)

        dF = []
        psi = []
        cor = []
        maxjump = []
        ntot = len(self.pafi[rep, 0])
        for rep in range(len(xi)):
            # Remove steps with high jumps, the default value is 0.4.
            mj = self.pafi[rep, 4].tolist()
            dF.append(np.average([self.pafi[rep, 0, i]
                      for i, x in enumerate(mj) if x < self.maxjump]))
            psi.append(np.average([self.pafi[rep, 2, i]
                       for i, x in enumerate(mj) if x < self.maxjump]))
            cor.append(np.average([np.log(np.abs(
                       self.pafi[rep, 2, i] / self.pafi[_ref, 2, i]))
                       for i, x in enumerate(mj) if x < self.maxjump]))
            maxjump.append([x for x in mj if x > self.maxjump])
#            dF.append(np.average(self.pafi[rep, 0]))
#            psi.append(np.average(self.pafi[rep, 2]))
#            cor.append(np.average(
#                np.log(np.abs(self.pafi[rep, 2] / self.pafi[_ref, 2]))))
        dF = np.array(dF)
        cor = np.array(cor)
        psi = np.array(psi)
        maxjump = np.array(maxjump)
        F = -np.array(IntP(xi, dF, xi))
        int_xi = np.linspace(xi[0], xi[F.argmax()], len(xi)//2)
        v = np.array(IntP(xi, np.exp(- F / kB * self.temperature), int_xi))
        vo = np.sqrt((kB * self.temperature * J) /
                     (2 * np.pi * self.eff_masses * kg)) / (v[-1] * m)
        Fcor = -np.array(IntP(xi, dF + kB * self.temperature * cor, xi))
        dFM = max(F) - min(F)
        # Ipsi = np.array(IntP(xi, psi, xi))
        if self.print:
            with open(self.workdir + 'free_energy.dat', 'w') as w:
                w.write(f'##  Free energy barier: {dFM} eV | ' +
                        f'frequency: {vo} s-1 | ' +
                        f'effective mass: {self.eff_masses} uma\n' +
                        '##  xi  <dF/dxi>  <F(xi)>  <psi>  ' +
                        'cor  Fcor(xi) v(xi) NUsedConf ##\n')
                strformat = ('{:18.10f} ' * 6) + ' {} {}\n'
                for i in range(len(xi)):
                    _v = v[-1]
                    if i < len(v):
                        _v = v[i]
                    w.write(strformat.format(xi[i],
                                             dF[i],
                                             F[i],
                                             psi[i],
                                             kB * self.temperature * cor[i],
                                             Fcor[i],
                                             _v,
                                             ntot - len(maxjump[i])))
        return dFM, vo, self.eff_masses

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = self.damp
        if damp is None:
            damp = 100 * self.dt
        coord = self.NEBcoord

        msg = "NEB calculation as implemented in LAMMPS\n"
        msg += f"Number of replicas :                     {self.nreplica}\n"
        msg += f"String constant :                        {self.Kspring}\n"
        msg += "\n"
        msg += "Constrain dynamics as implemented in LAMMPS with fix PAFI\n"
        msg += f"Temperature (in Kelvin) :                {self.temperature}\n"
        msg += f"Number of MLMD equilibration steps :     {self.nsteps_eq}\n"
        msg += f"Number of MLMD production steps :        {self.nsteps}\n"
        msg += f"Timestep (in fs) :                       {self.dt}\n"
        msg += f"Themostat damping parameter (in fs) :    {damp}\n"
        if isinstance(coord, float):
            msg += f"Reaction coordinate :                    {coord}\n"
        elif coord is None:
            msg += "Reaction coordinate :                    Automatic\n"
        else:
            step = coord[1]-coord[0]
            i, f = (coord[0], coord[-1])
            msg += f"Reaction interval :                      [{i} : {f}]\n"
            msg += f"Reaction step interval :                 {step}\n"
        msg += "\n"
        return msg


# ========================================================================== #
# ========================================================================== #
class BlueMoonLammpsState(PafiLammpsState):
    """
    Class to manage constrained MD along a linear reaction coordinate using
    the fix Pafi with LAMMPS. This is similar to a Blue Moon sampling.

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.
    reaction_coordinate: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        if ``None``, automatic search of the saddle point.
        Default ``None``
    maxjump: :class:`float`
        Maximum atomic jump authorized for the free energy calculations.
        Configurations with an high `maxjump` will be removed.
        Default ``0.4``
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    damp: :class:`float` or ``None``
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    langevin: :class:`Bool`
        If ``True``, a Langevin thermostat is used.
        Else, a Brownian dynamic is used.
        Default ``True``
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    prt : :class:`Bool` (optional)
        Printing options. Default ``True``
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
    """
    def __init__(self,
                 temperature,
                 configurations,
                 reaction_coordinate=None,
                 maxjump=0.4,
                 dt=1.5,
                 damp=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 langevin=True,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=49,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 prt=True,
                 workdir=None):
        PafiLammpsState.__init__(self,
                                 temperature,
                                 configurations,
                                 reaction_coordinate,
                                 1.0,
                                 maxjump,
                                 dt,
                                 damp,
                                 nsteps,
                                 nsteps_eq,
                                 langevin,
                                 fixcm,
                                 logfile,
                                 trajfile,
                                 interval,
                                 loginterval,
                                 trajinterval,
                                 rng,
                                 init_momenta,
                                 prt,
                                 workdir)
        self.xilinear = True
