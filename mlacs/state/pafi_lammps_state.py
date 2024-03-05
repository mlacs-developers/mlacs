from concurrent.futures import ThreadPoolExecutor

import copy
import numpy as np

from ase.units import kB, J, kg, m

from .lammps_state import LammpsState

from ..utilities import get_elements_Z_and_masses
from ..utilities import integrate_points as intgpts
from ..utilities.io_lammps import LammpsBlockInput


# ========================================================================== #
# ========================================================================== #
class PafiLammpsState(LammpsState):
    """
    Class to manage constrained MD along a NEB reaction coordinate using
    the fix Pafi with LAMMPS.

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.

    path: :class:`NebLammpsState`
        NebLammpsState object contain all informations on the transition path.

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

    def __init__(self, temperature, path=None, maxjump=0.4, dt=1.5, damp=None,
                 prt=False, langevin=True,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, workdir=None, blocks=None):
        super().__init__(temperature=temperature, dt=dt, damp=damp,
                         langevin=langevin,
                         nsteps=nsteps, nsteps_eq=nsteps_eq, logfile=logfile,
                         trajfile=trajfile, loginterval=loginterval,
                         workdir=workdir, blocks=blocks)


        self.temperature = temperature
        self.path = path
        if path is None:
            raise TypeError('A NebLammpsState must be given!')
        self.path.print = prt
        if self.path.xi is None:
            self.path.mode = None
        else:
            self.path.mode = self.path.xi
        self.path.workdir = self.workdir / 'TransPath'
        self.print = prt
        self.maxjump = maxjump

        self.replica = None

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     workdir=None):
        """
        Run state function.
        """

        # Run NEB calculation.
        self.path.run_dynamics(self.path.atoms[0],
                               pair_style,
                               pair_coeff,
                               model_post,
                               atom_style)
        self.path.extract_NEB_configurations()
        self.path.compute_spline()
        supercell = self.path.spline_atoms[0].copy()
        self.isrestart = False

        # Run Pafi dynamic at xi.
        atoms = LammpsState.run_dynamics(self,
                                         supercell,
                                         pair_style,
                                         pair_coeff,
                                         model_post,
                                         atom_style,
                                         eq)

        return atoms.copy()

# ========================================================================== #
    def run_pafipath_dynamics(self,
                              supercell,
                              pair_style,
                              pair_coeff,
                              model_post=None,
                              atom_style="atomic",
                              workdir=None,
                              ncpus=1,
                              restart=0,
                              xi=None,
                              nsteps=10000,
                              nthrow=2000):
        """
        Run full Pafi path.
        """

        if workdir is None:
            workdir = self.workdir
        if xi is None:
            xi = np.arange(0, 1.01, 0.01)
        nrep = len(xi)

        afname = self.atomsfname
        lfname = self.lammpsfname

        # Run NEB calculation.
        self.path.run_dynamics(self.path.atoms[0],
                               pair_style,
                               pair_coeff,
                               model_post,
                               atom_style)
        self.path.extract_NEB_configurations()
        self.path.compute_spline(xi)
        self.isrestart = False

        # Run Pafi dynamics.
        with ThreadPoolExecutor(max_workers=ncpus) as executor:
            for rep in range(restart, nrep):
                worker = copy.deepcopy(self)
                worker.replica = rep
                worker.atomsfname = afname + f'.{rep}'
                worker.lammpsfname = lfname + f'.{rep}'
                worker.workdir = workdir / f'PafiPath_{rep}'
                atoms = self.path.spline_atoms[rep].copy()
                atoms.set_pbc([1, 1, 1])
                executor.submit(LammpsState.run_dynamics,
                                *(worker, atoms, pair_style, pair_coeff,
                                  model_post, atom_style, False))

        # Reset some attributes.
        self.replica = None
        self.workdir = workdir
        self.atomsfname = afname
        self.lammpsfname = lfname
        return self.log_free_energy(xi, workdir, nthrow)

# ========================================================================== #
    def _write_lammps_atoms(self, atoms, atom_style):
        """

        """
        rep = self.replica
        afnames = self.workdir / self.atomsfname

        if rep is None:
            splatoms = self.path.spline_atoms[0]
            splcoord = self.path.spline_coordinates[0]
        else:
            splatoms = self.path.spline_atoms[rep]
            splcoord = self.path.spline_coordinates[rep]

        self._write_PafiPath_atoms(afnames, splatoms, splcoord)

# ========================================================================== #
    def _get_block_init(self, atoms, atom_style):
        """

        """
        pbc = atoms.get_pbc()
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        block = LammpsBlockInput("init", "Initialization")
        block("units", "units metal")
        block("boundary", f"boundary {pbc}")
        block("atom_style", f"atom_style {atom_style}")
        block("atom_modify", "atom_modify  map array sort 0 0.0")
        txt = "neigh_modify every 2 delay 10" + \
              " check yes page 1000000 one 100000"
        block("neigh_modify", txt)
        txt = "fix pat all property/atom d_nx d_ny d_nz" + \
              " d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz"
        block("property_atoms", txt)
        txt = f"read_data {self.atomsfname} fix pat NULL PafiPath"
        block("read_data", txt)
        for i, mass in enumerate(masses):
            block(f"mass{i}", f"mass {i+1}  {mass}")
        return block

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        """

        """
        temp = self.temperature
        seed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("pafi", "Pafi dynamic")

        block("timestep", f"timestep {self.dt / 1000}")
        block("thermo", "thermo 1")
        block("min_style", "min_style fire")  # RB test if we can modify
        txt = "compute cpat all property/atom d_nx d_ny d_nz " + \
              "d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz"
        block("c_pat", txt)
        block("run_compute", "run 0")

        # RB
        # If we are using Langevin, we want to remove the random part
        # of the forces. RB don't know if i have to do it.
        # if self.langevin:
        #     block("rmv_langevin", "fix ff all store/force")

        if self.langevin:
            txt = f"fix pafi all pafi cpat {temp} {self.damp} {seed} " + \
                  "overdamped no com yes"
            block("langevin", txt)
        else:
            txt = f"fix pafi all pafi cpat {temp} {self.damp} {seed} " + \
                  "overdamped yes com yes"
            block("brownian", txt)
        block("run_fix", "run 0")
        block("minimize", "minimize 0 0 250 250")
        block("reset_timestep", "reset_timestep 0")
        return block

# ========================================================================== #
    def _get_block_custom(self):
        """

        """
        _rep = self.replica
        if self.replica is None:
            _rep = 0

        block = LammpsBlockInput("pafilog", "Pafi log files")
        block("v_dU", "variable dU equal f_pafi[1]")
        block("v_dUe", "variable dUerr equal f_pafi[2]")
        block("v_psi", "variable psi equal f_pafi[3]")
        block("v_err", "variable err equal f_pafi[4]")
        block("c_disp", "compute disp all displace/atom")
        block("c_maxdisp", "compute maxdisp all reduce max c_disp[4]")
        block("v_maxjump", "variable maxjump equal sqrt(c_maxdisp)")
        txt = 'fix pafilog all print 1 ' + \
              '"${dU}  ${dUerr} ${psi} ${err} ${maxjump}" file ' + \
              f'pafi.log.{_rep} title "# dU/dxi (dU/dxi)^2 psi err maxjump"'
        block("pafilog", txt)
        return block

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
        if workdir is None:
            workdir = self.workdir
        temp = self.temperature
        meff = self.path.eff_masses

        self.pafi = []
        for rep in range(len(xi)):
            logfile = workdir / f'PafiPath_{rep}' / f'pafi.log.{rep}'
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
        F = -np.array(intgpts(xi, dF, xi))
        int_xi = np.linspace(xi[0], xi[F.argmax()], len(xi)//2)
        v = np.array(intgpts(xi, np.exp(- F / kB * temp), int_xi))
        vo = np.sqrt((kB * temp * J) / (2 * np.pi * meff * kg)) / (v[-1] * m)
        Fcor = -np.array(intgpts(xi, dF + kB * temp * cor, xi))
        # Ipsi = np.array(intgpts(xi, psi, xi))
        txt = f'##  Max free energy: {max(F)} eV | frequency: {vo} s-1 | ' + \
              f'effective mass: {meff} uma\n' + \
              '##  xi <dF/dxi> <F(xi)> <psi> cor Fcor(xi) v(xi) NConf ##\n'
        with open(self.workdir / 'free_energy.dat', 'w') as w:
            w.write(txt)
            strformat = ('{:18.10f} ' * 6) + ' {} {}\n'
            for i in range(len(xi)):
                _v = v[-1]
                if i < len(v):
                    _v = v[i]
                w.write(strformat.format(xi[i], dF[i], F[i], psi[i],
                                         kB * temp * cor[i], Fcor[i], _v,
                                         ntot - len(maxjump[i])))
        return np.r_[[F, Fcor, _v]]

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = self.damp
        if damp is None:
            damp = 10 * self.dt
        xi = self.path.xi

        msg = self.path.log_recap_state()
        msg += "Constrained dynamics as implemented in LAMMPS with fix PAFI\n"
        msg += f"Temperature (in Kelvin) :                {self.temperature}\n"
        msg += f"Number of MLMD equilibration steps :     {self.nsteps_eq}\n"
        msg += f"Number of MLMD production steps :        {self.nsteps}\n"
        msg += f"Timestep (in fs) :                       {self.dt}\n"
        msg += f"Themostat damping parameter (in fs) :    {damp}\n"
        if isinstance(xi, float):
            msg += f"Path coordinate :                        {xi}\n"
        elif xi is None:
            msg += "Path coordinate :                        Automatic\n"
        else:
            step = xi[1]-xi[0]
            i, f = (xi[0], xi[-1])
            msg += f"Path interval :                          [{i} : {f}]\n"
            msg += f"Path step interval :                     {step}\n"
        msg += "\n"
        return msg
