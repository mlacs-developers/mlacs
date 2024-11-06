"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from concurrent.futures import ThreadPoolExecutor

import copy
import numpy as np

from ase.units import kB, J, kg, m

from .lammps_state import LammpsState

from ..core.manager import Manager
from ..utilities import save_cwd
from ..utilities import integrate_points as intgpts
from ..utilities.io_lammps import LammpsBlockInput


# ========================================================================== #
# ========================================================================== #
class PafiLammpsState(LammpsState):
    """
    Class to manage constrained MD along a reaction path using the fix Pafi
    with LAMMPS.

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.

    mep: :class:`NebLammpsState`
        Object contain all informations on the MEP (Minimum Energy Path).

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

    Examples
    --------

    >>> from ase.io import read
    >>> initial = read('A.traj')
    >>> final = read('B.traj')
    >>>
    >>> from mlacs.state import PafiLammpsState, NebLammpsState
    >>> neb = NebLammpsState([initial, final])
    >>> state = PafiLammpsState(temperature=300, mep=neb)
    >>> state.run_dynamics(atoms, mlip.pair_style, mlip.pair_coeff)
    """

    def __init__(self, temperature, mep=None, maxjump=0.4,
                 dt=1.5, damp=None, prt=False, langevin=True,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, blocks=None, **kwargs):

        super().__init__(temperature=temperature, dt=dt, damp=damp,
                         langevin=langevin,
                         nsteps=nsteps, nsteps_eq=nsteps_eq, logfile=logfile,
                         trajfile=trajfile, loginterval=loginterval,
                         blocks=blocks, **kwargs)

        self.temperature = temperature
        self.mep = mep
        if mep is None:
            raise TypeError('A reaction path must be given!')
        self.mep.print = prt

        self.print = prt
        self.maxjump = maxjump

        self._replica = None

# ========================================================================== #
    @Manager.exec_from_path
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     elements=None):
        """
        Run state function.
        """

        # Run NEB calculation.
        self.mep.workdir = self.workdir
        self.mep.folder = self.folder
        self.mep.subfolder = 'TransPath'
        self.mep.run_dynamics(self.mep.patoms.initial,
                              pair_style, pair_coeff,
                              model_post, atom_style, elements)
        supercell = self.mep.patoms.splined
        self.isrestart = False

        # Run Pafi dynamic at xi.
        atoms = LammpsState.run_dynamics(self, supercell,
                                         pair_style, pair_coeff, model_post,
                                         atom_style, eq, elements)
        return atoms.copy()

# ========================================================================== #
    @Manager.exec_from_path
    def run_pafipath_dynamics(self,
                              supercell,
                              pair_style,
                              pair_coeff,
                              model_post=None,
                              atom_style="atomic",
                              ncpus=1,
                              restart=0,
                              xi=None,
                              nsteps=10000,
                              nthrow=2000,
                              elements=None):
        """
        Run full Pafi path.
        """

        if xi is None:
            xi = np.arange(0, 1.01, 0.01)
        nrep = len(xi)

        afname = self.atomsfname
        lfname = self.lammpsfname

        # Run NEB calculation.
        self.mep.workdir = self.folder
        self.mep.folder = 'TransPath'
        self.mep.run_dynamics(self.mep.patoms.initial,
                              pair_style, pair_coeff,
                              model_post, atom_style, elements)
        self.isrestart = False

        # Run Pafi dynamics.
        with save_cwd(), ThreadPoolExecutor(max_workers=ncpus) as executor:
            for rep in range(restart, nrep):
                worker = copy.deepcopy(self)
                worker._replica = rep
                worker.atomsfname = afname + f'.{rep}'
                worker.lammpsfname = lfname + f'.{rep}'
                worker.subfolder = f'PafiPath_{rep}'
                self.mep.patoms.xi = xi[rep]
                atoms = self.mep.patoms.splined.copy()
                atoms.set_pbc([1, 1, 1])
                executor.submit(LammpsState.run_dynamics,
                                *(worker, atoms, pair_style, pair_coeff,
                                  model_post, atom_style, False, elements))
            executor.shutdown(wait=True)

        # Reset some attributes.
        self._replica = None
        self.atomsfname = afname
        self.lammpsfname = lfname
        return self.log_free_energy(xi, nthrow)

# ========================================================================== #
    @Manager.exec_from_path
    def _write_lammps_atoms(self, atoms, atom_style, elements=None):
        """

        """
        filename = self.path / self.atomsfname

        splat = self.mep.patoms.splined
        splR = self.mep.patoms.splR
        splDR = self.mep.patoms.splDR
        splD2R = self.mep.patoms.splD2R

        from ase.calculators.lammps import Prism, convert
        symbol = splat.get_chemical_symbols()
        species = sorted(set(symbol))
        p = Prism(splat.get_cell())
        xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(),
                                            'distance', 'ASE', 'metal')
        instr = f'#{filename} (written by MLACS)\n\n'
        instr += f'{len(symbol)} atoms\n'
        instr += f'{len(species)} atom types\n'
        instr += f'0 {xhi} xlo xhi\n'
        instr += f'0 {yhi} ylo yhi\n'
        instr += f'0 {zhi} zlo zhi\n'
        if p.is_skewed():
            instr += f'{xy} {xz} {yz}  xy xz yz\n'
        instr += '\nAtoms\n\n'
        for i, r in enumerate(splR):
            strformat = '{:>6} ' + '{:>3} ' + ('{:12.8f} ' * 3) + '\n'
            instr += strformat.format(i+1, species.index(symbol[i]) + 1, *r)
        instr += '\nPafiPath\n\n'
        for i, (r, dr, d2r) in enumerate(zip(splR, splDR, splD2R)):
            strformat = '{:>6} ' + ('{:12.8f} ' * 9) + '\n'
            instr += strformat.format(i+1, *r, *dr, *d2r)
        with open(filename, 'w') as w:
            w.write(instr)

# ========================================================================== #
    def _get_block_init(self, atom_style, pbc, el, masses):
        """

        """
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))

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
        _rep = self._replica
        if self._replica is None:
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
              f'pafi.{_rep}.log title "# dU/dxi (dU/dxi)^2 psi err maxjump"'
        block("pafilog", txt)
        return block

# ========================================================================== #
    @Manager.exec_from_workdir
    def log_free_energy(self, xi, nthrow=2000, _ref=0):
        """
        Extract the MFEP gradient from log files.
        Integrate the MFEP and compute the Free energy barrier.
        """
        temp = self.temperature
        meff = self.mep.patoms.masses

        self.pafi = []
        for rep in range(len(xi)):
            self.subfolder = f'PafiPath_{rep}'
            self.prefix = f'path.{rep}'
            logfile = str(self.get_filepath('.log'))
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
        if self.mep.patoms._xi is None and self.mep.patoms.mode == 'saddle':
            xi = None
        else:
            xi = self.mep.patoms.xi

        msg = self.mep.log_recap_state()
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
