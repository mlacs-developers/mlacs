"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os

import numpy as np

from ase.units import kB
from ase.io import read

from mlacs.ti.solids import EinsteinSolidState
from mlacs.ti.liquids import UFLiquidState


def prepare_ti(trajprefix,
               pair_style,
               pair_coeff,
               state="solid",
               atoms_start=None,
               nthrow=20,
               repeat_cell=None,
               temperature=None,
               k=None,
               p=50,
               sigma=2.0,
               dt=1,
               damp=None,
               nsteps=10000,
               nsteps_eq=5000,
               nsteps_msd=5000,
               rng=None,
               suffixdir=None,
               logfile=True,
               trajfile=True,
               interval=500,
               loginterval=500,
               trajinterval=500
               ):

    traj  = read(trajprefix + ".traj", index=":")
    ntraj = len(traj)
    nat   = len(traj[0])

    if atoms_start == None:
        if state == "solid":
            atoms_start = traj[0]
        elif state == "liquid":
            atoms_start = traj[-1]

    # Get average cell and temperature
    cell = []
    temp = []
    for i in range(nthrow, ntraj):
        cell.append(traj[i].get_cell())
        temp.append(traj[i].get_temperature())
    cell = np.mean(cell, axis=0)
    # Set atoms to average cell
    atoms_start.set_cell(cell, True)

    if repeat_cell is not None:
        atoms_start = atoms_start.repeat(repeat_cell)

    if temperature is None:
        temperature = np.mean(temp[nthrow:])

    # get free energy corrections
    kBT = kB * temperature
    vtrue, vmlip = np.loadtxt(trajprefix + "_potential.dat", unpack=True)
    dv = (vtrue - vmlip)[nthrow:]
    fcorr1 = dv.mean() / nat
    fcorr2 = -0.5 * dv.var() / (nat * kBT)


    if state == "solid":
        state = EinsteinSolidState(atoms_start,
                                   pair_style,
                                   pair_coeff,
                                   temperature,
                                   fcorr1,
                                   fcorr2,
                                   k,
                                   dt,
                                   damp,
                                   nsteps,
                                   nsteps_eq,
                                   nsteps_msd,
                                   rng,
                                   suffixdir,
                                   logfile,
                                   trajfile,
                                   interval,
                                   loginterval,
                                   trajinterval
                                  )

    elif state == "liquid":
        state = UFLiquidState(atoms_start,
                              pair_style,
                              pair_coeff,
                              temperature,
                              fcorr1,
                              fcorr2,
                              p,
                              sigma,
                              dt,
                              damp,
                              nsteps,
                              nsteps_eq,
                              rng,
                              suffixdir,
                              logfile,
                              trajfile,
                              interval,
                              loginterval,
                              trajinterval
                             )
    else:
        msg = "state should be either \"solid\" or \"liquid\""
        raise ValueError(msg)
    return state
