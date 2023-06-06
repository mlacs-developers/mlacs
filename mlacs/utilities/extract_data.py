import numpy as np

from ase.io import read
from ase.units import Bohr, GPa

# ========================================================================== #


def extract_data_from_files(file_confs,
                            weights=None,
                            **kwargs):
    
    '''
    Funtion to prepare input as ase object for pressure/volume calculations

    Parameters
    ----------

    file_confs: .json or .traj file of generated configurations
    weights: .dat file of weights 
        The weights of the configurations, should sum up to one
   
    Optional
    --------
    '''

    confs = read('configurations.json', index=':')
    
    if weights:
        weights = np.loadtxt('weights.dat')

    return extract_data(confs, weights)


def extract_data(confs,
                 weights=None,
                 fname=None,
                 **kwargs):
    '''
    Funtion to compute pressure/vol from NVT simulations

    Parameters
    ----------
    confs: generated configurations as list of ase atoms objects
 
    weights: list of float
        The weights of the configurations, should sum up to one
   
    Optional
    --------
    '''

    # Initialize some variables and inputs 
    nconfs = len(confs)
    
    # To avoid complication, we add normalized weights if not given
    if weights is None:
        weights = np.ones(nconfs) / nconfs

    cell     = confs[0].get_cell() / Bohr
    natom    = len(confs[0].get_scaled_positions(wrap=False))
    volume   = cell[0][0]*cell[1][1]*cell[2][2] / natom
    stress   = np.array([at.get_stress(voigt=True, include_ideal_gas=True) /GPa\
                         for at in confs])

    stress_av = [] 
    pressure_pot = 0
    for i in range(len(stress)):
        sum = 0
        for j in range(3):
            sum += stress[i][j]
        stress_av.append(sum/3*weights[i])
        pressure_pot += -stress_av[i]

    # pressure_kin =  kB*temperature/Ha/(volume*29421.033)
    # p = p /29421.033
    return volume, pressure_pot
