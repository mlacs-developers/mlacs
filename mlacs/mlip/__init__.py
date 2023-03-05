"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from .mlip_manager import MlipManager
from .linear_mlip import LinearMlip
from .mlip_lammps import LammpsMlip
from .linearfit_factory import FitLammpsMlip
try:
    from .mlip_lammps_nn import LammpsMlipNn
    __all__ = ['MlipManager',
               'LinearMlip',
               'LammpsMlipNn',
               'LammpsMlip',
               'FitLammpsMlip']
except ImportError:
    __all__ = ['MlipManager',
               'LinearMlip',
               'LammpsMlip',
               'FitLammpsMlip']
