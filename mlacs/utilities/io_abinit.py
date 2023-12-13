"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import numpy as np

from ase import Atoms
from ase.units import Bohr, Hartree
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc

try:
    import netCDF4 as nc
except ImportError:
    nc = None


def set_aseAtoms(results=dict()) -> Atoms:
    """Read results dictionary to construct Ase Atoms object"""
    if 'reduced_atom_positions' in results.keys():
        return _set_from_gsrNC(results)
    elif 'xred' in results.keys():
        return _set_from_outNC(results)
    else:
        msg = 'No atomic positions found in dictionary'
        raise AttributeError(msg)


def _set_from_outNC(results=dict()) -> Atoms:
    """Read results from OUT.nc"""
    Z = np.r_[[results['znucl'][i - 1] for i in results['typat']]]
    nat = len(results['typat'])
    cell = results['acell'] * np.reshape(results['rprim'], (3, 3)) * Bohr
    atoms = Atoms(numbers=Z,
                  cell=cell,
                  positions=np.reshape(results['xangst'], (nat, 3)),
                  pbc=True)
    calc = SPCalc(atoms,
                  energy=results['etotal'] * Hartree,
                  forces=results['fcart'] * Hartree / Bohr,
                  stress=results['strten'] * Hartree / Bohr**3)
    atoms.calc = calc
    return atoms


def _set_from_gsrNC(results=dict()) -> Atoms:
    """Read results from GSR.nc"""
    Z = np.r_[[results['atomic_numbers'][i - 1]
               for i in results['atom_species']]]
    cell = results['primitive_vectors'] * Bohr
    positions = np.matmul(results['reduced_atom_positions'], cell)
    atoms = Atoms(numbers=Z,
                  cell=cell,
                  positions=positions,
                  pbc=True)
    calc = SPCalc(atoms,
                  energy=results['etotal'] * Hartree,
                  forces=results['cartesian_forces'] * Hartree/Bohr,
                  stress=results['cartesian_stress_tensor'] * Hartree/Bohr**3,
                  free_energy=results['entropy'] * Hartree)
    atoms.calc = calc
    return atoms


# ========================================================================== #
# ========================================================================== #
class AbinitNC:
    """
    Class to handle all Abinit NetCDF files.

    Parameters
    ----------
    workdir: :class:`str` (optional)
        The root for the directory in which the computation are to be done
        Default 'DFT'
    """
    def __init__(self, workdir=None, prefix='abinit'):

        if nc is None:
            raise ImportError("You need Netcdf4 to use the AbinitNC class")

        self.workdir = workdir
        if self.workdir is None:
            self.workdir = os.getcwd() + "/DFT/"
        if self.workdir[-1] != "/":
            self.workdir[-1] += "/"
        if not os.path.exists(self.workdir):
            self.workdir = ''

        self.ncfile = self.workdir + f'{prefix}o_GSR.nc'
        self.results = {}

# ========================================================================== #
    def read(self, filename=None):
        """Read NetCDF output of Abinit"""

        if filename is not None:
            self.dataset = nc.Dataset(filename)
        elif filename is None and hasattr(self, 'ncfile'):
            self.dataset = nc.Dataset(self.ncfile)
        else:
            raise FileNotFoundError('No NetCDF file defined')

        self._keyvar = [_ for _ in self.dataset.variables]
        self._defattr()
        if not hasattr(self, 'results'):
            self.results = {}
        self.results.update(vars(self))
        return self.results

# ========================================================================== #
    def ncdump(self, filename=None) -> str:
        """Read NetCDF output of Abinit"""
        from subprocess import check_output
        return check_output(['ncdump', filename])

# ========================================================================== #
    def _defattr(self):
        for attr in self._keyvar:
            setattr(self, attr, self._extractattr(attr))
        for attr in self._keyvar:
            value = getattr(self, attr)
            if isinstance(value, (int, float, str)):
                continue
            elif isinstance(value, np.ndarray):
                setattr(self, attr, self._decodearray(value))
            elif isinstance(value, memoryview):
                if () == value.shape:
                    setattr(self, attr, value.tolist())
                else:
                    setattr(self, attr, self._decodearray(value.obj))
            else:
                delattr(self, attr)
                msg = f'Unknown object type: {type(value)}\n'
                msg += '-> deleted attribute from AbiAtoms object.\n'
                msg += 'Should be added in the class AbiAtoms, if needed !'
                raise Warning(msg)

# ========================================================================== #
    def _extractattr(self, value):
        return self.dataset[value][:].data

# ========================================================================== #
    def _decodeattr(self, value) -> str:
        # Weird thing to check the end of the file, don't ask ...
        _chk = ''.join([' ' for i in range(80)])
        _str = ''
        for s in value.tolist():
            if _chk == _str[-80:]:
                break
            _str += bytes.decode(s)
        return _str.strip()

# ========================================================================== #
    def _decodearray(self, value):
        if 'S1' != value.dtype:
            return value
        elif 1 == len(value.shape):
            return self._decodeattr(value)
        else:
            return np.r_[[self._decodeattr(v) for v in value]]
