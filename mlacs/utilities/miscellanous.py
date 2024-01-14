"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from scipy import interpolate
from scipy.integrate import simps
from scipy.optimize import minimize

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC


# ========================================================================== #
def get_elements_Z_and_masses(supercell):
    '''
    Get the unique chemical symbols and atomic numbers of a supercell.
    The list are returned according to the alphabetical order of the elements.

    Parameters
    ----------
    supercell: :class:`ase.Atoms`
        ASE atoms object

    Return
    ------
    elements: :class:`list` of :class:`str`
        list of unique elements in the supercell
    Z: :class:`list` of :class:`int`
        list of unique Z in the supercell
    masses: :class:`list` of :class:`float`
        list of unique masses in the supercell
    '''
    elements = supercell.get_chemical_symbols()
    Z = supercell.get_atomic_numbers()
    masses = supercell.get_masses()
    charges = supercell.get_initial_charges()

    un_elements = sorted(set(elements))
    un_Z = []
    un_masses = []
    un_charges = []
    for iel in range(len(un_elements)):
        idx = elements.index(un_elements[iel])
        un_Z.append(Z[idx])
        un_masses.append(masses[idx])
        un_charges.append(charges[idx])

    if np.allclose(un_charges, 0.0, atol=1e-8):
        un_charges = None
    un_elements = np.array(un_elements)
    un_Z = np.array(un_Z)
    return np.array(un_elements), un_Z, un_masses, un_charges


# ========================================================================== #
def create_random_structures(atoms, std, nconfs):
    """
    Create n random structures by displacing atoms around position

    Parameters
    ----------
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled
    std: :class:`float`
        Standard deviation of the gaussian used to generate
        the random displacements. In angstrom.
    nconfs: :class:`int`
        Number of configurations to generate

    Return
    ------
    confs: :class:`list` of :class:`ase.Atoms`
        Configurations with random displacements
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    rng = np.random.default_rng()
    confs = []
    for iat, at in enumerate(atoms):
        for i in range(nconfs):
            iatoms = at.copy()
            iatoms.rattle(stdev=std, rng=rng)
            confs.append(iatoms)
    return confs


# ========================================================================== #
def compute_correlation(data):
    """
    Function to compute the RMSE and MAE

    Parameters
    ----------

    data: :class:`numpy.ndarray` of shape (ndata, 2)
        The data for which to compute the correlation.
        The first column should be the gound truth and the second column
        should be the prediction of the model
    datatype: :class:`str`
        The type of data to which the correlation are to be computed.
        Can be either energy, forces or stress
    """
    datatrue = data[:, 0]
    datatest = data[:, 1]
    rmse = np.sqrt(np.mean((datatrue - datatest)**2))
    mae = np.mean(np.abs(datatrue - datatest))
    sse = ((datatrue - datatest)**2).sum()
    sst = ((datatrue - datatrue.mean())**2).sum()
    rsquared = 1 - sse / sst
    return rmse, mae, rsquared


# ========================================================================== #
def _create_ASE_object(Z, positions, cell, energy):
    """
    Create ASE Atoms object.
    """
    atoms = Atoms(numbers=Z,
                  positions=positions,
                  cell=cell,
                  pbc=True)
    calc = SPC(atoms=atoms,
               energy=energy)
    atoms.set_calculator(calc)
    return atoms


# ========================================================================== #
def compute_averaged(traj):
    """
    Function to compute the averaged Atoms configuration from a Trajectory.

    Parameters
    ----------

    data: :class:`ase.Trajectory`
        List of ase.Atoms object.

    Return
    ------
    confs: :class:`ase.Atoms`
        Averaged configuration
    """
    if isinstance(traj, Atoms):
        traj = [traj]
    Z = traj[-1].get_atomic_numbers()
    cell = np.average([at.get_cell()[:] for at in traj], axis=0)
    positions = np.average([at.get_positions() for at in traj], axis=0)
    energy = np.average([at.get_potential_energy() for at in traj])
    atoms = _create_ASE_object(Z=Z,
                               positions=positions,
                               cell=cell,
                               energy=energy)
    return atoms.copy()


# ========================================================================== #
def compute_volume(confs, weights):
    nconfs = len(confs)
    natoms = len(confs[0])
    vol = []
    cell = []
    if weights is None:
        weights = np.ones(nconfs) / nconfs
    for i, at in enumerate(confs):
        cell.append(at.get_cell() * weights[i])
        vol.append(at.get_volume() * weights[i] / natoms)

    cell = np.sum(cell, axis=0)
    vol = np.sum(vol)

    return cell, vol


# ========================================================================== #
def interpolate_points(x, y, xf, order=0, smooth=0, periodic=0, border=None):
    """
    Interpolate points.

    Parameters
    ----------
    x : :class:`numpy.array`
        List of points to interpolate
    y : :class:`numpy.array`
        List of points to interpolate
    xf : :class:`numpy.array`
        New thiner list of points
    order : :class:`int`
        Order of the spline
    smooth : :class:`int`
        Smoothing parameter
    periodic : :class:`int`
        Activate periodic function boundary conditions
    border : :class:`bol`
        Impose a zero derivative condition at the function boundaries
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled

    Return
    ------
    yf : :class:`list`
        List of interpolated points
    """

    def err(c, x, y, t, k):
        """The error function to minimize"""
        diff = y - interpolate.splev(x, (t, c, k))
        diff = np.einsum('...i,...i', diff, diff)
        return np.abs(diff)

    tck = interpolate.splrep(x, y, s=smooth, per=periodic)
    if border is not None:
        t, c0, k = tck
        x0 = (x[0], x[-1])
        con = {'type': 'eq',
               'fun': lambda c: interpolate.splev(x0, (t, c, k), der=1),
               }
        opt = minimize(err, c0, (x, y, t, k), constraints=con)
        copt = opt.x
        tck = (t, copt, k)

    if isinstance(xf, list):
        yf = [interpolate.splev(_, tck, der=order) for _ in xf]
        return yf
    elif isinstance(xf, np.ndarray):
        yf = [interpolate.splev(_, tck, der=order) for _ in xf]
        return yf
    else:
        return float(interpolate.splev(xf, tck, der=order))


# ========================================================================== #
def integrate_points(x, y, xf, order=0, smooth=0, periodic=0, border=None):
    """
    Interpolate points and return derivatives.

    Parameters
    ----------
    x : :class:`numpy.array`
        List of points to interpolate
    y : :class:`numpy.array`
        List of points to interpolate
    xf : :class:`numpy.array`
        New thiner list of points
    order : :class:`int`
        Order of the spline
    smooth : :class:`int`
        Smoothing parameter
    periodic : :class:`int`
        Activate periodic function boundary conditions
    border : :class:`bol`
        Impose a zero derivative condition at the function boundaries
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled

    Return
    ------
    yf : :class:`list`
        Integral of spline from start to xf
    """

    def err(c, x, y, t, k):
        """The error function to minimize"""
        diff = y - interpolate.splev(x, (t, c, k))
        diff = np.einsum('...i,...i', diff, diff)
        return np.abs(diff)

    def integ(x, tck):
        """Integral of spline from start to x"""
        x = np.atleast_1d(x)
        prim = np.zeros(x.shape, dtype=x.dtype)
        for i in range(len(prim)):
            prim[i] = interpolate.splint(0, x[i], tck)
        return prim

    tck = interpolate.splrep(x, y, s=smooth, per=periodic)
    if border is not None:
        t, c0, k = tck
        x0 = (x[0], x[-1])
        con = {'type': 'eq',
               'fun': lambda c: interpolate.splev(x0, (t, c, k), der=1),
               }
        opt = minimize(err, c0, (x, y, t, k), constraints=con)
        copt = opt.x
        tck = (t, copt, k)
    if isinstance(xf, list):
        yf = integ(xf, tck)
        return yf
    elif isinstance(xf, np.ndarray):
        yf = integ(xf, tck)
        return yf
    else:
        return float(integ(xf, tck))


# ========================================================================== #
def normalized_integration(x, y, norm=1.0, scale=True, func=simps):
    """
    Compute normalized integral of y to `norm`.

    Parameters
    ----------
    x : :class:`numpy.array`
    y : :class:`numpy.array`
    norm : :class:`float`
        Norm of the integral.
    scale : :class:`Bool`
        Scale x and y to the same order of magnitude to avoid numerical
        errors.
    func : :class:`scipy.integrate.func`
        Scipy function for intergration (simps, trapz, ...).

    Return
    ------
    Scaled y
    """
    fx, fy = 1.0, 1.0
    sx, sy = x, y
    if scale:
        fx, fy = np.abs(x).max(), np.abs(y).max()
        sx, sy = x / fx, y / fy
    _norm = func(sy, sx) * fx * fy
    return y * norm / _norm


# ========================================================================== #
def read_distribution_files(filename):
    """
    Function to the distribution files of LAMMPS
    Return the averaged values and the extrem values.
    """
    yaxis, buf = None, None
    with open(filename, 'r') as r:
        for line in r:
            if line[0] == '#':
                continue
            rowdat = line.split()
            if len(rowdat) == 2:
                if yaxis is None and buf is None:
                    yaxis = None
                elif yaxis is None and buf is not None:
                    yaxis = buf
                else:
                    yaxis = np.c_[yaxis, buf]
                buf = np.zeros(int(rowdat[1]))
                xaxis = np.zeros(int(rowdat[1]))
                i = 0
                continue
            xaxis[i] = float(rowdat[1])
            buf[i] = float(rowdat[2])
            i += 1
    _gav = np.average(np.c_[yaxis, buf].T, axis=0)
    _gmin = np.min(np.c_[yaxis, buf].T, axis=0)
    _gmax = np.max(np.c_[yaxis, buf].T, axis=0)
    return xaxis, _gav, _gmin, _gmax
