"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from ase.neighborlist import NeighborList
from ase.units import kB


# ========================================================================== #
def compute_pdf(configurations,
                temperature=None,
                weights=None,
                fname=None,
                **kwargs):
    '''
    Function to compute the pdf

    Parameters
    ----------

    configurations: list of ase atoms object

    Optional
    --------
    temperature: float
       The temperature of the simulation, needed for the force sampling method
    weights: list of float
       The weights of the configurations, should sum up to one
    rcut: float
       The cutoff for the distances, default to 5.0 angstrom
    method: str
       Method to compute the pdf.
       Available options are 'count' and 'force_sampling'
    nbins: int
       Number of distance bins
    '''
    # Initialize some variables and inputs
    rcut = kwargs.get('rcut', 5.0)
    method = kwargs.get('method', 'force_sampling')
    nbins = kwargs.get('nbins', 250)

    nconfs = len(configurations)
    natoms = len(configurations[0])
    volume = configurations[0].get_volume()

    # To avoid complication, we add normalized weights if not given
    if weights is None:
        weights = np.ones(nconfs) / nconfs

    if method == 'force_sampling':
        if temperature is None:
            msg = 'The temperature is needed to compute the pdf ' + \
                  'with the force sampling method'
            raise ValueError(msg)

    elements = np.array(sorted(set(configurations[0].get_chemical_symbols())))
    nelements = len(elements)
    nint = int(nelements * (nelements - 1) / 2) + nelements

    # Create a matrix to place the interactions in the right place
    R, C = np.triu_indices(nelements)
    idx_int = np.zeros((nelements, nelements), dtype=int)
    idx_int[R, C] = np.arange(nint)
    idx_int[C, R] = np.arange(nint)

    # Initialize the histogram
    count, edges = np.histogram([1.0], bins=nbins, range=(0.0, rcut))
    count = np.zeros((nint, count.shape[0]))

    points = edges[:-1] - (edges[:-1] - edges[1:]) * 0.5

    # Prepare the neighbor list
    nl = NeighborList([rcut / 2.] * natoms, self_interaction=False)
    # Loop on configurations
    for iconf in range(nconfs):
        if np.abs(weights[iconf]) >= 1e-8:
            atoms = configurations[iconf]
            nl.update(atoms)

            # We need the chemical symbols to be able to compute a
            # per-interaction g(R)
            symbols = np.array(atoms.get_chemical_symbols())

            if method == 'force_sampling':
                forces = atoms.get_forces()

            # Loop on atoms
            for iat in range(natoms):
                jat_idx, offsets = nl.get_neighbors(iat)

                # Compute the distances between atom i and its neighbor
                rij = atoms.positions[jat_idx] + \
                    np.dot(offsets, atoms.cell) - atoms.positions[iat]
                dij = np.sqrt(np.sum(rij**2, axis=1))

                iat_el = np.where(elements == symbols[iat])[0][0]

                count_iat = np.zeros_like(count)
                # If counting method, we use the histogram function of numpy
                if method == 'count':
                    for el in range(nelements):
                        el_idx = np.where(symbols[jat_idx] == elements[el])[0]
                        count_iat_tmp, edges_iat = np.histogram(
                                                      dij[el_idx],
                                                      bins=nbins,
                                                      range=(0.0, rcut))
                        inte = idx_int[iat_el, el]
                        count_iat[inte] = count_iat_tmp

                if method == 'force_sampling':
                    f_jat = forces[jat_idx]
                    tmp = -0.5 * (forces[iat] - f_jat) * \
                        rij / (dij**3).repeat(3).reshape(-1, 3)
                    # Loop on the pdf distances
                    for n in range(len(points)):
                        for el in range(nelements):
                            # We need the interactions between the elements
                            # of iat and element el
                            tmp_el_idx = np.where(
                                symbols[jat_idx] == elements[el])[0]
                            tmp_el = tmp[tmp_el_idx]
                            # inte is the index allowing to place
                            # the interaction the right place
                            inte = idx_int[iat_el, el]
                            # We extract only the distances for
                            # this type of interaction
                            dij_el = dij[tmp_el_idx]
                            # In the force-sampling formulation,
                            # all atoms within a distance inferior
                            # to R contributes
                            count_iat[inte, n] = np.sum(
                                tmp_el[dij_el < points[n]])

                # Finally we weight the contribution
                # The 2 is to account for the bothways=False in the neigh-list
                count += 2 * count_iat * weights[iconf]

    # For the multielement case, we compute the total g(R) here
    if nelements > 1:
        count = np.vstack((count, np.sum(count, axis=0)))

    # Compute the normalization factor
    norm = volume / (4 * np.pi)
    for i in range(nint):
        idx = np.where(i == idx_int)[0]
        if len(idx) > 1:
            el1 = np.array(elements[idx[0]])
            nat1 = len(np.where(el1 == atoms.get_chemical_symbols())[0])
            el2 = np.array(elements[idx[1]])
            nat2 = len(np.where(el2 == atoms.get_chemical_symbols())[0])
        else:
            el1 = np.array(elements[idx[0]])
            nat1 = len(np.where(el1 == atoms.get_chemical_symbols())[0])
            nat2 = nat1

        if method == 'count':
            count[i] *= norm / (points**2 * nat1 * nat2)
        if method == 'force_sampling':
            count[i] *= norm / (kB * temperature * nat1 * nat2)

    if nelements > 1:
        if method == 'count':
            count[-1] *= norm / (points**2 * natoms**2)
        if method == 'force_sampling':
            count[-1] *= norm / (kB * temperature * natoms**2)

    """
    # Normalize the pdf
    if method == 'count':
        count  *= norm / (points**2) * natoms
    elif method == 'force_sampling':
        count  *= norm / (kB * temperature)

    # For the multielement case, we compute the total g(R) here
    if nelements > 1:
        count = np.vstack((count, np.sum(count,axis=0)))
    """

    pdf = np.vstack((points, count)).T

    # If asked, we write the pdf
    if fname is not None:
        header = "Distances            "
        for i in range(nint):
            idx = np.where(i == idx_int)[0]
            if len(idx) > 1:
                header += f"{elements[idx[0]]}-{elements[idx[1]]}             "
            else:
                header += f"{elements[idx[0]]}-{elements[idx[0]]}             "
        if nelements > 1:
            header += "tot"
        np.savetxt(fname, pdf, header=header, fmt="%20.16f")
    return pdf
