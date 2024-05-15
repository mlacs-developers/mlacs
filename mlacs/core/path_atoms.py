import numpy as np

from scipy.spatial import distance

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC

from ..utilities import interpolate_points as intpts


class PathAtoms:
    """
    Base class for managing for transiton state.
    """
    def __init__(self,
                 images,
                 xi=None,
                 mode="saddle",
                 fixcom=True,
                 **kwargs):

        self.mode = mode
        self.images = images
        if len(images) < 2:
            msg = 'You need at least a starting and a final configuration '
            msg += 'to define a Transition Path !'
            raise TypeError(msg)
        self.fixcom = fixcom

        self._xi = xi
        self._memxi = None
        self._splR = None
        self._splined = None
        if self._xi is None:
            self._splprec = 1001
            self._memxi = [0.0, 1.0]

    @property
    def images(self):
        """Get true images"""
        return self._images

    @images.setter
    def images(self, img):
        self._images = img

    @property
    def imgxi(self):
        return np.linspace(0, 1, len(self.images))

    @property
    def imgR(self):
        return np.array([a.positions for a in self.images])

    @property
    def imgE(self):
        return np.array([a.get_potential_energy() for a in self.images])

    @property
    def imgC(self):
        return np.array([a.cell.array for a in self.images])

    @property
    def masses(self):
        """Get the effective masses of particles"""
        pos = np.transpose(self.imgR, (1, 0, 2))
        meff = np.array([np.max(distance.cdist(d, d, "euclidean"))
                         for d in pos])
        return meff / np.max(meff)

    @property
    def xi(self):
        """
        Get the reaction coordinate.
        """
        if self._xi is not None:
            return self._xi

        def find_dist(_l):
            m = []
            _l.sort()
            for i, val in enumerate(_l[1:]):
                m.append(np.abs(_l[i+1] - _l[i]))
            i = np.array(m).argmax()
            return _l[i+1], _l[i]

        mode = self.mode
        if isinstance(mode, float):
            return mode
        elif mode == 'saddle':
            x = np.linspace(0, 1, self._splprec)
            y = intpts(self.imgxi, self.imgE, x, 0, border=1)
            y = np.array(y)
            xi = x[y.argmax()]
            return xi
        elif mode == 'rdm_spl':
            xi = np.random.uniform(0, 1)
            return xi
        elif mode == 'rdm_memory':
            x, y = find_dist(self.finder)
            xi = np.random.uniform(x, y)
            self._memxi.append(xi)
            return xi
        elif mode == 'rdm_true':
            r = np.random.default_rng()
            nrep = len(self.images)
            xi = r.integers(nrep) / nrep
            return xi
        else:
            msg = 'The reaction coordinate is not defined.'
            msg += 'You need to define `xi` or the `mode` to find it !'
            raise TypeError(msg)

    @property
    def splined(self):
        """Get splined images"""
        return self._splined

    @splined.setter
    def splined(self, xi):
        self._xi = xi or None
        self._splined = self.get_splined_atoms()

    @property
    def splR(self):
        if self._splR is None:
            self.set_splined_matrices()
        return self._splR[0:3]

    @property
    def splDR(self):
        if self._splR is None:
            self.set_splined_matrices()
        if self.fixcom:
            return self._splR[9:12]
        return self._splR[3:6]

    @property
    def splD2R(self):
        if self._splR is None:
            self.set_splined_matrices()
        if self.fixcom:
            return self._splR[12:15]
        return self._splR[6:9]

    @property
    def splE(self):
        return intpts(self.imgxi, self.imgE, self.xi, 0, border=1)

    @property
    def splC(self):
        return intpts(self.imgxi, self.imgC, self.xi, 0, border=1)

    def get_splined_atoms(self, xi=None):
        """
        Return splined Atoms objects at the xi coordinates.
        """
        if xi is None:
            xi = self.xi

        if isinstance(xi, float):
            xi = [xi]

        splatoms = []
        Z = self.images[0].get_atomic_numbers()
        for x in xi:
            self.set_splined_matrices(x)
            at = Atoms(numbers=Z, positions=self.splR, cell=self.splC)
            calc = SPC(atoms=at, energy=self.splE)
            at.calc = calc
            at.set_array('first_derivatives', self.splDR)
            at.set_array('second_derivatives', self.splD2R)
            at.info['reaction_coordinate'] = x
            splatoms.append(at)
        return splatoms

    def set_splined_matrices(self, xi=None):
        """
        Compute a 1D CubicSpline interpolation from a NEB calculation.
        The function also set up the lammps data file for a constrained MD.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at
                reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at
                reaction coordinate xi.
        """
        if xi is None:
            xi = self.xi

        N = len(self.images[0])
        if isinstance(xi, float):
            _nim = 1
            self._splR = np.zeros((N, 15))
        else:
            _nim = len(xi)
            self._splR = np.zeros((N, 15, _nim))

        # Spline interpolation of the referent path and calculation of
        # the path tangent and path tangent derivate
        for i in range(N):
            self._splR[i, 0:3] = np.r_[[intpts(self.imgxi, self.imgR[:, i, j],
                                        xi, 0) for j in range(3)]]
            self._splR[i, 3:6] = np.r_[[intpts(self.imgxi, self.imgR[:, i, j],
                                        xi, 1) for j in range(3)]]
            self._splR[i, 6:9] = np.r_[[intpts(self.imgxi, self.imgR[:, i, j],
                                        xi, 2) for j in range(3)]]

        if self.fixcom:
            self._com_corrections()

    def _com_corrections(self):
        """
        Correction of the path tangent to have zero center of mass
        """
        N = len(self.images[0])
        n, d, r = self._splR.shape
        spl = np.zeros((n, 6, r))
        pos, der, der2 = np.hsplit(self._splR, 3)
        com = np.array([np.average(der[:, i]) for i in range(3)])
        n = 0
        for i in range(N):
            n += np.sum([(der[i, j] - com[j])**2 for j in range(3)])
        n = np.sqrt(n)
        for i in range(N):
            spl[i, 9:12] = np.r_[[(der[i, j] - com[j]) / n for j in range(3)]]
            spl[i, 12:15] = np.r_[[der2[i, j] / n / n for j in range(3)]]
        return spl
