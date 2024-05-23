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
                 interval=None,
                 **kwargs):

        self.mode = mode
        self.images = images
        if len(images) < 2:
            msg = 'You need at least a starting and a final configuration '
            msg += 'to define a Transition Path !'
            raise TypeError(msg)
        self.fixcom = fixcom

        self._xi = xi
        self._tmp_xi = xi
        self._memxi = interval
        self._splR = None
        self._splined = None
        if self._xi is None:
            self._splprec = 1001
        if self._memxi is None:
            self._memxi = [0.0, 1.0]

    @property
    def images(self):
        """Get true images"""
        return self._images

    @property
    def initial(self):
        """Get first image"""
        return self._images[0]

    @property
    def final(self):
        """Get last image"""
        return self._images[-1]

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
    def nreplica(self):
        return len(self.images)

    @property
    def masses(self):
        """Get the effective masses of particles"""
        pos = np.transpose(self.imgR, (1, 0, 2))
        meff = np.array([np.max(distance.cdist(d, d, "euclidean"))
                         for d in pos])
        return meff / np.max(meff)

    @property
    def update(self):
        """
        Update the reaction coordinate.
        """
        if self._xi is not None:
            self._tmp_xi = self._xi
            return self._xi

        def find_dist(_l):
            m = []
            _l.sort()
            for i, val in enumerate(_l[1:]):
                m.append(np.abs(_l[i+1] - _l[i]))
            i = np.array(m).argmax()
            return _l[i+1], _l[i]

        mode = self.mode
        if isinstance(mode, (float, list, np.ndarray)):
            self._tmp_xi = mode
            return mode
        elif mode == 'saddle':
            x = np.linspace(0, 1, self._splprec)
            # RB : The derivatives at xi=0 or 1 is not always null.
            # y = np.r_[intpts(self.imgxi, self.imgE, x, 0, border=1)]
            y = np.r_[intpts(self.imgxi, self.imgE, x, 0)]
            self._tmp_xi = x[y.argmax()]
            return self._tmp_xi
        elif mode == 'rdm':
            i, f = tuple(self._memxi)
            self._tmp_xi = np.random.uniform(i, f)
            return self._tmp_xi
        elif mode == 'rdm_memory':
            i, f = find_dist(self._memxi)
            self._tmp_xi = np.random.uniform(i, f)
            self._memxi.append(self._tmp_xi)
            return self._tmp_xi
        elif mode == 'rdm_true':
            def cond(x, i, f): return x >= i and x <= f
            i, f = tuple(self._memxi)
            xiint = list(filter(lambda x: cond(x, i, f), self.imgxi))
            self._tmp_xi = np.random.choice(xiint)
            return self._tmp_xi
        else:
            msg = 'The reaction coordinate is not defined.'
            msg += 'You need to define `xi` or the `mode` to find it !'
            raise TypeError(msg)

    @property
    def xi(self):
        """
        Get the reaction coordinate.
        """
        if self._xi is not None:
            return self._xi
        elif self._tmp_xi is None:
            return self.update
        else:
            return self._tmp_xi

    @xi.setter
    def xi(self, xi):
        self._xi = xi

    @property
    def splined(self):
        """Get splined images"""
        self._splined = self.get_splined_atoms(self.xi)
        return self._splined

    @property
    def splR(self):
        self.set_splined_matrices(self.xi)
        return self._splR[:, 0:3]

    @property
    def splDR(self):
        self.set_splined_matrices(self.xi)
        if self.fixcom:
            return self._splR[:, 9:12]
        return self._splR[:, 3:6]

    @property
    def splD2R(self):
        self.set_splined_matrices(self.xi)
        if self.fixcom:
            return self._splR[:, 12:15]
        return self._splR[:, 6:9]

    @property
    def splE(self):
        # RB : The derivatives at xi=0 or 1 is not always null.
        # return np.r_[intpts(self.imgxi, self.imgE, self.xi, 0, border=1)]
        return np.r_[intpts(self.imgxi, self.imgE, self.xi, 0)]

    @property
    def splC(self):
        _imgC = self.imgC.reshape((self.nreplica, 9)).T
        if isinstance(self.xi, (float, int)):
            shape = (3, 3)
            _splC = np.zeros(9)
        else:
            shape = (len(self.xi), 3, 3)
            _splC = np.zeros((9, len(self.xi)))
        for i in range(9):
            _splC[i] = intpts(self.imgxi, _imgC[i], self.xi, 0, border=1)
        return _splC.T.reshape(shape)

    def get_splined_atoms(self, xi=None):
        """
        Return splined Atoms objects at the xi coordinates.
        """
        if xi is None:
            xi = self.update

        def set_atoms(X, C, M, R, DR, D2R):
            Z = self.images[0].get_atomic_numbers()
            at = Atoms(numbers=Z, positions=R, cell=C)
            at.set_pbc(True)
            at.set_array('first_derivatives', DR)
            at.set_array('second_derivatives', D2R)
            at.set_array('effective_masses', M)
            at.info['reaction_coordinate'] = X
            return at

        if isinstance(xi, float):
            splat = set_atoms(xi, self.splC, self.masses,
                              self.splR, self.splDR, self.splD2R)
            calc = SPC(atoms=splat, energy=self.splE)
            splat.calc = calc
            return splat

        splat = []
        for i, (x, c, e) in enumerate(zip(xi, self.splC, self.splE)):
            at = set_atoms(x, c, self.masses, self.splR[:, :, i],
                           self.splDR[:, :, i], self.splD2R[:, :, i])
            calc = SPC(atoms=at, energy=e)
            at.calc = calc
            splat.append(at)
        return splat

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
            xi = self.update

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
        return self._splR.round(8)

    def _com_corrections(self):
        """
        Correction of the path tangent to have zero center of mass
        """
        N = len(self.images[0])
        der = self._splR[:, 3:6]
        der2 = self._splR[:, 6:9]
        com = np.array([np.average(der[:, i]) for i in range(3)])
        n = 0
        for i in range(N):
            n += np.sum([(der[i, j] - com[j])**2 for j in range(3)])
        n = np.sqrt(n)
        for i in range(N):
            self._splR[i, 9:12] = np.r_[[(der[i, j] - com[j]) / n
                                        for j in range(3)]]
            self._splR[i, 12:15] = np.r_[[der2[i, j] / n / n
                                         for j in range(3)]]
