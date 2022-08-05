from ase.io import Trajectory


# ========================================================================== #
# ========================================================================== #
def PathIntegralTrajectory(prefix,
                           mode="r",
                           qpolymer=None,
                           properties=None,
                           master=None,
                           nbeads=None):
    """
    """
    if mode == "r":
        return PathIntegralTrajectoryReader(prefix, nbeads)
    return PathIntegralTrajectoryWriter(prefix,
                                        mode,
                                        qpolymer,
                                        properties,
                                        master=master,
                                        nbeads=nbeads)


# ========================================================================== #
class PathIntegralTrajectoryWriter:
    """
    """
    def __init__(self, prefix, mode="w", qpolymer=None, properties=None,
                 extra=[], master=None, nbeads=None):
        """
        """
        self.qpolmer = qpolymer
        self.nbeads = nbeads
        if qpolymer is not None:
            self.nbeads = len(qpolymer)
        if self.nbeads is None:
            raise ValueError

        self._traj = []
        for ibead in range(self.nbeads):
            if qpolymer is None:
                self._traj.append(Trajectory(prefix + f"_{ibead}.traj",
                                             "w",
                                             None,
                                             properties,
                                             master=master))
            else:
                self._traj.append(Trajectory(prefix + f"_{ibead}.traj",
                                             "w",
                                             qpolymer[ibead],
                                             properties,
                                             master=master))

# ========================================================================== #
    def write(self, qpolymer=None, **kwargs):
        """
        """
        qpolymer = qpolymer
        for ibead in range(self.nbeads):
            self[ibead].write(qpolymer[ibead], **kwargs)

# ========================================================================== #
    def __len__(self):
        return self.nbeads

# ========================================================================== #
    def __getitem__(self, i):
        return self._traj[i]

# ========================================================================== #
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ========================================================================== #
class PathIntegralTrajectoryReader:
    """
    """
    def __init__(self, prefix, nbeads):
        self._traj = []
        self.nbeads = nbeads
        for ibead in range(self.nbeads):
            self._traj.append(Trajectory(prefix+f"_{ibead}.traj", "r"))

# ========================================================================== #
    def __len__(self):
        return self.nbeads

# ========================================================================== #
    def __getitem__(self, i):
        return self._traj[i]

# ========================================================================== #
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
