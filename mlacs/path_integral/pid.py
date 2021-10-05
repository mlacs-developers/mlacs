

from ase.utils import IOContext


class PathIntegralDynamics(IOContext):
    """
    """
    def __init__(self,  qpolymer, logfile, trajectory, append_trajectory=False, master=None):
        """
        """
        self.qpolymer = qpolymer
        self.logfile = self.openfile(logfile, mode="a", comm=world)
        self.observers = []
        self.nsteps = 0

        # maximum number of steps placeholder with maxint
        self.max_steps = 100000000

        if trajectory is not None:
            if isinstance(trajectory, str):
                mode = "a" if append_trajectory else "w"
                trajectory = self.closelater(Trajectory(
                    trajectory, mode=mode, atoms=atoms, master=master
                ))
            self.attach(trajectory)


    def get_number_of_steps(self):
        return self.nsteps


    def insert_observer(
        self, function, position=0, interval=1, *args, **kwargs
    ):
        """Insert an observer."""
        if not isinstance(function, collections.abc.Callable):
            function = function.write
        self.observers.insert(position, (function, interval, args, kwargs))


    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        If *interval > 0*, at every *interval* steps, call *function* with
        arguments *args* and keyword arguments *kwargs*.

        If *interval <= 0*, after step *interval*, call *function* with
        arguments *args* and keyword arguments *kwargs*.  This is
        currently zero indexed."""

        if hasattr(function, "set_description"):
            d = self.todict()
            d.update(interval=interval)
            function.set_description(d)
        if not hasattr(function, "__call__"):
            function = function.write
        self.observers.append((function, interval, args, kwargs))


    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            call = False
            # Call every interval iterations
            if interval > 0:
                if (self.nsteps % interval) == 0:
                    call = True
            # Call only on iteration interval
            elif interval <= 0:
                if self.nsteps == abs(interval):
                    call = True
            if call:
                function(*args, **kwargs)


    def irun(self):
        """Run dynamics algorithm as generator. This allows, e.g.,
        to easily run two optimizers or MD thermostats at the same time.

        Examples:
        >>> opt1 = BFGS(atoms)
        >>> opt2 = BFGS(StrainFilter(atoms)).irun()
        >>> for _ in opt2:
        >>>     opt1.run()
        """

        # compute initial structure and log the first step
        self.qpolymer.get_forces()

        # yield the first time to inspect before logging
        yield False

        if self.nsteps == 0:
            self.log()
            self.call_observers()

        # run the algorithm until converged or max_steps reached
        while not self.converged() and self.nsteps < self.max_steps:

            # compute the next step
            self.step()
            self.nsteps += 1

            # let the user inspect the step and change things before logging
            # and predicting the next step
            yield False

            # log the step
            self.log()
            self.call_observers()

        # finally check if algorithm was converged
        yield self.converged()


    def run(self):
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        for converged in Dynamics.irun(self):
            pass
        return converged


    def converged(self, *args):
        """" a dummy function as placeholder for a real criterion, e.g. in
        Optimizer """
        return False


    def log(self, *args):
        """ a dummy function as placeholder for a real logger, e.g. in
        Optimizer """
        return True


    def step(self):
        """this needs to be implemented by subclasses"""
        raise RuntimeError("step not implemented.")
