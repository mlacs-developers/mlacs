

import weakref

from ase import units
from ase.parallel import world
from ase.utils import IOContext


class PIMDLogger(IOContext):
    """
    Class for logging path integral molecular dynamics simulations
    """
    def __init__(self, pidyn, qpolymer, logfile, header=True, stress=False, mode="a"):
        """
        """
        if hasattr(pidyn, "get_time"):
            self.pidyn = weakref.proxy(pidyn)
        else:
            self.pidyn = None
        self.qpolymer = qpolymer

        self.logfile = self.openfile(logfile, comm=world, mode="a")
        self.stress = stress
        if self.pidyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                 "Ekin/N[eV]", "T[K]")
            self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                 "Ekin[eV]", "T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 3 * ("%%12.%df " % (digits,)) + " %6.1f"
        if self.stress:
            self.hdr += ('      ---------------------- stress [GPa] '
                         '-----------------------')
            self.fmt += 6 * " %10.3f"
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        epot = self.qpolymer.get_potential_estimator()
        ekin = self.qpolymer.get_kinetic_estimator(virial=True)
        temp = self.qpolymer.kBT
        global_natoms = self.qpolymer.get_global_number_of_atoms()
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (epot + ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.qpolymer.get_stress(
                include_ideal_gas=True) / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()
