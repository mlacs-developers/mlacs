import weakref

from ase import units
from ase.parallel import world
from ase.utils import IOContext
from ase.units import kB


class PIMDLogger(IOContext):
    """
    Class for logging path integral molecular dynamics simulations
    """
    def __init__(self, pidyn, qpolymer, logfile, header=True, stress=False, peratom=False, kin="both", mode="a"):
        """
        """
        if hasattr(pidyn, "get_time"):
            self.pidyn = weakref.proxy(pidyn)
        else:
            self.pidyn = None
        self.qpolymer = qpolymer

        self.logfile = self.openfile(logfile, comm=world, mode="a")
        self.stress = stress
        self.peratom = peratom
        self.kin = kin
        global_natoms = qpolymer.natoms
        if self.pidyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            if self.kin == "both":
                self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                     "Ekin/N[eV] (virial)", "Ekin/N[eV] (primitive)", "T[K]")
                self.fmt += "%12.4f %12.4f %12.4f %12.4f %6.1f"
            elif self.kin == "virial":
                self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                     "Ekin/N[eV] (virial)", "T[K]")
                self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
            elif self.kin == "primitive":
                self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                     "Ekin/N[eV] (primitive)", "T[K]")
                self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            if self.kin == "both":
                self.hdr += "%12s %12s %12s  %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                     "Ekin[eV] (virial)", "Ekin[eV] (primitive)", "T[K]")
            if self.kin == "virial":
                self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                  "Ekin[eV] (virial)", "T[K]")
            if self.kin == "primitive":
                self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                  "Ekin[eV] (primitive)", "T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 2 * ("%%12.%df " % (digits,))
            if self.kin == "both":
                self.fmt += 2 * ("%%12.%df " % (digits,))
            else:
                self.fmt += 1 * ("%%12.%df " % (digits,))
            self.fmt += " %6.1f"
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
        if self.kin == "both" or self.kin == "virial":
            ekin_virial = self.qpolymer.get_kinetic_estimator(virial=True)#True)
        if self.kin == "both" or self.kin == "primitive":
            ekin_prim = self.qpolymer.get_kinetic_estimator(virial=False)#True)
        #temp = self.qpolymer.kBT / kB
        temp = self.qpolymer[0].get_temperature()
        #global_natoms = self.qpolymer.get_global_number_of_atoms()
        global_natoms = self.qpolymer.natoms
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.pidyn is not None:
            t = self.pidyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()
        if self.kin == "both" or self.kin == "virial":
            etot = epot + ekin_virial
        else:
            etot = epot + ekin_prim
        dat += (etot, epot)
        if self.kin == "both":
            dat += (ekin_virial, ekin_prim)
        if self.kin == "virial":
            dat += (ekin_virial)
        if self.kin == "primitive":
            dat += (ekin_prim)
        dat += (temp,)
        if self.stress:
            dat += tuple(self.qpolymer.get_stress(
                include_ideal_gas=True) / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()
