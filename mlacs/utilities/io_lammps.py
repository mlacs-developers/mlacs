from pathlib import Path

import numpy as np
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammps import Prism, convert

from .path_integral import hbar


class LammpsInput:
    """

    """
    def __init__(self, preambule=None):
        if preambule is not None:
            self.preambule = f"# {preambule}\n\n"
        else:
            self.preambule = ""
        self.nvar = 0
        self.vardict = dict()

    def add_block(self, name, block, order=-1, before=None, after=None):
        """

        """
        if before is not None and after is not None:
            msg = "before and after can't be both set"
            raise ValueError(msg)

        if before is not None:
            order = self.vardict[before]["order"]
        elif after is not None:
            order = self.vardict[after]["order"] + 1

        if order < 0:
            order = self.nvar + 1
        else:
            keys = []
            values = []
            for key, val in self.vardict.items():
                keys.append(key)
                values.append(val["order"])
            keys = np.array(keys)
            values = np.array(values)
            argsort = np.argsort(values)
            values = values[argsort]
            keys = keys[argsort]
            if order > np.max(values) or order not in values:
                self.vardict["name"]["order"] = order
            elif order in values:
                values[values >= order] += 1
                for i, (key, val) in enumerate(zip(keys, values)):
                    self.vardict[key]["order"] = values[i]
        self.vardict[name] = dict(order=order, block=block)
        self.nvar += 1

    def to_string(self):
        """

        """
        keys = []
        orders = []
        blocks = []
        for key, val in self.vardict.items():
            keys.append(key)
            orders.append(val["order"])
            blocks.append(val["block"])

        keys = np.array(keys)
        orders = np.array(orders)
        blocks = np.array(blocks)

        argsort = np.argsort(orders)
        blocks = blocks[argsort]

        txt = self.preambule
        txt += "\n\n".join(str(block) for block in blocks)
        return txt

    def pop(self, name):
        return self.vardict.pop(name)

    def __str__(self):
        return self.to_string()

    def __call__(self, name, block, order=-1):
        self.add_block(name, block, order)


class LammpsBlockInput:
    """

    """
    def __init__(self, name, title=None):
        self.name = name
        self.vardict = dict()
        self.nvar = 0
        if title is not None:
            title = title.strip()
            nchar = len(title)
            self.title = "#" * (12 + nchar) + "\n"
            self.title += title.center(nchar + 10, " ").center(nchar + 12, "#")
            self.title += "\n"
            self.title += "#" * (12 + nchar) + "\n"
        else:
            self.title = "\n"

    def add_variable(self, name, line, order=-1, before=None, after=None):
        """

        """
        if before is not None and after is not None:
            msg = "before and after can't be both set"
            raise ValueError(msg)

        if before is not None:
            order = self.vardict[before]["order"]
        elif after is not None:
            order = self.vardict[after]["order"] + 1

        if order < 0:
            order = self.nvar + 1
        else:
            keys = []
            values = []
            for key, val in self.vardict.items():
                keys.append(key)
                values.append(val["order"])
            keys = np.array(keys)
            values = np.array(values)
            argsort = np.argsort(values)
            values = values[argsort]
            keys = keys[argsort]
            if order in values:
                values[values >= order] += 1
                for i, (key, val) in enumerate(zip(keys, values)):
                    self.vardict[key]["order"] = values[i]
        self.vardict[name] = dict(order=order, line=line)
        self.nvar += 1

    def to_string(self):
        """

        """
        keys = []
        orders = []
        lines = []
        for key, val in self.vardict.items():
            keys.append(key)
            orders.append(val["order"])
            lines.append(val["line"])

        keys = np.array(keys)
        orders = np.array(orders)
        lines = np.array(lines)

        argsort = np.argsort(orders)
        line = lines[argsort]

        txt = self.title
        txt += "\n".join(line)
        return txt

    def pop(self, name):
        return self.vardict.pop(name)

    def extend(self, block):
        for key, val in block.vardict.items():
            self.__call__(key, val['line'])

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"LammbsBlockInput({self.name})"

    def __call__(self, name, line, order=-1):
        self.add_variable(name, line, order)


class EmptyLammpsBlockInput(LammpsBlockInput):
    """

    """
    def __init__(self, name):
        self.name = name

    def to_string(self):
        return ""


# ========================================================================== #
def get_block_rdf(nsteps, filename='spce-rdf.dat', rmax=None):
    """
    Function to compute and output the radial distribution function
    """
    # freq = int(nsteps/5)
    block = LammpsBlockInput("RDF", "Compute RDF")
    block("v_rep", f"variable repeat equal {nsteps}/2")
    txt = "compute rdf all rdf ${repeat} 1 1"
    if rmax is not None:
        txt += ' cutoff {rmax}'
    block("c_rdf", txt)
    txt = "fix rdf all ave/time 1 ${repeat}" + \
          f" {nsteps} c_rdf[*] file {filename} mode vector\n"
    block("rdf", txt)
    return block


# ========================================================================== #
def get_block_adf(nsteps, filename='spce-adf.dat'):
    """
    Function to compute and output the angle distribution function
    """
    # freq = int(nsteps/5)
    block = LammpsBlockInput("ADF", "Compute ADF")
    block("v_rep", "variable repeat equal 1")
    block("c_adf", "compute adf all adf 360")
    txt = "fix adf all ave/time 100 ${repeat}" + \
          f" {nsteps} c_adf[*] file {filename} mode vector\n"
    block("adf", txt)
    return block


# ========================================================================== #
def get_block_diffusion(nsteps, filename='diffusion.dat'):
    """
    Function to compute and output the diffusion coefficient
    """
    # freq = int(nsteps/5)
    block = LammpsBlockInput("MSD", "Compute MSD and diffusion coefficient")
    block("v_t", "variable t equal step")
    block("c_msd", "compute msd all msd")
    block("v_msd", "variable msd equal c_msd[4]")
    block("v_twopts", "variable twopoint equal c_msd[4]/6/(step*dt+1.0e-6)")
    block("f_msd", "fix msd all vector 1000 c_msd[4]")
    block("v_slope", "variable fislope equal slope(f_msd)/6/(10000*dt)")
    txt = 'fix dcoeff all print 100 "${t} ${msd} ${twopoint} ${fitslope}"' + \
          f' append {filename} title "# Step MSD D(start) D(slope)"'
    block("diffusion", txt)
    return block


# ========================================================================== #
def get_log_input(loginterval, logfile):
    """
    Function to write the log of the mlmd run
    """
    input_string = "#####################################\n"
    input_string += "#          Logging\n"
    input_string += "#####################################\n"
    input_string += "variable    t equal step\n"
    input_string += "variable    mytemp equal temp\n"
    input_string += "variable    mype equal pe\n"
    input_string += "variable    myke equal ke\n"
    input_string += "variable    myetot equal etotal\n"
    input_string += "variable    mypress equal press/10000\n"
    input_string += "variable    mylx  equal lx\n"
    input_string += "variable    myly  equal ly\n"
    input_string += "variable    mylz  equal lz\n"
    input_string += "variable    vol   equal (lx*ly*lz)\n"
    input_string += "variable    mypxx equal pxx/(vol*10000)\n"
    input_string += "variable    mypyy equal pyy/(vol*10000)\n"
    input_string += "variable    mypzz equal pzz/(vol*10000)\n"
    input_string += "variable    mypxy equal pxy/(vol*10000)\n"
    input_string += "variable    mypxz equal pxz/(vol*10000)\n"
    input_string += "variable    mypyz equal pyz/(vol*10000)\n"
    input_string += f'fix mythermofile all print {loginterval} ' + \
                    '"$t ${vol} ${myetot}  ${mype} ${myke} ' + \
                    '${mytemp}  ${mypress} ${mypxx} ${mypyy} ' + \
                    '${mypzz} ${mypxy} ${mypxz} ${mypyz}" ' + \
                    f'append {logfile} title "# Step  Vol  Etot  ' + \
                    'Epot  Ekin  Temp Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_pafi_log_input(rep=0,
                       isappend=False):
    """
    Function to write several PAFI outputs
    """
    input_string = "#####################################\n"
    input_string += "#          Logging\n"
    input_string += "#####################################\n"
    input_string += "variable    dU    equal f_pafihp[1]\n"
    input_string += "variable    dUerr equal f_pafihp[2]\n"
    input_string += "variable    psi   equal f_pafihp[3]\n"
    input_string += "variable    err   equal f_pafihp[4]\n"
    input_string += "compute     disp    all displace/atom\n"
    input_string += "compute     maxdisp all reduce max c_disp[4]\n"
    input_string += "variable    maxjump equal sqrt(c_maxdisp)\n"

    if isappend:
        input_string += 'fix logpafi all print 1 ' + \
                        '"${dU}  ${dUerr} ${psi} ${err} ${maxjump}" ' + \
                        f'append pafi.log.{rep} title ' + \
                        '"# dU/dxi  (dU/dxi)^2  psi  err  maxjump"\n'
    else:
        input_string += 'fix logpafi all print 1 ' + \
                        '"${dU}  ${dUerr} ${psi} ${err} ${maxjump}" ' + \
                        f'file pafi.log.{rep} title ' + \
                        '"# dU/dxi  (dU/dxi)^2  psi  err  maxjump"\n'
    input_string += "\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_traj_input(loginterval, trajfile, elem):
    """
    Function to write the dump of the mlmd run
    """
    input_string = "#####################################\n"
    input_string += "#           Dumping\n"
    input_string += "#####################################\n"
    input_string += f"dump dum1 all custom {loginterval} " + \
                    f"{trajfile} id type xu yu zu " + \
                    "vx vy vz fx fy fz element \n"
    input_string += "dump_modify dum1 append yes\n"
    input_string += "dump_modify dum1 element "  # Add element type
    input_string += " ".join([p for p in elem])
    input_string += "\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_general_input(pbc,
                      masses,
                      charges,
                      atom_style,
                      replicate=None,
                      filename='atoms.in',
                      custom='',
                      nbeads=1,
                      ispimd=False):
    """
    Function to write the general parameters in the input
    """
    input_string = "# LAMMPS input file " + \
                   "to run a MLMD simulation for MLACS\n"
    input_string += "#####################################\n"
    input_string += "#           General parameters\n"
    input_string += "#####################################\n"
    if ispimd:
        input_string += "atom_modify map yes\n"
    input_string += "units        metal\n"
    input_string += "boundary     " + \
        "{0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
    input_string += f"atom_style {atom_style}\n"
    input_string += custom
    input_string += f"read_data    {filename}\n"
    if replicate is not None:
        input_string += f"replicate    {replicate}\n"
    for i, mass in enumerate(masses):
        input_string += "mass      " + str(i + 1) + "  " + str(mass) + "\n"
    if nbeads > 1:
        input_string += f"variable ibead uloop {nbeads} pad\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_pafi_input(dt,
                   temperature,
                   seed,
                   damp=None,
                   langevin=True):
    """
    Function to write the general parameters for PAFI dynamics
    """
    input_string = "#####################################\n"
    input_string += "# Compute relevant field for PAFI simulation\n"
    input_string += "#####################################\n"
    input_string += f"timestep  {dt}\n"
    input_string += "thermo    1\n"
    input_string += "min_style fire\n"
    input_string += "compute   1 all property/atom d_nx d_ny d_nz "
    input_string += "d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz\n"
    input_string += "run 0\n"
    input_string += "\n"

    input_string += "# Set up PAFI Langevin/Brownian integration\n"
    if damp is None:
        damp = "$(10*dt)"
    if not langevin:
        input_string += "fix       pafihp all pafi 1 " + \
                        f"{temperature} {damp} {seed} " + \
                        "overdamped yes com yes\n"
    else:
        input_string += "fix       pafihp all pafi 1 " + \
                        f"{temperature} {damp} {seed} " + \
                        "overdamped no com yes\n"
    input_string += "\n"
    input_string += "run 0\n"
    input_string += "\n"
    input_string += "minimize 0 0 250 250\n"
    input_string += "reset_timestep  0\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_neb_input(dt,
                  Kspring,
                  linear=False):
    """
    Function to write the general parameters for NEB
    """
    input_string = "#####################################\n"
    input_string += "# Compute relevant field for NEB simulation\n"
    input_string += "#####################################\n"
    input_string += f"timestep    {dt}\n"
    input_string += "thermo      1\n"
    input_string += f"fix         neb all neb {Kspring} " + \
                    "parallel ideal\n"
    input_string += "run 100\n"
    input_string += "reset_timestep  0\n\n"
    input_string += "variable    i equal part\n"
    input_string += "min_style   quickmin\n"
    if linear:
        input_string += "neb         0.0 0.001 1 1 1 "
    else:
        input_string += "neb         0.0 0.001 200 100 10 "
    input_string += "final atoms-1.data\n"
    input_string += "write_data  neb.$i\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_minimize_input(style,
                       criterions,
                       nitmax,
                       press=None,
                       ptype="iso",
                       vmax=None):
    """
    Function to write the general parameters for geometric optimization
    """
    etol, ftol = criterions
    input_string = "#####################################\n"
    input_string += "# Geometry optimization \n"
    input_string += "#####################################\n"
    if press is not None:
        input_string += "fix      box all box/relax " + \
                        f"{ptype} {press*10000} vmax {vmax}\n"
    input_string += "thermo    1\n"
    input_string += f"min_style {style}\n"
    input_string += f"minimize  {etol} {ftol} {nitmax} {nitmax}\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_interaction_input(pair_style,
                          pair_coeff,
                          model_post):
    """
    Function to write the interaction in the input
    """
    input_string = "#####################################\n"
    input_string += "#           Interactions\n"
    input_string += "#####################################\n"

    input_string += f"pair_style    {pair_style}\n"
    for pair in pair_coeff:
        if 'hybrid' in pair_style:
            input_string += f"pair_coeff    {pair}\n"
        else:
            input_string += f"pair_coeff    {pair}\n"
    if model_post is not None:
        for model in model_post:
            input_string += model
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_last_dump_input(workdir, elem, nsteps, nbeads=1, with_delay=True):
    """
    Function to write the dump of the last configuration of the mlmd
    """
    fname = "configurations.out"
    if nbeads > 1:
        fname = f"{fname}_${{ibead}}"
    input_string = "#####################################\n"
    input_string += "#         Dump last step\n"
    input_string += "#####################################\n"
    input_string += f"dump last all custom {nsteps} " + \
                    f"{fname}  id type xu yu zu " + \
                    "vx vy vz fx fy fz element\n"
    input_string += "dump_modify last element "
    input_string += " ".join([p for p in elem])
    input_string += "\n"
    if with_delay:
        input_string += f"dump_modify last delay {nsteps}\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_diffusion_input(msdfile):
    """
    Function to compute and output the diffusion coefficient
    """
    input_string = "#####################################\n"
    input_string += "# Compute MSD and diffusion coef\n"
    input_string += "#####################################\n"
    input_string += "variable t equal step\n"
    input_string += "compute  msd all msd\n"
    input_string += "variable msd equal c_msd[4]\n"
    input_string += "variable twopoint equal c_msd[4]/6/(step*dt+1.0e-6)\n"
    input_string += "fix      msd all vector 1000 c_msd[4]\n"
    input_string += "variable fitslope equal slope(f_msd)/6/(10000*dt)\n"
    input_string += "fix      D all print 1000 " + \
                    '"${t} ${msd} ${twopoint} ${fitslope}" ' + \
                    f"append {msdfile} title " + \
                    '"# Step   MSD   D(start)   D(slope)"\n'
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_rdf_input(rdffile, nsteps):
    """
    Function to compute and output the radial distribution function
    """
    # freq = int(nsteps/5)
    input_string = "#####################################\n"
    input_string += "#           Compute RDF\n"
    input_string += "#####################################\n"
    input_string += f"variable repeat equal {nsteps}/2 \n"
    input_string += "compute myrdf all rdf 500 1 1 \n"
    # input_string += "fix rdf all ave/time 100 10 ${freq} c_myrdf[*] " + \
    input_string += "fix rdf all ave/time 1 ${repeat}" + \
                    f" {nsteps} c_myrdf[*] " + \
                    f"file {rdffile} mode vector\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def write_lammps_NEB_ASCIIfile(filename, supercell):
    '''
    Convert Ase Atoms into an ASCII file for lammps neb calculations.

    Parameters
    ----------
    filename : :class:`str`
        name of the output file
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled

    Return
    ------
       Final NEB configuration :class: `file`
    '''
    instr = '# Final coordinates of the NEB calculation.\n'
    instr += '{0}\n'.format(len(supercell))
    for atoms in supercell:
        instr += '{} {} {} {}\n'.format(atoms.index+1, *atoms.position)
    with open(filename, "w") as w:
        w.write(instr)


# ========================================================================== #
def write_atoms_lammps_spin_style(fd, atoms, spin, velocities=True):
    """
    Function to write atoms in the LAMMPS spin style
    Loosely adapted from ASE write_lammpsdata function
    """
    fd.write("# Atoms in spin style, Written by MLACS\n\n")

    nat = len(atoms)
    fd.write(f"{nat} atoms\n")

    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    n_atom_type = len(species)
    fd.write(f"{n_atom_type} atom types\n\n")

    prismobj = Prism(atoms.get_cell())
    xhi, yhi, zhi, xy, xz, yz = convert(prismobj.get_lammps_prism(),
                                        'distance',
                                        'ASE',
                                        'metal')

    fd.write(f'0.0 {xhi:23.17g} xlo xhi\n')
    fd.write(f'0.0 {yhi:23.17g} ylo yhi\n')
    fd.write(f'0.0 {zhi:23.17g} zlo zhi\n')
    fd.write("\n\n")

    fd.write("Atoms # spin\n\n")

    pos = prismobj.vector_to_lammps(atoms.get_positions(), wrap=False)
    for i, r in enumerate(pos):
        r = convert(r, "distance", "ASE", "metal")
        s = species.index(symbols[i]) + 1
        line = f"{i+1:>6} {s:>3} "  # Index and species
        line += f"{r[0]:23.17f} {r[1]:23.17f} {r[2]:23.17f} "  # Positions
        norm = np.linalg.norm(spin[i])
        if np.isclose(norm, 0, 1e-5):
            norm = 0.0
            sp = np.zeros(3)
        else:
            sp = spin[i] / norm
        line += f"{sp[0]:23.17f} {sp[1]:23.17f} {sp[2]:23.17f} "
        line += f"{norm} "
        line += "\n"
        fd.write(line)

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities \n\n")
        vel = prismobj.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            v = convert(v, "velocity", "ASE", "metal")
            fd.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    fd.flush()


def reconstruct_mlmd_trajectory(trajfile, logfile):
    """
    Function to reconstruct a trajectory from LAMMPS
    Note that this function is made specifically for
    trajectory launched with MLACS, and won't work
    for general LAMMPS trajectories.

    Parameters
    ----------
    trajfile: Path or str
        The file for the trajectory
    logfile: Path or str
        The file for the log
    return
    ------
        Traj: list of ase.Atoms object
    """
    trajfile = Path(trajfile)
    logfile = Path(logfile)

    isspin = False

    # First we get the index of the quantities we want
    with open(logfile, "r") as fd:
        line = np.array(fd.readline().split())
        assert line[0] == "#"

        idx_epot = np.where(line == "Epot")[0][0] - 1

        if "Magn_x" in line:
            isspin = True
            idx_emagn = np.where(line == "Espin")

    trajconf = read(trajfile, ":")
    log = np.loadtxt(logfile)[:-1]  # the -1 is due to the last dump
    nconf = len(trajconf)
    assert nconf == log.shape[0]

    traj = []
    for at, logat in zip(trajconf, log):
        newat = at.copy()
        epot = logat[idx_epot]
        if isspin:
            espin = logat[idx_emagn]

            sp_n = newat.arrays.pop("c_spin[1]")
            sp_x = newat.arrays.pop("c_spin[2]")
            sp_y = newat.arrays.pop("c_spin[3]")
            sp_z = newat.arrays.pop("c_spin[4]")
            sp = np.c_[sp_x, sp_y, sp_z] * sp_n

            fsp_x = newat.arrays.pop("c_spin[5]")
            fsp_y = newat.arrays.pop("c_spin[6]")
            fsp_z = newat.arrays.pop("c_spin[7]")
            fsp = np.c_[fsp_x, fsp_y, fsp_z] * hbar / (2 * np.pi)

            epot = epot + espin

            newat.set_array("spins", sp)
            newat.set_array("spin_forces", fsp)

        try:
            f_x = newat.arrays.pop("f_ff[1]")
            f_y = newat.arrays.pop("f_ff[2]")
            f_z = newat.arrays.pop("f_ff[3]")
            forces = np.c_[f_x, f_y, f_z]
        except KeyError:
            forces = at.get_forces()

        calc = SinglePointCalculator(newat, energy=epot, forces=forces,
                                     stress=np.zeros(6))
        newat.calc = calc
        traj.append(newat)
    return traj
