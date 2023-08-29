

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
                    'Epot  Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'
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
        input_string += f"pair_coeff    {pair}\n"
    if model_post is not None:
        for model in model_post:
            input_string += model
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_last_dump_input(workdir, elem, nsteps, nbeads=1):
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
def get_rdf_input(rdffile):
    """
    Function to compute and output the radial distribution function
    """
    input_string = "#####################################\n"
    input_string += "# Compute RDF\n"
    input_string += "#####################################\n"
    input_string += "compute myrdf all rdf 250 1 1 \n"
    input_string += "fix rdf all ave/time 100 10 1000 c_myrdf[*] " + \
                    f"file {rdffile} mode vector\n"
    return input_string
