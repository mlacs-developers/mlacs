
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
                    '"$t ${myetot}  ${mype} ${myke} ' + \
                    '${mytemp}  ${mypress} ${mypxx} ${mypyy} ' + \
                    '${mypzz} ${mypxy} ${mypxz} ${mypyz}" ' + \
                    f'append {logfile} title "# Step  Etot  ' + \
                    'Epot  Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'
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
def get_general_input(pbc, masses, charges, atom_style):
    """
    Function to write the general parameters in the input
    """
    input_string = "# LAMMPS input file " + \
                   "to run a MLMD simulation for MLACS\n"
    input_string += "#####################################\n"
    input_string += "#           General parameters\n"
    input_string += "#####################################\n"
    input_string += "units        metal\n"
    input_string += "boundary     " + \
        "{0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
    if atom_style == "full":
        input_string += "atom_style   full\n"
    else:
        if charges is None:
            input_string += "atom_style   atomic\n"
        else:
            input_string += "atom_style   charge\n"
    input_string += "read_data    atoms.in\n"
    for i, mass in enumerate(masses):
        input_string += "mass      " + str(i + 1) + "  " + str(mass) + "\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string


# ========================================================================== #
def get_interaction_input(bond_style,
                          bond_coeff,
                          angle_style,
                          angle_coeff,
                          pair_style,
                          pair_coeff,
                          model_post):
    """
    Function to write the interaction in the input
    """
    input_string = "#####################################\n"
    input_string += "#           Interactions\n"
    input_string += "#####################################\n"
    if bond_style is not None:
        input_string += f"bond_style   {bond_style}\n"
        for bc in bond_coeff:
            input_string += f"bond_coeff {bc}\n"

    if angle_style is not None:
        input_string += f"angle_style   {angle_style}\n"
        for angc in angle_coeff:
            input_string += f"angle_coeff {angc}\n"

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
def get_last_dump_input(workdir, elem, nsteps):
    """
    Function to write the dump of the last configuration of the mlmd
    """
    input_string = "#####################################\n"
    input_string += "#         Dump last step\n"
    input_string += "#####################################\n"
    input_string += f"dump last all custom {nsteps} {workdir}" + \
                    "configurations.out  id type xu yu zu " + \
                    "vx vy vz fx fy fz element\n"
    input_string += "dump_modify last element "
    input_string += " ".join([p for p in elem])
    input_string += "\n"
    input_string += f"dump_modify last delay {nsteps}\n"
    input_string += "#####################################\n"
    input_string += "\n\n\n"
    return input_string
