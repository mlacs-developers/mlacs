from ase import Atoms
from ase.units import Hartree as Ha2eV

from mlacs.mlip import SnapDescriptor, MliapDescriptor, LinearPotential
from mlacs.mlip import MomentTensorPotential, AceDescriptor, TensorpotPotential
from mlacs.mlip import UniformWeight
from mlacs.state import LammpsState
from mlacs.calc import AbinitManager
from mlacs import OtfMlacs


"""
The file I've used to generate the C-O-H MLIP. Just here for reference
"""


def generate_atoms():
    # Create_H2O
    positions = [[0.0, 0.0, 0.0],          # O
                 [0.9584, 0.0, 0.0],       # H
                 [-0.2396, 0.9273, 0.0]]   # H
    cell = [10.0, 10.0, 10.0]
    h2o = Atoms(symbols="OH2", positions=positions, cell=cell, pbc=True)

    # Create_H2O2
    positions = [[0.0, 0.0, 0.0],          # O
                 [1.45, 0.0, 0.0],         # O
                 [-0.78, 0.76, 0.0],       # H
                 [2.23, 0.76, 0.0]]        # H
    cell = [10.0, 10.0, 10.0]
    h2o2 = Atoms(symbols="H2O2", positions=positions, cell=cell, pbc=True)

    # Create_CH4
    positions = [[0.0, 0.0, 0.0],          # C
                 [0.63, 0.63, 0.63],       # H
                 [-0.63, -0.63, 0.63],     # H
                 [0.63, -0.63, -0.63],     # H
                 [-0.63, 0.63, -0.63]]      # H
    cell = [10.0, 10.0, 10.0]
    ch4 = Atoms(symbols="CH4", positions=positions, cell=cell, pbc=True)
    return [h2o, h2o2, ch4]


def generate_mlips(atoms):
    rcut = 5
    uni = UniformWeight(nthrow=10,
                        energy_coefficient=1.0,
                        forces_coefficient=1.0,
                        stress_coefficient=0.0)

    # SNAP
    snap_params = {"twojmax": 4}
    snap_desc = SnapDescriptor(atoms=atoms,
                               rcut=rcut,
                               parameters=snap_params,
                               model="linear")
    snap = LinearPotential(descriptor=snap_desc,  # noqa
                           weight=uni, folder="SNAP")

    # MLIAP
    mliap_desc = MliapDescriptor(atoms=atoms,
                                 rcut=rcut,
                                 parameters=snap_params,
                                 model="linear",
                                 style="snap")
    mliap = LinearPotential(descriptor=mliap_desc,  # noqa
                            weight=uni, folder="MLIAP")

    # MTP
    mtp_parameters = {"level": 6}
    mtp = MomentTensorPotential(atoms=atoms,  # noqa
                                mtp_parameters=mtp_parameters)

    # ACE
    ace_desc = AceDescriptor(atoms, free_at_e={'C': -1.55731250283922E+02,
                                               'H': -1.28596715366637E+01,
                                               'O': -4.43862503906249E+02},
                             tol_e=2, tol_f=150, rcut=rcut)
    ace = TensorpotPotential(ace_desc, weight=uni, folder="ACE")

    return [ace]
    # return [snap, mliap, mtp, ace]


atoms = generate_atoms()
mlips = generate_mlips(atoms)
state = LammpsState(temperature=500, dt=0.01, nsteps=5)


abi_var = dict(
    ixc=11,
    ecut=40*Ha2eV,
    tsmear=0.01*Ha2eV,
    occopt=3,
    nband=10,
    ngkpt=[1, 1, 1],
    istwfk=1,
    toldfe=1e-4,
    autoparal=1,
    prtwf=0,
    prtden=0,
    chksymtnons=0)
pseudos = {"H": "Pseudos/H.psp8",
           "C": "Pseudos/C.psp8",
           "O": "Pseudos/O.psp8"}
calc = AbinitManager(parameters=abi_var,
                     pseudos=pseudos,
                     abinit_cmd="abinit",
                     mpi_runner="mpirun",
                     logfile="abinit.log",
                     errfile="abinit.err",
                     nproc=12)

for mlip in mlips:
    nconfs = 20
    nsteps = 150
    nsteps_eq = 50
    neq = 10
    temperature = 300
    dt = 0.1

    # #### State
    state = []
    nstate = 3
    for i in range(nstate):
        state.append(LammpsState(temperature, nsteps=nsteps,
                     nsteps_eq=nsteps_eq, dt=dt))

    # #### Otf MLACS
    sampling = OtfMlacs(atoms, state, calc, mlip,
                        workdir="generate_mlip", neq=neq)
    sampling.run(nconfs)
