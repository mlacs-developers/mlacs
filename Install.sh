"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

nameofenv='venv'
namedirlammps='lammps'
listofpackage='ml-snap ml-iap manybody meam molecule class2 kspace replicate extra-fix extra-pair extra-compute extra-dump qtb'

# This script can be used as a starting point for a quick installation.

# Create a Python environment
mkdir $nameofenv
python -m venv $nameofenv || python -m virtualenv $nameofenv
source $nameofenv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

# Get and install LAMMPS latest stable version
git clone -b stable https://github.com/lammps/lammps.git $namedirlammps
cd $namedirlammps/src
make no-all
for i in listofpackage
do
	make yes-$i
done
make mpi
cd ../..

# Links LAMMPS executable with MLACS
ln -s $namedirlammps/src/lmp_mpi $nameofenv/bin/lmp_mpi
export ASE_LAMMPSRUN_COMMAND='lmp_mpi'
