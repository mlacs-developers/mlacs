
listofpackage='ml-snap ml-iap manybody meam molecule class2 kspace replicai extra-fix extra-pair extra-compute extra-dump qtb'


mkdir $nameofenv
python -m venv $nameofenv
source $nameofenv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

mkdir lammps

