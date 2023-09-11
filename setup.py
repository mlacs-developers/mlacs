import os
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

versionfile = os.path.join('mlacs', 'version.py')
with open(versionfile) as f:
    code = compile(f.read(), versionfile, 'exec')
    exec(code)

install_requires = ["numpy>=1.17.0",
                    "ase>=3.22"]

extra_requires = ["icet>=1.4",
                  "scikit-learn",
                  "pymbar"
                  "cython",
                  "netCDF4==1.2.2"]


if __name__ == "__main__":
    setup(
          name="mlacs",
          version=__version__,
          author="Aloïs Castellano",
          author_email="alois.castellano@uliege.be",
          packages=find_packages(),
          entry_points={"console_scripts": ["mlacs=mlacs.cli.main:main"]},
          description="Machine-Learning Assisted Canonical Sampling",
          license_files=("LICENSE.txt",),
          long_description=long_description,
          install_requires=install_requires,
          extra_requires=extra_requires
         )
