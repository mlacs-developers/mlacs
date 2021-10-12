import re
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

with open("mlacs/version.py") as f:
    version = re.search("__version__ = '(.*)'", f.read()).group(1)

install_requires = ["numpy>=1.17.0",
                    "ase>=3.22"]

extra_requires = ["icet>=1.4"]

if __name__ == "__main__":
    setup(
          name="mlacs",
          version=version,
          author="Aloïs Castellano",
          author_email="alois.castellano@cea.fr",
          packages=["mlacs", "test"],
          description="Machine-Learning Assisted Canonical Sampling",
          long_description=long_description,
          install_requires=install_requires,
          extra_requires=extra_requires
         )
