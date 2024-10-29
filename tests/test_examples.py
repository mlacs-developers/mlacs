"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
import sys
import shutil
import pytest
import subprocess
from pathlib import Path


@pytest.fixture(autouse=True)
def root():
    return Path().absolute().parents[0] / 'examples'


@pytest.mark.examples
def test_mlacs_examples(root):
    os.chdir(root)
    examples = [f.name for f in root.iterdir() if f.name.startswith('mlacs_')]
    for expl in examples:
        if 'Abinit' in expl or 'QEspresso' in expl:
            continue
        exe = sys.executable
        returncode = subprocess.call(f'{str(exe)} {str(expl)}', shell=True)
        assert returncode == 0
        assert (root / f'{expl.split(".")[0]}').exists()
        assert (root / f'{expl.split(".")[0]}' / 'MLACS.log').exists()
        shutil.rmtree(root / f'{expl.split(".")[0]}')
