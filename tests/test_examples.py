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


def mlacs_examples():
    root = Path().absolute().parents[0] / 'examples'
    os.chdir(root)
    expls = [f.name for f in root.iterdir() if f.name.startswith('mlacs_')]
    not_tested_expl = ['Abinit', 'QEspresso', '108Cu_EMT_300K_Snap_Rdf',
                       '256Cu_EMT_400K50GPax5_SnapMBAR']
    for expl in expls:
        if any(_ in expl for _ in not_tested_expl):
            expls.remove(expl)
    return expls


@pytest.mark.examples
@pytest.mark.parametrize("example", mlacs_examples())
def test_mlacs_examples(example):
    root = Path().absolute().parents[0] / 'examples'
    exe = sys.executable
    returncode = subprocess.call(f'{exe} {example}', shell=True)
    assert returncode == 0, \
        f'The example {example} is broken, please check it.'
    assert (root / f'{example.replace(".py", "")}').exists()
    shutil.rmtree(root / f'{example.replace(".py", "")}')


def post_examples():
    root = Path().absolute().parents[0] / 'examples'
    os.chdir(root)
    expls = [f.name for f in root.iterdir() if f.name.startswith('post_')]
    return expls


@pytest.mark.examples
@pytest.mark.parametrize("example", post_examples())
def test_mlacs_post_examples(example):
    root = Path().absolute().parents[0] / 'examples'
    exe = sys.executable
    prefix = example.replace('.py', '').replace('post_', '')
    returncode = subprocess.call(f'{exe} mlacs_{prefix}.py',
                                 shell=True)
    assert returncode == 0, \
        f'The example mlacs_{prefix}.py is broken, please check it.'
    assert (root / f'mlacs_{prefix}').exists()
    returncode = subprocess.call(f'{exe} {example}', shell=True)
    assert returncode == 0, \
        f'The example {example} is broken, please check it.'
    assert (root / f'mlacs_{prefix}' / f'{prefix}_plot.pdf').exists()
    shutil.rmtree(root / f'mlacs_{prefix}')
