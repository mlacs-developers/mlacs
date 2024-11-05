"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import sys
import shutil
import pytest
import subprocess
from pathlib import Path


def mlacs_examples():
    root = Path().absolute().parents[0] / 'examples'
    expls = [f.name for f in root.iterdir() if f.name.startswith('mlacs_')]
    not_tested_expl = ['Abinit', 'QEspresso', '32Ag_EMT_300K10GPa_Snap',
                       '500Cu_EMT_400K50GPax5_SnapMBAR']
    for expl in expls:
        if any(_ in expl for _ in not_tested_expl):
            expls.remove(expl)
    return expls


@pytest.mark.examples
@pytest.mark.parametrize("example", mlacs_examples())
def test_mlacs_examples(example):
    root = Path().absolute().parents[0]
    file = root / 'examples' / example
    exe = sys.executable
    returncode = subprocess.call(f'{exe} {file}', shell=True)
    assert returncode == 0, \
        f'The example {example} is broken, please check it.'
    assert (root / 'tests' / f'{example.replace(".py", "")}').exists()
    shutil.rmtree(root / 'tests' / f'{example.replace(".py", "")}')


def post_examples():
    root = Path().absolute().parents[0] / 'examples'
    expls = [f.name for f in root.iterdir() if f.name.startswith('post_')]
    return expls


@pytest.mark.examples
@pytest.mark.parametrize("example", post_examples())
def test_mlacs_post_examples(example):
    prefix = example.replace('.py', '').replace('post_', '')
    root = Path().absolute().parents[0]
    file = root / 'examples' / f'mlacs_{prefix}.py'
    exe = sys.executable
    returncode = subprocess.call(f'{exe} {file}', shell=True)
    assert returncode == 0, \
        f'The example mlacs_{prefix}.py is broken, please check it.'
    assert (root / 'tests' / f'mlacs_{prefix}').exists()
    file = root / 'examples' / example
    returncode = subprocess.call(f'{exe} {file}', shell=True)
    assert returncode == 0, \
        f'The example {example} is broken, please check it.'
    assert (root / 'tests' / f'mlacs_{prefix}' / f'{prefix}_plot.pdf').exists()
    shutil.rmtree(root / 'tests' / f'mlacs_{prefix}')


def ti_examples():
    root = Path().absolute().parents[0] / 'examples'
    expls = [f.name for f in root.iterdir() if f.name.startswith('neti_')
             or f.name.startswith('rsti_')]
    return expls


@pytest.mark.examples
@pytest.mark.parametrize("example", ti_examples())
def test_ti_examples(example):
    root = Path().absolute().parents[0]
    file = root / 'examples' / example
    exe = sys.executable
    returncode = subprocess.call(f'{exe} {file}', shell=True)
    assert returncode == 0, \
        f'The example {example} is broken, please check it.'
    assert (root / 'tests' / f'{example.replace(".py", "")}').exists()
    shutil.rmtree(root / 'tests' / f'{example.replace(".py", "")}')
