"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
from pytest_console_scripts import ScriptRunner


def test_mlacs_cli(script_runner: ScriptRunner) -> None:
    result = script_runner.run(['mlacs', '--version'])
    assert result.returncode == 0
    assert result.stdout == 'mlacs-1.0.1\n'
    assert result.stderr == ''
    script_runner.run('mlacs --version', shell=True, check=True)

    result = script_runner.run(['mlacs', '--help'])
    assert result.returncode == 0
    assert 'MLACS command line tool' in result.stdout.split('\n')
    assert result.stderr == ''
    functions = ['correlation', 'plot_error', 'plot_weights',
                 'plot_thermo', 'plot_neff']
    for f in functions:
        assert f in result.stdout
    script_runner.run('mlacs --help', shell=True, check=True)
