"""
// Copyright (C) 2022-2024 MLACS group (AC, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest
import os


def pytest_collect_file(parent, path):
    if path.ext == ".py" and path.basename.startswith("mlacs_"):
        if path.dirname == os.path.abspath("../examples"):
            return pytest.Module.from_parent(parent, fspath=path)
