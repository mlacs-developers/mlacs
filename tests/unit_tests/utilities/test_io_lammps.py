"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ... import context  # noqa
from mlacs.utilities.io_lammps import (LammpsInput, LammpsBlockInput,
                                       EmptyLammpsBlockInput)


pred1 = """######################
#     test title     #
######################
test_variable 1
test_variable 2
"""


pred2 = """######################
#     test title     #
######################
test_variable 1
test_variable 3
test_variable 2
"""


pred3 = """######################
#     test title     #
######################
test_variable 4
test_variable 1
test_variable 3
test_variable 2
"""


def test_block():
    block_test = LammpsBlockInput("test", "test title")
    block_test.add_variable("test_variable1", "test_variable 1")
    block_test.add_variable("test_variable2", "test_variable 2")

    for pred, ref in zip(pred1.split("\n"), str(block_test).split("\n")):
        assert pred == ref

    block_test.add_variable("test_variable3", "test_variable 3", order=2)
    for pred, ref in zip(pred2.split("\n"), str(block_test).split("\n")):
        assert pred == ref

    block_test.add_variable("test_variable4", "test_variable 4",
                            before="test_variable1")
    for pred, ref in zip(pred3.split("\n"), str(block_test).split("\n")):
        assert pred == ref

    block_test.pop("test_variable4")
    for pred, ref in zip(pred2.split("\n"), str(block_test).split("\n")):
        assert pred == ref

    emptyblock_test = EmptyLammpsBlockInput("empty_test")
    assert str(emptyblock_test) == ""


lmp_in1 = """# preambule

##################
#     block1     #
##################
input 1

##################
#     block2     #
##################
input 2
"""


lmp_in2 = """# preambule

##################
#     block1     #
##################
input 1

##################
#     block3     #
##################
input 3

##################
#     block2     #
##################
input 2
"""


lmp_in3 = """# preambule

##################
#     block4     #
##################
input 4

##################
#     block1     #
##################
input 1

##################
#     block3     #
##################
input 3

##################
#     block2     #
##################
input 2
"""


def test_lammps_input():
    block1 = LammpsBlockInput("block1", "block1")
    block1.add_variable("input1", "input 1")
    block2 = LammpsBlockInput("block2", "block2")
    block2.add_variable("input2", "input 2")

    lmp_in = LammpsInput("preambule")
    lmp_in.add_block("block1", block1)
    lmp_in.add_block("block2", block2)
    for pred, ref in zip(lmp_in1.split("\n"), str(lmp_in).split("\n")):
        assert pred == ref

    block3 = LammpsBlockInput("block3", "block3")
    block3.add_variable("input3", "input 3")

    lmp_in.add_block("block3", block3, order=2)
    for pred, ref in zip(lmp_in2.split("\n"), str(lmp_in).split("\n")):
        assert pred == ref

    block4 = LammpsBlockInput("block4", "block4")
    block4.add_variable("input4", "input 4")

    lmp_in.add_block("block4", block4, before="block1")
    for pred, ref in zip(lmp_in3.split("\n"), str(lmp_in).split("\n")):
        assert pred == ref

    lmp_in.pop("block4")
    for pred, ref in zip(lmp_in2.split("\n"), str(lmp_in).split("\n")):
        assert pred == ref
