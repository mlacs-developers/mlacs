import numpy as np
from ase.build import bulk

from .. import context  # noqa
from mlacs.mlip.descriptor import Descriptor


def create_obj():
    at = bulk("Si")
    desc = Descriptor(at)
    return at, desc


def test_elem():
    """
    Here we test that the descriptor has the right elements,
    Z, masses and charges compared to the input atoms
    """
    at, desc = create_obj()
    ref_elements = np.array(["Si"])
    ref_Z = np.array([14])
    ref_masses = np.array([at.get_masses()[0]])
    ref_charges = None

    assert desc.elements == ref_elements
    assert desc.Z == ref_Z
    assert desc.masses == ref_masses
    assert desc.charges == ref_charges

    ref_charges = np.array([1])
    at.set_initial_charges([1, 1])
    desc = Descriptor(at)
    assert desc.charges == ref_charges
