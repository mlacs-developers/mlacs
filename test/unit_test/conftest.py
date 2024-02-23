import pytest

from pathlib import Path


@pytest.fixture(autouse=True)
def root():
    return Path()
