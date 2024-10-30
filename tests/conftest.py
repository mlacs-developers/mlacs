# conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False,
        help="run tests and examples"
    )
    parser.addoption(
        "--fast", action="store_true", default=False,
        help="run tests and examples"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "examples: mark test as examples")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        # --full given in cli: do not skip examples and slow tests
        return
    skip = pytest.mark.skip(reason="need --full option to run")
    for item in items:
        if "examples" in item.keywords:
            item.add_marker(skip)
    if not config.getoption("--fast"):
        # --fast given in cli: skip slow tests
        return
    skip = pytest.mark.skip(reason="remove --fast option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
