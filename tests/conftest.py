# conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--examples", action="store_true", default=False, help="run examples"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "examples: mark test as examples")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--examples"):
        # --runslow given in cli: do not skip slow tests
        return
    skip = pytest.mark.skip(reason="need --examples option to run")
    for item in items:
        if "examples" in item.keywords:
            item.add_marker(skip)
