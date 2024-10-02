import pytest

from pathlib import Path
import shutil
import os


@pytest.fixture(autouse=True)
def root():
    return Path().absolute()


@pytest.fixture(autouse=True)
def clean_up_hist(root):
    str_dir = "tmp_hist_dir"
    dir_list = [x[0] for x in os.walk(root) if x[0][:-1].endswith(str_dir)]
    for directory in dir_list:
        shutil.rmtree(directory)
    yield
    dir_list = [x[0] for x in os.walk(root) if x[0][:-1].endswith(str_dir)]
    for directory in dir_list:
        shutil.rmtree(directory)
