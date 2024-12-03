"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path

from ..utilities.miscellanous import execute_from


class Manager:
    """
    Base class for managing computation files and logs,

    Parameters
    ----------

    workdir: :class:`str`
        The working directory.
        It is fixed, and may be shared with other managers writing in it.

    folder: :class:`str`
        A folder inside the working directory, specific to this object.
        Its name is relative to workdir, and its absolute path is accessed
        with the subdir property.

    subfolder: :class:`str`
        A subfolder inside the folder directory.
        Its name is relative to subdir, and its absolute path is accessed
        with the subsubdir property.
        It can be dynamic, e.g. if we perform multiple calculations.

    prefix: :class:`str`
        A prefix for making file names.


    Examples
    --------

    workdir/prefix.traj
    ├─ folder1/
    │  ├─ subfolder1/prefix.traj
    ├─ folder2/
    │  ├─ subfolder1/
    │  ├─ subfolder2/

    path -> /workdir/folder1/subfolder2/
    filepath -> /workdir/folder1/subfolder2/prefix.traj

    subdir -> /workdir/folder1/
    subsubdir -> /workdir/folder1/subfolder2/
    """

    _workdir = Path('')
    _folder = ''
    _subfolder = ''
    _prefix = ''

    def __init__(self,
                 workdir='',
                 folder='',
                 subfolder='',
                 prefix='',
                 log=None,
                 **kwargs):

        self.workdir = workdir
        self.folder = folder
        self.subfolder = subfolder
        self.prefix = prefix
        self.log = log

    @property
    def workdir(self):
        """Starting directory for execution."""
        return self._workdir

    @workdir.setter
    def workdir(self, x):
        self._workdir = Path(x or '').absolute()

    @property
    def subdir(self):
        """The path of the sub-directory."""
        return self.workdir / self.folder

    @subdir.setter
    def subdir(self, x):
        self.folder = x

    @property
    def subsubdir(self):
        """The path of the sub-sub-directory."""
        return self.subdir / self.subfolder

    @subsubdir.setter
    def subsubdir(self, x):
        self.subfolder = x

    @property
    def folder(self):
        """Name of sub-directory, relative to workdir."""
        return self._folder

    @folder.setter
    def folder(self, x):
        x = x or ''
        if isinstance(x, Path):
            if x.is_absolute():
                if x == self.subdir:
                    return
                else:
                    self._folder = str(x.relative_to(self.workdir))
            else:
                self._folder = str(x)
        else:
            self._folder = x

    @property
    def subfolder(self):
        """Name of sub-sub-directory, relative to sub-directory."""
        return self._subfolder

    @subfolder.setter
    def subfolder(self, x):
        x = x or ''
        if isinstance(x, Path):
            if x.is_absolute():
                if x == self.subsubdir:
                    return
                else:
                    self._subfolder = str(x.relative_to(self.subdir))
            else:
                self._subfolder = str(x)
        else:
            self._subfolder = x

    @staticmethod
    def exec_from_workdir(func):
        """Decorator that executes a function from the working directory."""
        def wrapper(self, *args, **kwargs):
            self.workdir.mkdir(exist_ok=True, parents=True)
            with execute_from(self.workdir):
                result = func(self, *args, **kwargs)
            return result
        return wrapper

    @staticmethod
    def exec_from_subdir(func):
        """Decorator that executes a function from the subdirectory."""
        def wrapper(self, *args, **kwargs):
            self.subdir.mkdir(exist_ok=True, parents=True)
            with execute_from(self.subdir):
                result = func(self, *args, **kwargs)
            return result
        return wrapper

    @staticmethod
    def exec_from_subsubdir(func):
        """Decorator that executes a function from the subdirectory."""
        def wrapper(self, *args, **kwargs):
            self.subsubdir.mkdir(exist_ok=True, parents=True)
            with execute_from(self.subsubdir):
                result = func(self, *args, **kwargs)
            return result
        return wrapper

    # Make an alias
    path = subsubdir
    exec_from_path = exec_from_subsubdir

    @property
    def prefix(self):
        """Prefix for naming various files."""
        return self._prefix

    @prefix.setter
    def prefix(self, x):
        self._prefix = str(x or '')

    @property
    def filepath(self):
        """
        Absolute path of the sub-sub-directory,
        or path of a file named self.prefix.
        """
        if self.prefix:
            return self.subsubdir / self.prefix
        return str(self.subsubdir / 'X')[:-1]

    def get_filepath(self, suffix: str):
        """Construct a file name and return absolute path."""
        if self.prefix:
            return str(self.subsubdir / (self.prefix + suffix))
        return str(self.subsubdir / suffix)

    def iter_subsubdir(self, nmax=None, rootname=''):
        pass

    def iter_prefix(self, prefix=None, nmax=None, rootname=''):
        pass
