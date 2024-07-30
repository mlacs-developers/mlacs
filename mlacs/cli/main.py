"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import argparse
import textwrap
from argparse import RawTextHelpFormatter
from importlib import import_module

from ..version import __version__


commands = [('correlation', 'mlacs.cli.correlation'),
            ('plot_error', 'mlacs.cli.plot_error'),
            ('plot_weights', 'mlacs.cli.plot_weights')]


def main(prog='mlacs', description='MLACS command line tool',
         version=__version__, commands=commands, hook=None, args=None):
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--version', action='version',
                        version=f"{prog}-{__version__}")
    subparsers = parser.add_subparsers(title="Sub-commands",
                                       dest="command")
    subparser = subparsers.add_parser('help',
                                      description='Help',
                                      help='Help for sub-command.')

    function = {}
    parsers = {}
    for command, module_name in commands:
        cmd = import_module(module_name).CLICommand
        docstring = cmd.__doc__
        parts = docstring.split("\n", 1)
        short, body = parts
        long = short + "\n" + textwrap.dedent(body)
        subparser = subparsers.add_parser(
            command,
            help=short,
            description=long,
            formatter_class=RawTextHelpFormatter)
        cmd.add_arguments(subparser)
        function[command] = cmd.run
        parsers[command] = subparser

    if hook:
        args = hook(parser, args)
    else:
        args = parser.parse_args(args)

    if args.command == "help":
        parser.print_help()
    elif args.command is None:
        parser.print_usage()
    else:
        f = function[args.command]
        try:
            if f.__code__.co_argcount == 1:
                f(args)
            else:
                f(args, parsers[args.command])
        except KeyboardInterrupt:
            pass
