#!/usr/bin/env python3
"""Run the shared pose command from an editable checkout."""
import sys
from openarm_tools.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['pose', *sys.argv[1:]]))
