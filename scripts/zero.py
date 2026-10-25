#!/usr/bin/env python3
"""Run the shared zero command from an editable checkout."""
import sys
from openarm_tools.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['zero', *sys.argv[1:]]))
