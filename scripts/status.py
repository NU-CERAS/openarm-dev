#!/usr/bin/env python3
"""Run the shared status command from an editable checkout."""
import sys
from openarm_tools.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['status', *sys.argv[1:]]))
