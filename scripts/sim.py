#!/usr/bin/env python3
"""Run the shared sim command from an editable checkout."""
import sys
from openarm_tools.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['sim', *sys.argv[1:]]))
