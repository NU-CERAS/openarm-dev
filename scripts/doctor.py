#!/usr/bin/env python3
"""Run the shared doctor command from an editable checkout."""
import sys
from openarm_tools.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['doctor', *sys.argv[1:]]))
