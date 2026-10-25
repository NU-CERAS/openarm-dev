#!/usr/bin/env python3
"""Run the shared can-check command from an editable checkout."""
import sys
from openarm_tools.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['can-check', *sys.argv[1:]]))
