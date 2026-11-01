# Common scripts

These wrappers call the installed shared `openarm_tools` package. Activate the environment first. Each accepts `--help`; see the root README and `docs/hardware.md`.

`doctor`, `sim`, `status`, `can_check`, `zero`, `home`, `jog`, and `pose` (`save`/`move`) are the only common operator scripts. The installed CLI spells `can_check` as `can-check`.

Arm commands select arms with `--sim` and/or `--real` followed by `can0` (left) and/or `can1` (right), e.g. `python scripts/home.py --sim can0 can1 --real can1`. They run immediately; there is no preview step.
