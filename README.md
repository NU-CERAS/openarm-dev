# OpenArm Development

A pair of humanoid robotic arms built on [OpenArm](https://github.com/enactic/openarm), designed to interact with art: playing instruments, and more. Built by CERAS @ Northwestern University.

> **Status:** Early development. This README is preliminary and will evolve with the project.
>
> To ssh into Brady's computer to interface with CAN transceiver: ssh brady@10.105.182.237, pw: 12345678

---

## Overview

Building a bimanual robot from the open-source OpenArm platform and teaching it to engage with creative work and art, such as musical instruments and other artistic interactions.

This repo is the team's working space. It holds everything: production code, simulations, experiments, and work in progress.

## Goals

- Assemble and operate a pair of OpenArm humanoid arms
- Develop autonomy in simulation before deploying to hardware
- Achieve controlled interaction with an instrument
- Expand to other forms of artistic interaction

## Subteams

| Subteam | Focus |
|---|---|
| **Software / Autonomy** | Simulation in MuJoCo, ROS 2 integration, motion planning, learning |
| **Controls** | Low-level motor control, hardware interfacing, sim-to-real |
| **Mechanical** | Hardware design & modifications, assembly, and more |

## Software Stack

- **ROS 2 Jazzy** for middleware and robot software framework
- **MuJoCo** for physics simulation
- **Ubuntu 24.04** (native or WSL2)

## Repository Structure

> WIP, updating as the layout settles.

```
openarm-dev/
├── autonomy-dev/       # Autonomy code, using ROS2 Jazzy & MuJoCo
├── controls-dev/       # Controls subteam code
├── mechanical-dev/     # CAD, drawings, assembly notes
└── installation/       # Setup guides & notes for different systems
```

## Roadmap

- [ ] Simulation environment for both arms in MuJoCo
- [ ] ROS 2 ↔ MuJoCo integration
- [ ] Basic control on physical arms
- [ ] First instrument interactions
- [ ] *More to come*

## Acknowledgments

Built on [OpenArm](https://github.com/enactic/openarm) by Enactic.

## License

TBD
