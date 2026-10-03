#!/usr/bin/env bash
# Create a virtual CAN interface for developing motor drivers with no hardware.
# Needs the container to have NET_ADMIN (set in devcontainer.json) AND the host
# kernel to have the vcan module.
set -e

sudo modprobe vcan 2>/dev/null || true   # usually fails inside a container; harmless
if ! ip link show vcan0 >/dev/null 2>&1; then
  sudo ip link add dev vcan0 type vcan || {
    echo "Could not create vcan0. The host kernel probably lacks the vcan module."
    echo "On a native Linux host run:  sudo modprobe vcan"
    echo "On Docker Desktop (Mac/Windows) or WSL2 this may be unsupported; tell a lead."
    exit 1
  }
fi
sudo ip link set up vcan0
ip -details link show vcan0 | head -n 3
echo
echo "Test it: in one terminal 'candump vcan0', in another 'cansend vcan0 123#DEADBEEF'"
