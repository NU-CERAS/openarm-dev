# Examples

With the shared environment activated:

- `mjpython examples/actuator_motion.py` on macOS, or `python examples/actuator_motion.py` on Ubuntu: preserved v0.3 torque-driven demonstration. It is independent of saved CLI simulation sessions.
- After an Ubuntu ROS build: `openarm sim --ros`, then in another activated/sourced terminal `ros2 run openarm_mujoco_sim simple_motion.py --cycles 1`: preserved v1.0 ROS motion example.

The old direct-CAN sim-to-real mirror was removed: it duplicated hardcoded mappings and unchecked session ownership. Use the shared CAN commands for commissioning before developing project-specific sim-to-real behavior.
