# Hardware commands

These are commissioning tools for seven configured arm motors on one Linux SocketCAN channel. They do not scan/command every arbitrary device on a bus or include the gripper. No physical hardware was commissioned by this repository reorganization.

1. Install with `bash setup/linux.sh --hardware`.
2. Connect a supported adapter, set the appropriate bitrate, and bring up the interface using your adapter's instructions. `ip -details link show can0` is read-only. WSL additionally needs USB passthrough and kernel support.
3. Copy the matching arm JSON configuration and verify motor types/IDs, sign conventions, joint limits, temperature/tracking limits, and gains against the actual arm. Set `hardware_reviewed` to true only after that review. This flag is required for motion, not read-only diagnostics.
4. Support the arm mechanically; keep the physical emergency stop accessible. These short commands **disable motors at exit**, including errors. They do not provide sustained gravity support or a certified emergency stop. Killing the process or losing host power can bypass Python cleanup; motor/controller watchdog configuration must be established independently.

Examples (replace config/channel with the actual values):

```sh
openarm can-check --interface can0 --config /path/to/reviewed-left.json
openarm status --backend can --interface can0 --config /path/to/reviewed-left.json
openarm zero --backend can --interface can0 --config /path/to/reviewed-left.json --execute
openarm home --backend can --interface can0 --config /path/to/reviewed-left.json
openarm home --backend can --interface can0 --config /path/to/reviewed-left.json --duration 10 --execute --supported
```

`zero` only records a local software reference while motors are disabled; it does not rewrite persistent motor calibration. Existing raw limits remain in force. Jog and pose move use the same preview/execute workflow. Pose save only reads and writes a local file.

Commands check responses, stale feedback, faults, temperature, bus health, joint limits and tracking error. Motion uses a speed-bounded smoothstep trajectory. Local locks prevent concurrent commands from these tools; they do not exclude external ROS controllers or other CAN applications. Stop those before commanding an arm. These tools provide no collision checking or hard real-time guarantee; validate paths in simulation before physical execution.
