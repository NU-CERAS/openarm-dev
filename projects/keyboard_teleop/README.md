# Keyboard Teleop

Jog both OpenArm v1.0 arms in MuJoCo with keys pressed in the viewer window.

## Run

Activate the environment as in the root README, then:

```sh
mjpython projects/keyboard_teleop/sim.py      # macOS
python projects/keyboard_teleop/sim.py        # Ubuntu / WSL
```

## Keys

Click the viewer window so it has focus.

| Keys | Action |
|---|---|
| `1`–`7` | select joint 1–7 |
| `8` | select the gripper |
| `↑` / `↓` | move the selected joint plus / minus; hold to keep moving (gripper: open / close) |
| `←` / `→` | select the left / right arm |
| `-` / `=` | slower / faster (0.01–0.2 rad per tap, 10 taps/s while held) |
| `Backspace` | reset both arms to the home keyframe |

The controls are listed in the top-right corner of the viewer. The bottom-left
overlay shows the selected arm and joint, target, actual position, limits and
speed. A tap moves one step; holding an arrow for more than 0.25 s keeps moving.
MuJoCo's `key_callback` reports only key presses, so the script polls whether
the arrow is still down through the `glfw` module, which shares the viewer's
GLFW library.

Targets are clamped to the joint limits in `robots/v1.0/config/*.json`; the
model's position servos track them, so the actual position lags the target
while the arm moves.

## Notes

- The viewer's own shortcuts still apply. Its `0`–`5` geom-group toggles are
  undone every frame so selecting a joint never hides the robot, which also means
  geom groups cannot be changed from the viewer panel.
- Avoid letter keys: almost every one toggles a viewer visualization flag.
- This does not read or write the shared `runs/sim/` session snapshots.
