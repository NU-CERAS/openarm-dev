"""Bounded smoothstep trajectories, independent of transport."""
import math
from openarm_tools.robot.config import finite, validate_positions


def trajectory(cfg, start, target, duration, period=0.01):
    validate_positions(cfg, start)
    validate_positions(cfg, target)
    duration = finite(duration)
    if duration <= 0:
        raise ValueError('Duration must be positive.')
    # Cubic smoothstep has maximum derivative 1.5.
    required = max(1.5 * abs(b - a) / cfg['max_speed'] for a, b in zip(start, target))
    if duration < required:
        raise ValueError(f'Move requires at least {required:.2f}s at the configured speed limit.')
    count = max(1, math.ceil(duration / period))
    for i in range(count + 1):
        s = i / count
        blend = s * s * (3 - 2 * s)
        yield duration * s, [a + blend * (b - a) for a, b in zip(start, target)]
