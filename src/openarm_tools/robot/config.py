"""Repository-relative assets and validated per-arm configuration."""
import hashlib
import json
import math
import os
from pathlib import Path


def root():
    path = Path(os.environ.get('OPENARM_ROOT', Path(__file__).resolve().parents[3]))
    if not (path / 'robots').is_dir():
        raise ValueError('Use an editable checkout or set OPENARM_ROOT to the repository.')
    return path


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite number, got {value!r}')
    return float(value)


def load(path=None, robot='v1.0', arm='left'):
    path = Path(path) if path else root() / 'robots' / robot / 'config' / f'{arm}.json'
    cfg = json.loads(path.read_text())
    if cfg['robot'] not in ('v0.3', 'v1.0') or cfg['arm'] not in ('left', 'right'):
        raise ValueError('Configuration must identify a supported robot version and arm.')
    model = (root() / cfg['model']).resolve()
    if not model.is_relative_to((root() / 'robots').resolve()) or not model.is_file():
        raise ValueError('Model must be an existing file inside robots/.')
    joints = cfg['joints']
    if not joints or len({j['name'] for j in joints}) != len(joints):
        raise ValueError('Joint names must be nonempty and unique.')
    for j in joints:
        lo, hi = map(finite, j['limits'])
        if lo >= hi or not lo <= finite(j['home']) <= hi:
            raise ValueError(f'Invalid limits/home for {j["name"]}')
        if finite(j['sign']) not in (-1, 1):
            raise ValueError('Joint sign must be -1 or 1.')
        finite(j.get('motor_zero_rad', 0.0))
        for key in ('kp', 'kd'):
            if finite(j[key]) < 0:
                raise ValueError(f'{key} must be nonnegative.')
    for key in ('max_speed', 'feedback_timeout', 'max_tracking_error', 'max_temperature'):
        if finite(cfg[key]) <= 0:
            raise ValueError(f'{key} must be positive.')
    if type(cfg.get('hardware_reviewed')) is not bool:
        raise ValueError('hardware_reviewed must be a JSON boolean.')
    for key in ('send_id', 'recv_id'):
        ids = [j[key] for j in joints]
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 < i <= 0x7ff for i in ids):
            raise ValueError(f'Invalid/duplicate {key}.')
    cfg['_path'] = str(path.resolve())
    cfg['_fingerprint'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def validate_positions(cfg, q):
    if len(q) != len(cfg['joints']):
        raise ValueError('Position vector has the wrong number of joints.')
    for j, value in zip(cfg['joints'], q):
        value = finite(value)
        if not j['limits'][0] <= value <= j['limits'][1]:
            raise ValueError(f'{j["name"]}: {value:.4f} outside limits {j["limits"]}')
    return list(q)


def model_position(joint, motor_value):
    return finite(joint['sign']) * (finite(motor_value) - finite(joint.get('motor_zero_rad', 0.0)))


def motor_position(joint, model_value):
    return finite(joint.get('motor_zero_rad', 0.0)) + finite(joint['sign']) * finite(model_value)
