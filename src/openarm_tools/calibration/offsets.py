"""Software zero references; raw joint safety limits never move with the zero."""
import json
from openarm_tools.recording.storage import write_json


def read(path, cfg):
    if not path.exists():
        return [0.0] * len(cfg['joints'])
    record = json.loads(path.read_text())
    if record['config'] != cfg['_fingerprint']:
        raise ValueError('Calibration belongs to a different configuration. Recalibrate explicitly.')
    from openarm_tools.robot.config import validate_positions
    return validate_positions(cfg, record['offsets'])


def save(path, cfg, positions):
    from openarm_tools.robot.config import validate_positions
    write_json(path, {'config': cfg['_fingerprint'], 'offsets': validate_positions(cfg, positions)})
