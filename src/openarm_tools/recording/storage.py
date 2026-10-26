"""Atomic JSON files and process locks for local tool sessions."""
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def lock(identity):
    # Linux/macOS/WSL only. Lock survives crashes as an unlocked file; never unlink
    # an active lock inode. Hardware key is independent of checkout/config.
    import fcntl
    name = hashlib.sha256(identity.encode()).hexdigest()[:24]
    folder = Path.home() / '.cache' / 'openarm' / 'locks'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'openarm-{name}.lock'
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another OpenArm command owns this session/interface.') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
