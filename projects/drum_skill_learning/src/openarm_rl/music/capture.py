"""Receive MIDI without opening a robot connection or sending motor commands."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import mido


TICKS_PER_BEAT = 9600
TEMPO = 500000


def input_backend():
    """Import the optional native MIDI dependency only for live capture."""
    try:
        backend = mido.Backend("mido.backends.rtmidi")
        backend.get_input_names()
        return backend
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Live MIDI needs python-rtmidi. Install: "
            "pip install -e 'projects/drum_skill_learning[midi]'"
        ) from error


def write_timed_midi(path, messages, duration_s, description):
    """Write absolute seconds at a fixed tempo; preserve incoming channel messages."""
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("MIDI duration must be finite and nonnegative")
    midi = mido.MidiFile(type=0, ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO))
    track.append(mido.MetaMessage("text", text=description))
    previous_tick = 0
    for time_s, message in sorted(messages, key=lambda row: row[0]):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("MIDI event times must be finite and nonnegative")
        # Clock/start/stop and other system messages remain in raw JSONL. SMF is
        # for channel messages and SysEx, not a transport-clock recording.
        if not hasattr(message, "channel") and message.type != "sysex":
            continue
        tick = round(mido.second2tick(time_s, TICKS_PER_BEAT, TEMPO))
        track.append(message.copy(time=tick-previous_tick))
        previous_tick = tick
    end_tick = max(previous_tick, round(mido.second2tick(duration_s, TICKS_PER_BEAT, TEMPO)))
    track.append(mido.MetaMessage("end_of_track", time=end_tick-previous_tick))
    midi.save(path)


def capture_midi(output, port_name, duration_s=None, *, backend=None, clock=time.monotonic, wait=time.sleep):
    """Record backend receive times; injected backend/clock enable hardware-free tests."""
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("Capture duration must be positive and finite")
    if not port_name:
        raise ValueError("Choose an exact input port name using capture --list-ports")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    backend = input_backend() if backend is None else backend
    if port_name not in backend.get_input_names():
        raise ValueError(f"MIDI input port not found: {port_name!r}; use capture --list-ports")
    messages, count, stopped = [], 0, "duration"
    # Open the device before creating an output so a missing/disconnected port
    # cannot masquerade as a successful empty recording.
    with backend.open_input(port_name) as port:
        output.mkdir(parents=True, exist_ok=False)
        started_utc = datetime.now(timezone.utc).isoformat()
        started = clock()
        with (output/"events.jsonl").open("w") as raw:
            try:
                while duration_s is None or clock()-started < duration_s:
                    for message in port.iter_pending():
                        received_s = clock()-started
                        if duration_s is not None and received_s > duration_s:
                            break
                        document = {"received_s": received_s, "message": message.dict()}
                        raw.write(json.dumps(document)+"\n")
                        raw.flush()
                        messages.append((received_s, message.copy()))
                        count += 1
                    wait(.001)
            except KeyboardInterrupt:
                stopped = "interrupted"
            elapsed = clock()-started
    elapsed = min(elapsed, duration_s) if duration_s is not None else elapsed
    midi_path = output/"performed.mid"
    write_timed_midi(midi_path, messages, elapsed, "Recorded MIDI input; timestamps are host receive times")
    document = {
        "schema": "openarm.performed_midi.v1", "measurement_source": "live_MIDI_input",
        "port": port_name, "started_utc": started_utc, "duration_s": elapsed,
        "message_count": count, "note_on_count": sum(m.type == "note_on" and m.velocity > 0 for _, m in messages),
        "stopped_by": stopped, "time_origin": "input_port_open_host_monotonic_clock",
        "timestamp_kind": "host_receive_time", "score_start_s": None,
        "latency_correction_s": None, "intensity": "received_MIDI_velocity",
        "physical_measurement_note": "The port source must be an actual drum module for physical feedback; no force or stick speed is inferred.",
        "performed_sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
        "raw_events": "events.jsonl", "robot_motion": False,
    }
    (output/"performed.mid.json").write_text(json.dumps(document, indent=2)+"\n")
    return document
