from pathlib import Path

import mido


def write_example(path, pattern="duet"):
    patterns = {
        "duet": [(0, 38), (0, 43), (1.5, 38), (3., 43), (4.5, 38), (4.5, 43)],
        "heldout": [(0, 38), (1.8, 38), (1.8, 43), (3.6, 43), (5.4, 38)],
        "fill": [(0, 38), (5., 48), (10., 47), (15., 43), (20., 38), (20., 43)],
        "groove": sorted([(0, 38), (0, 43), (30., 43)] + [(1.5*i, 38) for i in range(1, 21)]
                         + [(1.5*i+.75, 43) for i in range(1, 19)]),
    }
    if pattern not in patterns:
        raise ValueError("Unknown MIDI validation pattern")
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    events = []
    for time_s, note in patterns[pattern]:
        ticks = round(mido.second2tick(time_s, 480, 500000))
        events.extend([(ticks, mido.Message("note_on", channel=9, note=note, velocity=96)),
                       (ticks+1, mido.Message("note_off", channel=9, note=note, velocity=0))])
    previous = 0
    for tick, message in sorted(events, key=lambda row: row[0]):
        track.append(message.copy(time=tick-previous))
        previous = tick
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    return path
