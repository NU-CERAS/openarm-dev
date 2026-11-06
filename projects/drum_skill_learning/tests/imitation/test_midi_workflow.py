import hashlib
import json

import mido
import pytest

from openarm_rl.music.capture import capture_midi, input_backend
from openarm_rl.music.feedback import compare_performance
from openarm_rl.music.score import ContactScore
from openarm_rl.music.workflow import export_simulated_contacts, import_midi, main


def target_midi(path):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.Message("note_on", channel=9, note=38, velocity=90),
        mido.Message("note_off", channel=9, note=38, time=1),
        mido.Message("note_on", channel=9, note=38, velocity=100, time=479),
        mido.MetaMessage("set_tempo", tempo=250000),
        mido.Message("note_off", channel=9, note=38, time=1),
        mido.Message("note_on", channel=9, note=38, velocity=110, time=479),
        mido.Message("note_off", channel=9, note=38, time=1),
    ])
    midi.tracks.append(track)
    midi.save(path)
    return path


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now = round(self.now+seconds, 6)


class Input:
    def __init__(self, clock, scheduled):
        self.clock, self.scheduled = clock, list(scheduled)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_pending(self):
        while self.scheduled and self.scheduled[0][0] <= self.clock():
            yield self.scheduled.pop(0)[1]


class Backend:
    def __init__(self, clock, scheduled=()):
        self.port = Input(clock, scheduled)

    def get_input_names(self):
        return ["test drum"]

    def open_input(self, name):
        assert name == "test drum"
        return self.port


def test_import_capture_evaluate_roundtrip_with_tempo_change_and_clock_offset(tmp_path):
    source = target_midi(tmp_path/"input.mid")
    imported = tmp_path/"imported"
    report = import_midi(source, imported)
    assert (imported/"source.mid").read_bytes() == source.read_bytes() == (imported/"target.mid").read_bytes()
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert [goal.time_s for goal in ContactScore.read(imported/"target.mid").goals] == [0., .5, .75]
    clock = Clock()
    scheduled = [(at, mido.Message("note_on", channel=9, note=38, velocity=velocity))
                 for at, velocity in [(1.04, 80), (1.53, 90), (1.78, 100)]]
    recorded = tmp_path/"recording"
    capture = capture_midi(recorded, "test drum", 2., backend=Backend(clock, scheduled), clock=clock, wait=clock.wait)
    assert capture["note_on_count"] == 3
    assert capture["robot_motion"] is False
    with pytest.raises(ValueError, match="score-start-seconds"):
        compare_performance(imported/"target.mid", recorded/"performed.mid")
    result = compare_performance(imported/"target.mid", recorded/"performed.mid", latency_s=.03, score_start_s=1.)
    assert result["f1"] == 1.
    assert result["timing_errors_s"] == pytest.approx([.01, 0., 0.], abs=1e-7)
    assert result["velocity_mae"] == 10.
    assert result["feedback_source"] == "live_MIDI_input"
    assert result["reward_components"]["timing"] == pytest.approx(-.25*.01/.06)
    assert result["reward_components"]["intensity"] == pytest.approx(-3*.25*10/126)
    assert result["total_reward"] == pytest.approx(sum(hit["reward"] for hit in result["hits"]))
    assert main(["evaluate", "--target-midi", str(imported/"target.mid"), "--performed-midi", str(recorded/"performed.mid"),
                 "--score-start-seconds", "1", "--latency-seconds", ".03", "--output", str(tmp_path/"feedback.json")]) == 0


def test_empty_and_interrupted_capture_save_valid_midi_and_penalize_misses(tmp_path):
    source = target_midi(tmp_path/"target.mid")
    clock = Clock()
    recorded = tmp_path/"empty"
    result = capture_midi(recorded, "test drum", .01, backend=Backend(clock), clock=clock, wait=clock.wait)
    assert result["note_on_count"] == 0
    assert mido.MidiFile(recorded/"performed.mid").length == pytest.approx(.01)
    evaluated = compare_performance(source, recorded/"performed.mid", score_start_s=0.)
    assert evaluated["missed_hits"] == 3 and evaluated["extra_hits"] == 0
    assert evaluated["total_reward"] == -6.
    def stop(_):
        raise KeyboardInterrupt
    interrupted = capture_midi(tmp_path/"interrupted", "test drum", backend=Backend(clock), clock=clock, wait=stop)
    assert interrupted["stopped_by"] == "interrupted"
    assert (tmp_path/"interrupted/performed.mid").exists()


def test_capture_keeps_realtime_messages_in_raw_log(tmp_path):
    clock = Clock()
    backend = Backend(clock, [(0., mido.Message("start")), (0., mido.Message("clock")),
                             (0., mido.Message("note_on", channel=9, note=38, velocity=65))])
    report = capture_midi(tmp_path/"recording", "test drum", .002, backend=backend, clock=clock, wait=clock.wait)
    assert report["message_count"] == 3
    raw = [json.loads(line) for line in (tmp_path/"recording/events.jsonl").read_text().splitlines()]
    assert [row["message"]["type"] for row in raw] == ["start", "clock", "note_on"]
    assert [m.type for m in mido.MidiFile(tmp_path/"recording/performed.mid") if not m.is_meta] == ["note_on"]


def test_simulated_contact_export_keeps_early_hits_and_never_scores_placeholder_velocity(tmp_path):
    source = target_midi(tmp_path/"target.mid")
    evaluation = tmp_path/"evaluation.json"
    evaluation.write_text(json.dumps({"hits": [{"target": "snare", "time_s": -.02},
                                               {"target": "snare", "time_s": .5},
                                               {"target": "snare", "time_s": 1.}]}))
    metadata = export_simulated_contacts(evaluation, tmp_path/"performed.mid")
    assert metadata["score_start_s"] == .02
    assert metadata["intensity"] == "uncalibrated_placeholder"
    report = compare_performance(source, tmp_path/"performed.mid", require_alignment=True)
    assert report["correct_hits"] == 2 and report["missed_hits"] == report["extra_hits"] == 1
    assert report["timing_errors_s"] == pytest.approx([-.02, 0.])
    assert report["intensity_scored"] is False and report["velocity_mae"] is None
    assert all(hit["midi_velocity"] is None for hit in report["hits"])
    assert report["reward_components"]["extra"] == -.5
    assert report["reward_components"]["missed"] == -2.
    with pytest.raises(ValueError, match="placeholder"):
        compare_performance(source, tmp_path/"performed.mid", intensity="on")


def test_calibrated_simulated_export_and_sidecar_hash_validation(tmp_path):
    source = target_midi(tmp_path/"target.mid")
    evaluation = tmp_path/"evaluation.json"
    evaluation.write_text(json.dumps({"hits": [{"target": "snare", "time_s": 0., "midi_velocity": 84.8}]}))
    export_simulated_contacts(evaluation, tmp_path/"performed.mid")
    report = compare_performance(source, tmp_path/"performed.mid")
    assert report["intensity_scored"] is True and report["velocity_mae"] == 5.
    assert report["intensity_source"] == "calibrated_simulated_speed_estimate"
    (tmp_path/"performed.mid").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="SHA256"):
        compare_performance(source, tmp_path/"performed.mid")


def test_import_does_not_silently_drop_unsupported_drum_or_overwrite(tmp_path):
    source = target_midi(tmp_path/"target.mid")
    midi = mido.MidiFile(source)
    midi.tracks[0].insert(0, mido.Message("note_on", channel=9, note=36, velocity=90))
    midi.save(source)
    with pytest.raises(ValueError, match="explicit mapping or arrangement"):
        import_midi(source, tmp_path/"failed")
    assert not (tmp_path/"failed").exists()
    source = target_midi(source)
    import_midi(source, tmp_path/"imported")
    with pytest.raises(FileExistsError):
        import_midi(source, tmp_path/"imported")


@pytest.mark.parametrize("duration", [0., -1., float("nan"), float("inf")])
def test_capture_rejects_invalid_duration_without_touching_output(tmp_path, duration):
    with pytest.raises(ValueError, match="duration"):
        capture_midi(tmp_path/"invalid", "test drum", duration)
    assert not (tmp_path/"invalid").exists()


def test_cli_help_needs_no_midi_backend(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    assert "import" in capsys.readouterr().out


def test_missing_optional_backend_has_actionable_error_without_traceback(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("rtmidi")
    monkeypatch.setattr(mido, "Backend", missing)
    with pytest.raises(RuntimeError, match="python-rtmidi"):
        input_backend()
    assert main(["capture", "--list-ports"]) == 2
    error = capsys.readouterr().err
    assert "projects/drum_skill_learning[midi]" in error and "Traceback" not in error


@pytest.mark.parametrize("report", [{}, {"hits": None}, {"hits": [{"target": "snare", "time_s": None}]},
                                   {"hits": [{"target": "snare", "time_s": 0., "midi_velocity": "loud"}]}])
def test_invalid_contact_reports_never_create_performed_midi(tmp_path, report):
    source = tmp_path/"evaluation.json"
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        export_simulated_contacts(source, tmp_path/"performed.mid")
    assert not (tmp_path/"performed.mid").exists()


def test_zero_tempo_is_rejected_before_import_output_is_created(tmp_path):
    source = target_midi(tmp_path/"target.mid")
    midi = mido.MidiFile(source)
    midi.tracks[0][0].tempo = 0
    midi.save(source)
    with pytest.raises(ValueError, match="positive ticks per beat and tempo"):
        import_midi(source, tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()
