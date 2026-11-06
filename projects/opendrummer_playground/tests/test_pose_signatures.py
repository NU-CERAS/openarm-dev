from copy import deepcopy

from openarm_drummer import taught


def inputs(tmp_path, monkeypatch):
    source = tmp_path/'robot.urdf'
    source.write_text('<robot/>')
    for name in ('import.json', 'scene.json', 'contact.json'):
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr(taught, 'SOURCE', source)
    monkeypatch.setattr(taught, 'CONFIG', tmp_path/'import.json')
    return {'surfaces': [{'name': 'snare', 'center': [.3, .4, .6],
                          'normal': [-.042147173286551064, -1.9245286348744917e-13, .999111413098636],
                          'radius': .1573545837031895}]}


def test_scene_fingerprint_ignores_lapack_roundoff_but_detects_geometry(tmp_path, monkeypatch):
    scene = inputs(tmp_path, monkeypatch)
    changed = deepcopy(scene)
    changed['surfaces'][0]['normal'][0] += 1.11e-16
    assert taught.legacy_scene_signature(changed) != taught.legacy_scene_signature(scene)
    assert taught.scene_signature(changed) == taught.scene_signature(scene)
    assert taught.scene_signature(scene).startswith('openarm.scene.v2:')
    changed['surfaces'][0]['center'][2] += .000001
    assert taught.scene_signature(changed) != taught.scene_signature(scene)


def test_scene_fingerprint_keeps_source_bytes_exact(tmp_path, monkeypatch):
    scene = inputs(tmp_path, monkeypatch)
    original = taught.scene_signature(scene)
    (tmp_path/'import.json').write_text('{ }')
    assert taught.scene_signature(scene) != original


def test_scene_fingerprint_normalizes_signed_zero(tmp_path, monkeypatch):
    scene = inputs(tmp_path, monkeypatch)
    scene['surfaces'][0]['normal'][1] = -0.
    original = taught.scene_signature(scene)
    scene['surfaces'][0]['normal'][1] = 0.
    assert taught.scene_signature(scene) == original
