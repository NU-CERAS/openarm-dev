"""Teach/save/reload/playback regression checks using real robot geometry."""
from pathlib import Path
import json
import tempfile
import unittest

import glfw
import mujoco
import numpy as np

from openarm_drummer.scene import build_scene
from openarm_drummer.strike import StrikeSession
from openarm_drummer.taught import PoseStore,TaughtStrike,legacy_scene_signature
from openarm_drummer.teach import TeachingSession, NativeTeachingControls
from openarm_drummer.teach_panel import JogPanel


class TeachingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.robot,cls.scene = build_scene()
        cls.reference = StrikeSession(cls.model,cls.robot,cls.scene)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'poses.json'
        self.store = PoseStore(self.model,self.robot,self.scene,self.path)

    def save_example(self):
        self.store.save_ready('left','snare',self.reference.approach[:7])
        self.store.save_strike('left','snare',self.reference.bottom[:7])
        return self.store.get('left','snare')

    def test_roundtrip_and_separate_arm_drum_keys(self):
        item=self.save_example()
        self.store.save_ready('right','floor_tom',np.zeros(7))
        reloaded=PoseStore(self.model,self.robot,self.scene,self.path)
        self.assertEqual(item,reloaded.get('left','snare'))
        self.assertEqual(reloaded.get('right','floor_tom',False)['ready_rad'],[0.]*7)
        self.assertEqual(set(reloaded.document['presets']),{'left/snare','right/floor_tom'})
        self.store.save_ready('left','snare',self.reference.approach[:7])
        with self.assertRaisesRegex(ValueError,'strike angle'):
            self.store.get('left','snare')

    def test_exact_legacy_save_still_loads_without_migration(self):
        item = self.save_example()
        self.assertTrue(item['scene_signature'].startswith('openarm.scene.v2:'))
        item['scene_signature'] = legacy_scene_signature(self.scene)
        self.store.document['presets']['left/snare'] = item
        self.store.write()
        self.assertEqual(PoseStore(self.model,self.robot,self.scene,self.path).get('left','snare'), item)

    def test_rejects_changed_scene_limits_and_changed_arm(self):
        self.save_example()
        q=self.reference.bottom[:7].copy();q[0]+=.02
        with self.assertRaisesRegex(ValueError,'Joints 1–6'):
            self.store.save_strike('left','snare',q)
        with self.assertRaises(ValueError):self.store.save_ready('left','snare',[float('nan')]*7)
        self.store.document['presets']['left/snare']['scene_signature']='old'
        self.store.write()
        with self.assertRaisesRegex(ValueError,'Re-teach'):
            PoseStore(self.model,self.robot,self.scene,self.path).get('left','snare')

    def test_physical_playback_only_j7_with_one_hit_each_cycle(self):
        item=self.save_example()
        initial=self.reference.approach.copy()
        run=TaughtStrike(self.model,self.robot,self.scene,item,count=2,initial=initial)
        data=mujoco.MjData(self.model);run.reset(data)
        while not run.finished:run.step(data)
        self.assertEqual(run.report()['status'],'PASS',run.failure)
        self.assertEqual(run.completed,2)
        self.assertEqual(len(run.detector.events),2)
        self.assertLess(run.metrics['max_held_joint_command_change_rad'],1e-12)
        commands=np.array([r['target_rad'] for r in run.rows if r['state']=='downstroke'])
        self.assertGreater(np.ptp(commands[:,6]),.2)
        self.assertLess(np.max(np.ptp(commands[:,run.held],axis=0)),1e-12)
        # Default demo start also approaches the saved pose through dynamics.
        demo=TaughtStrike(self.model,self.robot,self.scene,item)
        demo.reset(data)
        while not demo.finished:demo.step(data)
        self.assertEqual(demo.report()['status'],'PASS',demo.failure)

    def test_stroke_timing_uses_motor_limits_instead_of_saved_duration(self):
        preset = self.save_example()
        preset['downstroke_seconds'] = .00001
        preset['retract_seconds'] = 999
        run = TaughtStrike(self.model,self.robot,self.scene,preset)
        distance = abs(run.bottom[run.wrist]-run.ready[run.wrist])
        duration = run.cfg['downstroke_seconds']
        self.assertLessEqual(1.5*distance/duration,.98*run.controller.velocity[run.wrist]+1e-10)
        self.assertLessEqual(6*distance/duration**2,run.wrist_acceleration+1e-10)
        del preset['downstroke_seconds'], preset['retract_seconds']
        other = TaughtStrike(self.model,self.robot,self.scene,preset)
        self.assertEqual(other.cfg['downstroke_seconds'],duration)
        data = mujoco.MjData(self.model)
        run.reset(data)
        data.qpos[run.controller.q[run.wrist]] = (run.ready[run.wrist]+run.bottom[run.wrist])/2
        run.transition('retract',data)
        self.assertLess(run.cfg['retract_seconds'],duration)

    def test_live_transfer_starts_downstroke_without_ready_dwell(self):
        preset=self.save_example()
        run=TaughtStrike(self.model,self.robot,self.scene,preset,concurrent=True)
        data=mujoco.MjData(self.model)
        run.reset(data)
        data.time=float(run.approach_ends[-1])
        self.assertIsNotNone(run.prepare(data))
        self.assertEqual(run.state,'downstroke')
        self.assertEqual(run.state_start,data.time)
        self.assertFalse(run.step_context[2])
        # Standalone teaching tests retain their initial settling phase.
        standalone=TaughtStrike(self.model,self.robot,self.scene,preset)
        standalone.reset(data)
        data.time=float(standalone.approach_ends[-1])
        standalone.prepare(data)
        self.assertEqual(standalone.state,'ready')

    def test_ui_workflow_lock_and_load(self):
        session=TeachingSession(self.model,self.robot,self.scene,self.path)
        session.pose=self.reference.approach.copy();session.preview()
        session.command({'action':'save_ready'})
        self.assertTrue(session.wrist_only)
        with self.assertRaisesRegex(ValueError,'Only J7'):
            session.command({'action':'jog','joint':0,'degrees':0})
        session.command({'action':'jog','joint':6,'degrees':float(np.degrees(self.reference.bottom[6]))})
        session.command({'action':'save_strike'})
        session.command({'action':'load'})
        np.testing.assert_allclose(session.pose[:7],self.reference.approach[:7])
        json.dumps(session.state(),allow_nan=False)
        session.command({'action':'test'})
        while not session.routine.finished:session.step()
        self.assertIsNone(session.routine.failure,session.routine.failure)

    def test_publish_button_preserves_other_drums_and_replaces_arm(self):
        preset = self.save_example()
        demo_path = Path(self.temp.name)/'demo.json'
        other = {'arm':'right','drum':'floor_tom','keep':'unchanged'}
        demo_path.write_text(json.dumps({'version':1,'notes':'keep', 'presets':{
            'right/snare':{'arm':'right','drum':'snare'},'right/floor_tom':other}}))
        session = TeachingSession(self.model,self.robot,self.scene,self.path,demo_path)
        panel = JogPanel(NativeTeachingControls(session))
        panel.draw(810).save(Path(self.temp.name)/'panel.png')
        self.assertTrue(panel.enabled('publish'))
        panel.press(100,660)
        result = json.loads(demo_path.read_text())
        self.assertEqual(result['presets']['left/snare'],preset)
        self.assertNotIn('right/snare',result['presets'])
        self.assertEqual(result['presets']['right/floor_tom'],other)
        self.assertEqual(result['notes'],'keep')
        self.assertIn('Restart',session.notice)
        self.assertEqual(json.loads(self.path.read_text())['presets']['left/snare'],preset)
        before = demo_path.read_bytes()
        session.store.save_ready('left','snare',self.reference.approach[:7])
        self.assertFalse(panel.enabled('publish'))
        with self.assertRaisesRegex(ValueError,'strike angle'):
            session.command({'action':'publish'})
        self.assertEqual(demo_path.read_bytes(),before)

    def test_preview_demo_original_and_latest_saved_without_overwriting(self):
        saved = self.save_example()
        demo_path = Path(self.temp.name)/'demo.json'
        original_path = Path(self.temp.name)/'original.json'
        original = dict(saved, ready_rad=(self.reference.approach[:7]+np.array([.01,0,0,0,0,0,0])).tolist())
        original_path.write_text(json.dumps({'version':1,'presets':{'left/snare':original}}))
        demo_path.write_text(json.dumps({'version':1,'presets':{'left/snare':saved}}))
        session = TeachingSession(self.model,self.robot,self.scene,self.path,demo_path,original_path)
        panel = JogPanel(NativeTeachingControls(session))
        panel.layout(850)
        before = [p.read_bytes() for p in (self.path,demo_path,original_path)]
        panel.press(250,705)
        self.assertEqual(session.arm,'left')
        np.testing.assert_allclose(session.pose[:7],original['ready_rad'])
        np.testing.assert_allclose(session.pose[7:],session.park[7:])
        self.assertFalse(session.wrist_only)
        panel.press(100,705)
        self.assertEqual(session.arm,'left')
        np.testing.assert_allclose(session.pose[:7],saved['ready_rad'])
        np.testing.assert_allclose(session.pose[7:],session.park[7:])
        self.assertEqual(before,[p.read_bytes() for p in (self.path,demo_path,original_path)])
        # A save from another teaching window is picked up on the next load.
        updated = self.reference.approach[:7].copy(); updated[0] += .01
        self.store.save_ready('left','snare',updated)
        panel.invoke('load')
        np.testing.assert_allclose(session.pose[:7],updated)
        self.assertTrue(session.wrist_only)
        session.command({'action':'select','arm':'left','drum':'kick'})
        unchanged = session.pose.copy()
        with self.assertRaisesRegex(ValueError,'No demo ready'):
            panel.invoke('show_demo')
        np.testing.assert_array_equal(session.pose,unchanged)

    def test_fixed_ownership_and_preview_keeps_other_arm(self):
        session = TeachingSession(self.model,self.robot,self.scene,self.path)
        session.command({'action':'select','arm':'left','drum':'floor_tom'})
        self.assertEqual(session.arm,'right')
        session.pose[:7] = self.reference.approach[:7]
        other = session.pose[:7].copy()
        session.command({'action':'show_demo'})
        np.testing.assert_array_equal(session.pose[:7],other)
        with self.assertRaisesRegex(ValueError,'belongs to right'):
            self.store.save_ready('left','floor_tom',self.reference.approach[:7])

    def test_native_sliders_lock_other_arm_and_saved_ready(self):
        session=TeachingSession(self.model,self.robot,self.scene,self.path)
        session.pose=self.reference.approach.copy();session.preview()
        controls=NativeTeachingControls(session)
        controls.key(glfw.KEY_F10)
        self.assertTrue(session.wrist_only)
        self.assertEqual(controls.joint,6)
        panel=JogPanel(controls)
        panel.layout(850)
        before=session.pose.copy()
        panel.press(100,208)  # Locked J1 slider cannot alter the ready pose.
        panel.motion(160)
        panel.release()
        np.testing.assert_array_equal(session.pose,before)
        bounds=session.controller.bounds[6]
        x=79+float((self.reference.bottom[6]-bounds[0])/(bounds[1]-bounds[0]))*179
        panel.press(x,208+6*43)
        panel.motion(x+1)
        self.assertNotEqual(session.pose[6],before[6])
        panel.motion(x)
        panel.release()
        np.testing.assert_allclose(session.pose[:6],before[:6])
        np.testing.assert_allclose(session.pose[7:],before[7:])
        controls.key(glfw.KEY_F11)
        controls.key(glfw.KEY_F12)
        self.assertFalse(panel.enabled('slider_6'))
        self.assertTrue(panel.enabled('stop'))
        while not session.routine.finished:session.step()
        self.assertIsNone(session.routine.failure,session.routine.failure)
        self.assertEqual(len(session.telemetry.events),1)
        controls.key(glfw.KEY_HOME)
        np.testing.assert_allclose(session.pose[:7],self.reference.approach[:7])
        controls.key(glfw.KEY_END)
        controls.key(glfw.KEY_F8)
        self.assertEqual(session.arm,'right')
        self.assertFalse(session.wrist_only)
        before=session.pose.copy()
        controls.key(glfw.KEY_RIGHT)
        self.assertAlmostEqual(session.pose[7]-before[7],np.pi/180)
        np.testing.assert_array_equal(session.pose[:7],before[:7])
        controls.key(glfw.KEY_F9)
        self.assertEqual(session.drum,'floor_tom')
        self.assertEqual(panel.draw(850).size,(370,850))


if __name__=='__main__':unittest.main()
