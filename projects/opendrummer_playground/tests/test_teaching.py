"""Teach/save/reload/playback regression checks using real robot geometry."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import glfw
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scene import build_scene
from strike import StrikeSession
from taught import PoseStore,TaughtStrike
from teach import TeachingSession, NativeTeachingControls
from teach_panel import JogPanel


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
        self.store.save_ready('right','snare',np.zeros(7))
        reloaded=PoseStore(self.model,self.robot,self.scene,self.path)
        self.assertEqual(item,reloaded.get('left','snare'))
        self.assertEqual(reloaded.get('right','snare',False)['ready_rad'],[0.]*7)
        self.assertEqual(set(reloaded.document['presets']),{'left/snare','right/snare'})
        self.store.save_ready('left','snare',self.reference.approach[:7])
        with self.assertRaisesRegex(ValueError,'strike angle'):
            self.store.get('left','snare')

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
        self.assertEqual(session.drum,'rack_tom_left')
        self.assertEqual(panel.draw(850).size,(370,850))


if __name__=='__main__':unittest.main()
