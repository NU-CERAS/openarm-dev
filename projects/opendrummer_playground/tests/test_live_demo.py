"""Live button dispatch, cancellation, and physical transfer planning checks."""
from pathlib import Path
import tempfile
import time
import unittest

import glfw
import mujoco
import numpy as np

from openarm_drummer.scene import build_scene
from openarm_drummer.live_demo import LiveDemo,DemoPanel,DemoControls
from openarm_drummer.transfers import plan_transfer, TransferTrajectory
from openarm_drummer.taught import TaughtStrike, DRUM_ARMS, PRESETS
import json


class TransferTrajectoryTests(unittest.TestCase):
    def test_continuous_corner_velocity_and_limits(self):
        path=np.array([[0.,0.,.4],[1.,0.,.4],[1.,1.,.4]])
        velocity=np.array([2.,3.,2.])
        acceleration=np.array([4.,5.,4.])
        trajectory=TransferTrajectory(path,velocity,acceleration,lambda q:True)
        knot=trajectory.ends[0]
        left,vl=trajectory.sample(knot-1e-7)
        right,vr=trajectory.sample(knot+1e-7)
        np.testing.assert_allclose(left,right,atol=1e-6)
        np.testing.assert_allclose(vl,vr,atol=1e-6)
        self.assertGreater(np.linalg.norm(vl),.1)
        v,a=trajectory.peaks()
        self.assertTrue(np.all(v<=velocity+1e-10))
        self.assertTrue(np.all(a<=acceleration+1e-10))
        for t in np.linspace(0,trajectory.ends[-1],501):
            q,rate=trajectory.sample(t)
            self.assertEqual(q[2],.4)
            self.assertEqual(rate[2],0.)
        np.testing.assert_allclose(trajectory.sample(0)[1],0,atol=1e-10)
        np.testing.assert_allclose(trajectory.sample(trajectory.ends[-1])[1],0,atol=1e-10)

    def test_reduces_blend_when_corner_clearance_is_tight(self):
        valid=lambda q:q[0]<=1.01 and q[1]>=-.01
        trajectory=TransferTrajectory([[0.,0.],[1.,0.],[1.,1.]],np.ones(2),np.ones(2),valid)
        self.assertLess(trajectory.blend_scale,1.)
        for t in np.linspace(0,trajectory.ends[-1],1001):
            self.assertTrue(valid(trajectory.sample(t)[0]))


class LiveDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.robot,cls.scene=build_scene()

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.demo=LiveDemo(self.model,self.robot,self.scene,Path(self.directory.name)/'demo.json')
        self.addCleanup(self.demo.close)

    def test_buttons_and_stop_queue(self):
        panel=DemoPanel(self.demo);panel.layout(850)
        panel.press(30,120)
        DemoControls(self.demo).key(glfw.KEY_2)
        self.assertEqual(list(self.demo.pending),['snare','rack_tom_left'])
        for _ in range(10):self.demo.hit('snare')
        self.assertEqual(len(self.demo.pending),8)
        panel.press(30,470)
        self.assertFalse(self.demo.busy)
        self.demo.hit('kick')
        self.assertFalse(self.demo.pending)
        self.assertIn('pedal',self.demo.notice)
        self.assertEqual(panel.draw(850).size,(370,850))

    def test_generated_presets_cover_stick_heads_and_preserve_taught_file(self):
        self.assertEqual(set(self.demo.presets),{'snare','rack_tom_left','rack_tom_right','floor_tom',
                                               'hi_hat','cymbal_left','cymbal_right'})
        self.assertEqual({p['arm'] for p in self.demo.presets.values()},{'left','right'})
        saved=json.loads(PRESETS.read_text())['presets']
        for drum,preset in self.demo.presets.items():
            self.assertEqual(preset['arm'],DRUM_ARMS[drum])
            if f"{preset['arm']}/{drum}" in saved:
                self.assertEqual(preset,saved[f"{preset['arm']}/{drum}"])

    def test_independent_queue_capacity_and_shared_physics_clock(self):
        for _ in range(9):self.demo.hit('snare')
        self.demo.hit('rack_tom_right')
        self.assertEqual(len(self.demo.lanes['left']['pending']),8)
        self.assertEqual(list(self.demo.lanes['right']['pending']),['rack_tom_right'])
        self.demo.stop()
        self.demo.hit('snare');self.demo.hit('rack_tom_right')
        deadline=time.monotonic()+40
        while self.demo.busy and time.monotonic()<deadline:
            before=float(self.demo.data.time)
            self.demo.step()
            self.assertAlmostEqual(self.demo.data.time-before,self.model.opt.timestep,places=10)
            if self.demo.concurrent_steps==0:time.sleep(.001)
        self.assertFalse(self.demo.busy)
        self.assertGreater(self.demo.concurrent_steps,0)
        self.assertEqual(len(self.demo.reports),2)
        self.assertTrue(all(r['status']=='PASS' for r in self.demo.reports),self.demo.reports)
        self.assertEqual({r['arm'] for r in self.demo.reports},{'left','right'})

    def test_each_ready_goal_preserves_the_other_arm(self):
        current=self.demo.park.copy()
        current[:7]=self.demo.presets['snare']['ready_rad']
        current[7:]=self.demo.presets['floor_tom']['ready_rad']
        for drum,preset in self.demo.presets.items():
            goal=self.demo.ready_goal(preset,current)
            selected=slice(0,7) if preset['arm']=='left' else slice(7,14)
            held=slice(7,14) if preset['arm']=='left' else slice(0,7)
            np.testing.assert_array_equal(goal[held],current[held])
            np.testing.assert_array_equal(goal[selected],preset['ready_rad'])

    def test_transfer_is_clear_and_live_start_preserves_state(self):
        c=self.demo.controller
        start=self.demo.park.copy()
        goal=start.copy();preset=self.demo.presets['snare'];goal[:7]=preset['ready_rad']
        path=plan_transfer(self.model,c,start,goal)
        probe=mujoco.MjData(self.model)
        for first,last in zip(path,path[1:]):
            for mix in np.linspace(0,1,31):
                probe.qpos[c.q]=np.array(first)*(1-mix)+np.array(last)*mix
                mujoco.mj_forward(self.model,probe)
                self.assertFalse(any(x.dist < -1e-6 for x in probe.contact))
        self.demo.data.time=12.
        before=self.demo.data.qpos.copy()
        routine=TaughtStrike(self.model,self.robot,self.scene,preset,initial=start,approach_path=path)
        routine.start_live(self.demo.data)
        fraction=self.demo.motion['transfer_speed_fraction']
        np.testing.assert_allclose(routine.transfer_velocity,fraction*c.velocity)
        distance=np.abs(np.diff(routine.approach_path,axis=0))
        peak_velocity=1.5*distance/routine.approach_durations[:,None]
        peak_acceleration=6*distance/routine.approach_durations[:,None]**2
        self.assertTrue(np.all(peak_velocity<=fraction*c.velocity+1e-10))
        self.assertTrue(np.all(peak_acceleration<=np.tile(self.demo.motion['transfer_acceleration_rad_s2'],2)+1e-10))
        self.assertEqual(self.demo.data.time,12.)
        np.testing.assert_array_equal(before,self.demo.data.qpos)
        self.assertEqual(routine.state_start,12.)


if __name__=='__main__':unittest.main()
