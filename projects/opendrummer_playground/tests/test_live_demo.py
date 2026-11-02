"""Live button dispatch, cancellation, and physical transfer planning checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import glfw
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scene import build_scene
from live_demo import LiveDemo,DemoPanel,DemoControls
from transfers import plan_transfer
from taught import TaughtStrike


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
        self.assertTrue(all(p['validation']['status']=='PASS' for p in self.demo.presets.values()))

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
