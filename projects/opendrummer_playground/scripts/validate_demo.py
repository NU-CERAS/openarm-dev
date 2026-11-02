"""Execute every enabled drum button through continuous physics, without a GUI."""
import json
import time

from live_demo import LiveDemo
from model import OUTPUT
from scene import build_scene
from telemetry import ORDER


def validate():
    model,robot,scene=build_scene()
    destination=OUTPUT/'live-demo-validation.json'
    demo=LiveDemo(model,robot,scene,log_path=destination)
    drums=[d for d in ORDER if d in demo.presets]
    for drum in drums:demo.hit(drum)
    ticks=0
    last_time=-1.
    deadline=time.monotonic()+180
    try:
        while demo.busy:
            if time.monotonic()>deadline:raise RuntimeError('Live demo validation timed out')
            demo.step()
            assert demo.data.time>=last_time,'Simulation time reset during a transfer'
            last_time=float(demo.data.time)
            ticks+=1
            if ticks%50==0:time.sleep(.001)
        assert len(demo.reports)==len(drums),demo.reports
        assert all(r['status']=='PASS' and r['hit_count']==1 for r in demo.reports),demo.reports
        assert all(r['metrics']['max_held_joint_command_change_rad']<1e-10 for r in demo.reports)
        fraction=demo.motion['transfer_speed_fraction']
        assert all(r['metrics']['max_transfer_command_motor_fraction']<=fraction+1e-8 for r in demo.reports)
        assert all(r['metrics']['max_transfer_actual_motor_fraction']<=fraction+.001 for r in demo.reports)
        assert len(demo.telemetry.events)==len(drums)
        assert demo.concurrent_steps>0,'Arms never ran concurrently'
        print(f'PASS: {len(drums)} continuous drum transfers and {len(drums)} J7-only hits.')
    finally:demo.close()
    doc=json.loads(destination.read_text())
    doc.update(status='PASS',timestep_s=float(model.opt.timestep),drums=drums,
               simulation_time_s=float(demo.data.time))
    destination.write_text(json.dumps(doc,indent=2)+'\n')


if __name__=='__main__':validate()
