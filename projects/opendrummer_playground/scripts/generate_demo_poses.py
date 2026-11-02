"""Generate provisional, physically checked J7-only presets without editing taught poses."""
import copy
import json

import mujoco
import numpy as np

from openarm_drummer.model import CONFIG, HoldController
from openarm_drummer.scene import build_scene, solve_approach
from openarm_drummer.taught import TaughtStrike, scene_signature
from openarm_drummer.telemetry import ORDER

DESTINATION = CONFIG.with_name('demo_poses.json')


def wrist_preset(model, robot, scene, arm, drum, impact, height=.04):
    c = HoldController(model,robot)
    index = 6 if arm=='left' else 13
    sid = model.site(arm+'_stick_tip').id
    surface = next(s for s in scene['surfaces'] if s['name']==drum)
    normal,center = np.array(surface['normal']),np.array(surface['center'])
    probe=mujoco.MjData(model)
    q=np.array(impact)
    def tip_height(angle):
        probe.qpos[c.q]=q
        probe.qpos[c.q[index]]=angle
        mujoco.mj_forward(model,probe)
        return float((probe.site_xpos[sid]-center)@normal)
    angle=q[index]
    slope=(tip_height(angle+.0001)-tip_height(angle-.0001))/.0002
    if abs(slope)<.04:raise ValueError('Ineffective wrist axis')
    def at(wanted):
        direction=np.sign(wanted/slope)
        prev=angle
        for test in np.linspace(angle,np.clip(angle+direction*1.4,*c.bounds[index]),151)[1:]:
            if (tip_height(prev)-wanted)*(tip_height(test)-wanted)<=0:
                lo,hi=prev,test
                for _ in range(30):
                    mid=(lo+hi)/2
                    if (tip_height(lo)-wanted)*(tip_height(mid)-wanted)<=0:hi=mid
                    else:lo=mid
                return (lo+hi)/2
            prev=test
        raise ValueError('Insufficient wrist travel')
    ready=q.copy();ready[index]=at(height)
    bottom=at(-.006)
    selected=slice(0,7) if arm=='left' else slice(7,14)
    return {'arm':arm,'drum':drum,'ready_rad':ready[selected].tolist(),
            'strike_joint7_rad':bottom,'scene_signature':scene_signature(scene),
            'downstroke_seconds':.09,'retract_seconds':.4,
            'source':'provisional generated pose; replace with a taught pose when desired'}


def generate():
    model,robot,scene=build_scene()
    document={'version':1,'presets':{},'unavailable':{},'notes':'Provisional poses, validated in simulation. User-taught poses are stored separately.'}
    for drum in ORDER:
        if drum=='kick':
            document['unavailable'][drum]='Kick is a pedal target; this demo drives the two stick arms.'
            continue
        if drum=='snare':
            from openarm_drummer.strike import StrikeSession
            reference=StrikeSession(model,robot,scene)
            preset={'arm':'left','drum':'snare','ready_rad':reference.approach[:7].tolist(),
                    'strike_joint7_rad':float(reference.bottom[6]),'scene_signature':scene_signature(scene),
                    'downstroke_seconds':.09,'retract_seconds':.65,'source':'provisional validated snare pose'}
            routine=TaughtStrike(model,robot,scene,preset)
            data=mujoco.MjData(model);routine.reset(data)
            while not routine.finished:routine.step(data)
            if routine.report()['status']!='PASS':raise RuntimeError(routine.failure)
            preset['validation']=routine.report()
            document['presets']['left/snare']=preset
            print('PASS left/snare',flush=True)
            continue
        surface=next(s for s in scene['surfaces'] if s['name']==drum)
        preferred='left' if surface['center'][1]>0 else 'right'
        found=False
        failures=[]
        for arm in (preferred,'right' if preferred=='left' else 'left'):
            c=HoldController(model,robot)
            shoulder=np.array([0,.0925 if arm=='left' else -.0925,1.0018])
            n=np.array(surface['normal'])
            toward=shoulder-np.array(surface['center']);toward-=n*(toward@n)
            toward/=np.linalg.norm(toward)
            for offset in (0.,.4,.65):
                local=copy.deepcopy(scene)
                target=next(s for s in local['surfaces'] if s['name']==drum)
                target['center']=(np.array(surface['center'])+offset*surface['radius']*toward).tolist()
                horizontal=np.array(target['center'])-shoulder;horizontal[2]=0
                horizontal/=np.linalg.norm(horizontal)
                for wrist in ((-.9,-.7,-1.1,0.,.7) if arm=='left' else (.9,.7,1.1,0.,-.7)):
                    for lean in (.65,.3,1.2):
                        direction=lean*horizontal-n
                        result=solve_approach(model,robot,local,arm,drum,height=0.,
                            wrist_angle=wrist,joint_margin=.04,tip_direction=direction,attempts=5,iterations=180)
                        if result['position_error_m']>.001 or result['direction_error_degrees']>1.:
                            continue
                        try:
                            preset=wrist_preset(model,robot,scene,arm,drum,result['joint_positions'])
                            routine=TaughtStrike(model,robot,scene,preset)
                            data=mujoco.MjData(model);routine.reset(data)
                            while not routine.finished:routine.step(data)
                            report=routine.report()
                            if report['status']!='PASS':raise ValueError(report['failure'])
                            preset['validation']=report
                            preset['impact_position_m']=target['center']
                            document['presets'][f'{arm}/{drum}']=preset
                            print(f"PASS {arm}/{drum}: {report['metrics']['peak_force_N']:.1f} N",flush=True)
                            found=True;break
                        except (ValueError,RuntimeError) as error:
                            failures.append(str(error))
                    if found:break
                if found:break
            if found:break
        if not found:
            document['unavailable'][drum]='No collision-free, dynamically valid wrist strike found. '+('; '.join(sorted(set(failures)))[:220] if failures else 'Target outside the tested wrist workspace.')
            print('UNAVAILABLE',drum,document['unavailable'][drum],flush=True)
        DESTINATION.write_text(json.dumps(document,indent=2)+'\n')
    DESTINATION.write_text(json.dumps(document,indent=2)+'\n')
    return document


if __name__=='__main__':generate()
