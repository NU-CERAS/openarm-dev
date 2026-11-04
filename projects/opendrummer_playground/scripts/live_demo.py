"""Live drum-button session: collision-checked travel, then a J7-only hit."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
import textwrap

import mujoco
import glfw
import numpy as np
from PIL import Image, ImageDraw

from openarm_drummer.model import CONFIG, OUTPUT, HoldController
from openarm_drummer.taught import PoseStore, TaughtStrike, DRUM_ARMS
from openarm_drummer.telemetry import DrumTelemetry, LABELS, ORDER, font
from openarm_drummer.transfers import plan_transfer

DEMO_POSES = CONFIG.with_name('demo_poses.json')


def prepare_transfer(model, robot, scene, start, goal, preset, motion, cancel):
    """Try fast motion first, backing off only when coupled simulation rejects it."""
    private=copy.copy(model)
    controller=HoldController(private,robot)
    path=plan_transfer(private,controller,start,goal,cancel,arm=preset['arm'])
    last_failure='No feasible motion'
    scales=(1.,.8,.6,.4,.25,.125,.0625,.03125)
    transfer_index=strike_index=0
    for attempt in range(15):
        transfer_scale,strike_scale=scales[transfer_index],scales[strike_index]
        if cancel.is_set():raise ValueError('Planning cancelled')
        candidate=copy.deepcopy(motion)
        candidate['transfer_acceleration_rad_s2']=[x*transfer_scale for x in motion['transfer_acceleration_rad_s2']]
        # Slow the strike only if the preflight also finds a physical failure.
        candidate['strike_acceleration_rad_s2']=motion.get('strike_acceleration_rad_s2',120)*strike_scale
        candidate['strike_acceleration_by_drum_rad_s2']={k:v*strike_scale for k,v in motion.get('strike_acceleration_by_drum_rad_s2',{}).items()}
        candidate['preflight_transfer_scale']=transfer_scale
        candidate['preflight_strike_scale']=strike_scale
        try:
            routine=TaughtStrike(private,robot,scene,preset,initial=start,approach_path=path,
                ready_pose=goal,transfer_motion=candidate,concurrent=True)
            data=mujoco.MjData(private)
            data.qpos[controller.q]=start
            mujoco.mj_forward(private,data)
            routine.start_live(data)
            while not routine.finished:
                if cancel.is_set():raise ValueError('Planning cancelled')
                torque=routine.prepare(data)
                if torque is None:break
                data.ctrl[controller.a]=torque
                mujoco.mj_step(private,data)
                # Only the candidate arm is commanded in this prediction; the
                # other arm's current drum contact belongs to its own controller.
                unrelated={frozenset((c.geom1,c.geom2)) for c in data.contact
                    if not any((private.geom(g).name or '').startswith(preset['arm']+'_') for g in (c.geom1,c.geom2))}
                routine.observe(data,unrelated)
            if not routine.failure and routine.completed==1:
                return path,candidate
            last_failure=routine.failure
            if routine.state=='approach':
                if transfer_index==len(scales)-1:break
                transfer_index+=1
            else:
                if strike_index==len(scales)-1:break
                strike_index+=1
        except ValueError as error:
            last_failure=str(error)
            break
    raise ValueError('Simulation preflight failed: '+str(last_failure))


class LiveDemo:
    def __init__(self,model,robot,scene,log_path=OUTPUT/'live-demo-report.json'):
        import mujoco
        self.model,self.robot,self.scene=model,robot,scene
        self.motion=json.loads(CONFIG.with_name('demo_motion.json').read_text())
        self.log_path=Path(log_path)
        self.controller=HoldController(model,robot)
        self.data=mujoco.MjData(model)
        self.controller.reset(self.data)
        self.park=self.controller.target.copy()
        self.pose=self.park.copy()
        store=PoseStore(model,robot,scene,DEMO_POSES)
        self.presets={}
        for drum,arm in DRUM_ARMS.items():
            if store.key(arm,drum) in store.document['presets']:
                self.presets[drum]=store.get(arm,drum)
        self.unavailable=store.document.get('unavailable',{})
        self.telemetry=DrumTelemetry(model,scene)
        self.lanes={arm:{'pending':deque(),'future':None,'cancel':None,'routine':None,'planning':None}
                    for arm in ('left','right')}
        self.executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='drum-transfer')
        self.concurrent_steps=0
        self.reports=[]
        self.notice='Click a drum. The arm moves to ready, strikes with J7, and rebounds.'

    def ready_goal(self, preset, current):
        # Each request changes only its owning arm. The other arm holds here.
        goal=np.array(current,dtype=float).copy()
        selected=slice(0,7) if preset['arm']=='left' else slice(7,14)
        goal[selected]=preset['ready_rad']
        return goal

    @property
    def pending(self):
        return list(self.lanes['left']['pending'])+list(self.lanes['right']['pending'])

    def hit(self,drum):
        if drum not in self.presets:
            self.notice=self.unavailable.get(drum,'No pose for this drum')
            return
        arm=DRUM_ARMS[drum]
        queue=self.lanes[arm]['pending']
        if len(queue)>=8:
            self.notice=f'{arm.title()} queue full; the other arm can still play.'
            return
        queue.append(drum)
        self.notice=f'{LABELS[drum]} queued for {arm} arm.'

    def stop(self):
        for lane in self.lanes.values():
            lane['pending'].clear()
            if lane['cancel']:lane['cancel'].set()
            lane['future']=None
            lane['planning']=None
            routine=lane['routine']
            if routine and not routine.finished:routine.abort('Stopped by user')
            lane['routine']=None
        self.controller.target=self.data.qpos[self.controller.q].copy()
        self.notice='Stopped. Both queues cleared; arms holding their current positions.'

    def step(self):
        c=self.controller
        for arm,lane in self.lanes.items():
            future=lane['future']
            if future is not None and future.done():
                try:
                    planned_path,verified_motion=future.result()
                    path=np.array(planned_path)
                    initial=self.data.qpos[c.q].copy()
                    selected=np.arange(7) if arm=='left' else np.arange(7,14)
                    other=np.arange(7,14) if arm=='left' else np.arange(7)
                    # The other arm may have moved during this background plan.
                    path[:,other]=initial[other]
                    path[0]=initial
                    preset=self.presets[lane['planning']]
                    routine=TaughtStrike(self.model,self.robot,self.scene,preset,initial=initial,
                        approach_path=path,ready_pose=path[-1],transfer_motion=verified_motion,concurrent=True)
                    routine.start_live(self.data)
                    lane['routine']=routine
                    self.notice=f"Moving {arm} arm to {LABELS[lane['planning']]}."
                except (ValueError,RuntimeError) as error:
                    lane['pending'].clear()
                    self.reports.append({'status':'FAIL','arm':arm,'drum':lane['planning'],'failure':str(error)})
                    self.notice=f'{arm.title()} transfer stopped: {error}'
                lane['future']=None
                lane['planning']=None
        # Both controllers read one shared state; combine their torques and
        # integrate the coupled robot exactly once per physics timestep.
        torque=c.kp*(c.target-self.data.qpos[c.q])-c.kd*self.data.qvel[c.v]+self.data.qfrc_bias[c.v]
        active=[]
        for lane in self.lanes.values():
            routine=lane['routine']
            if routine is not None and not routine.finished:
                command=routine.prepare(self.data)
                if command is not None:
                    torque[routine.selected]=command[routine.selected]
                    active.append(routine)
        if len(active)==2:self.concurrent_steps+=1
        self.data.ctrl[c.a]=np.clip(torque,-c.effort,c.effort)
        mujoco.mj_step(self.model,self.data)
        allowed={frozenset((r.tip,r.head)) for r in active}
        for routine in active:routine.observe(self.data,allowed)
        # Physical failures stop both arms: their contact dynamics are coupled.
        failures=[lane['routine'] for lane in self.lanes.values()
                  if lane['routine'] is not None and lane['routine'].failure]
        for arm,lane in self.lanes.items():
            routine=lane['routine']
            if routine is not None and routine.finished:
                report=routine.report()
                self.reports.append(report)
                selected=routine.selected
                c.target[selected]=(routine.ready[selected] if report['status']=='PASS'
                                    else self.data.qpos[c.q[selected]])
                self.notice=(f"{LABELS[routine.target]} hit complete." if report['status']=='PASS'
                             else 'Stopped: '+str(report['failure']))
                lane['routine']=None
                self.save()
        if failures:
            reason=failures[0].failure
            self.stop()
            self.notice='Both arms stopped: '+str(reason)
        self.telemetry.sample(self.data)
        self.pose=self.data.qpos[c.q].copy()
        for arm,lane in self.lanes.items():
            if lane['future'] is None and lane['routine'] is None and lane['pending']:
                lane['planning']=lane['pending'].popleft()
                preset=self.presets[lane['planning']]
                goal=self.ready_goal(preset,self.pose)
                lane['cancel']=threading.Event()
                lane['future']=self.executor.submit(prepare_transfer,self.model,self.robot,self.scene,self.pose.copy(),goal,
                    preset,self.motion,lane['cancel'])
                self.notice=f"Planning {arm}: {LABELS[lane['planning']]}..."
        self.telemetry.update_colors()

    @property
    def busy(self):
        return any(lane['future'] is not None or lane['routine'] is not None or lane['pending']
                   for lane in self.lanes.values())

    def save(self):
        self.log_path.write_text(json.dumps({'actions':self.reports,
            'concurrent_steps':self.concurrent_steps,
            'hit_count':len(self.telemetry.events),'hits':self.telemetry.events},indent=2)+'\n')

    def close(self):
        for lane in self.lanes.values():
            if lane['cancel']:lane['cancel'].set()
        self.executor.shutdown(wait=True,cancel_futures=True)
        self.save()


class DemoControls:
    def __init__(self,session):self.session=session
    def key(self,key):
        if key==glfw.KEY_SPACE:self.session.stop()
        elif glfw.KEY_1<=key<=glfw.KEY_8:self.session.hit(ORDER[key-glfw.KEY_1])


class DemoPanel:
    width=370
    minimum_height=760
    drag_joint=None

    def __init__(self,session):
        self.session=session
        self.regions=[]
    def layout(self,height):
        self.height=max(self.minimum_height,height)
        self.regions=[]
        for i,drum in enumerate(ORDER):
            x=18+(i%2)*172;y=103+(i//2)*83
            self.regions.append((drum,(x,y,x+162,y+68)))
        self.regions.append(('stop',(18,451,352,492)))
    def press(self,x,y):
        for name,(left,top,right,bottom) in self.regions:
            if left<=x<=right and top<=y<=bottom:
                if name=='stop':self.session.stop()
                else:self.session.hit(name)
                break
    def motion(self,x):pass
    def release(self):pass
    def draw(self,height):
        self.layout(height)
        session=self.session
        image=Image.new('RGB',(self.width,self.height),(17,23,34))
        draw=ImageDraw.Draw(image)
        main,muted=(235,242,249),(151,171,194)
        draw.text((18,15),'LIVE DRUM DEMO',font=font(21),fill=main)
        draw.text((18,44),'Click a drum to move and hit it live.',font=font(13),fill=(100,212,200))
        draw.text((18,68),f"Travel cap: {session.motion['transfer_speed_fraction']:.0%} motor speed | J7-only hits",font=font(12),fill=muted)
        for name,rect in self.regions:
            available=name in session.presets or name=='stop'
            active=any((lane['routine'] and lane['routine'].target==name) or lane['planning']==name
                       for lane in session.lanes.values())
            fill=(27,90,78) if active else (28,43,62) if available else (23,29,40)
            if name=='stop':fill=(104,46,57)
            draw.rounded_rectangle(rect,radius=7,fill=fill,outline=(57,77,99))
            label=LABELS.get(name,'Stop / clear queue')
            draw.text((rect[0]+12,rect[1]+11),label,font=font(14),fill=main if available else (105,118,135))
            if name!='stop':
                detail=session.presets[name]['arm']+' arm' if available else 'pedal target' if name=='kick' else 'unavailable'
                draw.text((rect[0]+12,rect[1]+38),detail,font=font(12),fill=muted)
        for i,(arm,lane) in enumerate(session.lanes.items()):
            phase='Checking motion' if lane['future'] is not None else lane['routine'].state.title() if lane['routine'] else 'Idle'
            draw.text((18,510+i*24),f"{arm.title()}: {phase} | queued: {len(lane['pending'])}",font=font(13),fill=(100,212,200))
        y=568
        for line in textwrap.wrap(session.notice,width=45)[:4]:
            draw.text((18,y),line,font=font(12),fill=main);y+=17
        hit=session.telemetry.events[-1] if session.telemetry.events else None
        draw.text((18,653),'LAST HIT',font=font(12),fill=muted)
        if hit:
            color=session.telemetry.color(hit['peak_force_N'])
            draw.text((18,676),f"{hit['peak_force_N']:.1f} N",font=font(27),fill=color)
            draw.text((190,684),f"{hit['normal_impulse_Ns']:.3f} N s",font=font(17),fill=main)
            draw.text((18,716),f"{LABELS[hit['drum']]} | {len(session.telemetry.events)} total hits",font=font(12),fill=muted)
        else:draw.text((18,684),'Waiting for the first hit',font=font(15),fill=main)
        if self.height>810:
            draw.text((18,788),'Each arm has its own queue (8 requests).',font=font(11),fill=muted)
            draw.text((18,807),'Keys 1-8: drums | Space: stop / clear',font=font(11),fill=muted)
        return image
