"""Live drum-button session: collision-checked travel, then a J7-only hit."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import textwrap

import glfw
import numpy as np
from PIL import Image, ImageDraw

from openarm_drummer.model import CONFIG, OUTPUT, HoldController
from openarm_drummer.taught import PoseStore, TaughtStrike
from openarm_drummer.telemetry import DrumTelemetry, LABELS, ORDER, font
from openarm_drummer.transfers import plan_transfer

DEMO_POSES = CONFIG.with_name('demo_poses.json')


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
        for key,item in store.document['presets'].items():
            self.presets[item['drum']]=store.get(item['arm'],item['drum'])
        self.unavailable=store.document.get('unavailable',{})
        self.telemetry=DrumTelemetry(model,scene)
        self.pending=deque()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='drum-transfer')
        self.future=None
        self.cancel=None
        self.routine=None
        self.planning=None
        self.reports=[]
        self.notice='Click a drum. The arm moves to ready, strikes with J7, and rebounds.'

    def hit(self,drum):
        if drum not in self.presets:
            self.notice=self.unavailable.get(drum,'No provisional pose for this drum')
            return
        if len(self.pending)>=8:
            self.notice='Queue full. Wait for a hit to finish or click Stop / clear.'
            return
        self.pending.append(drum)
        self.notice=f'{LABELS[drum]} queued.'

    def stop(self):
        self.pending.clear()
        if self.cancel:self.cancel.set()
        self.future=None
        self.planning=None
        if self.routine and not self.routine.finished:
            self.routine.abort('Stopped by user')
        self.controller.target=self.data.qpos[self.controller.q].copy()
        self.routine=None
        self.notice='Stopped. Queue cleared; arms holding their current positions.'

    def step(self):
        if self.future is not None and self.future.done():
            try:
                path=self.future.result()
                initial=self.data.qpos[self.controller.q].copy()
                path[0]=initial.tolist()
                preset=self.presets[self.planning]
                goal=self.park.copy()
                selected=slice(0,7) if preset['arm']=='left' else slice(7,14)
                goal[selected]=preset['ready_rad']
                self.routine=TaughtStrike(self.model,self.robot,self.scene,preset,initial=initial,
                                         approach_path=path,ready_pose=goal,transfer_motion=self.motion)
                self.routine.start_live(self.data)
                self.notice=f"Moving {preset['arm']} arm to {LABELS[self.planning]}."
            except (ValueError,RuntimeError) as error:
                self.pending.clear()
                self.reports.append({'status':'FAIL','drum':self.planning,'failure':str(error)})
                self.notice='Transfer stopped: '+str(error)
            self.future=None
            self.planning=None
        if self.routine is not None and not self.routine.finished:
            self.routine.step(self.data)
            if self.routine.finished:
                report=self.routine.report()
                self.reports.append(report)
                self.controller.target=(self.routine.ready.copy() if report['status']=='PASS'
                                        else self.data.qpos[self.controller.q].copy())
                self.notice=(f"{LABELS[self.routine.target]} hit complete." if report['status']=='PASS'
                             else 'Stopped: '+str(report['failure']))
                if report['status']!='PASS':
                    self.pending.clear()
                self.save()
        else:
            self.controller.step(self.data)
        self.telemetry.sample(self.data)
        self.pose=self.data.qpos[self.controller.q].copy()
        if self.future is None and (self.routine is None or self.routine.finished) and self.pending:
            self.planning=self.pending.popleft()
            preset=self.presets[self.planning]
            goal=self.park.copy()
            selected=slice(0,7) if preset['arm']=='left' else slice(7,14)
            goal[selected]=preset['ready_rad']
            self.cancel=threading.Event()
            self.future=self.executor.submit(plan_transfer,self.model,self.controller,self.pose.copy(),goal,self.cancel)
            self.notice=f'Planning a clear path to {LABELS[self.planning]}...'
        self.telemetry.update_colors()

    @property
    def busy(self):
        return self.future is not None or bool(self.routine and not self.routine.finished) or bool(self.pending)

    def save(self):
        self.log_path.write_text(json.dumps({'actions':self.reports,
            'hit_count':len(self.telemetry.events),'hits':self.telemetry.events},indent=2)+'\n')

    def close(self):
        if self.cancel:self.cancel.set()
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
            active=(session.routine and not session.routine.finished and session.routine.target==name) or session.planning==name
            fill=(27,90,78) if active else (28,43,62) if available else (23,29,40)
            if name=='stop':fill=(104,46,57)
            draw.rounded_rectangle(rect,radius=7,fill=fill,outline=(57,77,99))
            label=LABELS.get(name,'Stop / clear queue')
            draw.text((rect[0]+12,rect[1]+11),label,font=font(14),fill=main if available else (105,118,135))
            if name!='stop':
                detail=session.presets[name]['arm']+' arm' if available else 'pedal target' if name=='kick' else 'unavailable'
                draw.text((rect[0]+12,rect[1]+38),detail,font=font(12),fill=muted)
        phase='Ready for a click'
        if session.future is not None:phase='Planning transfer'
        elif session.routine and not session.routine.finished:phase=session.routine.state.title()
        draw.text((18,510),phase+f' | queued: {len(session.pending)}',font=font(14),fill=(100,212,200))
        y=541
        for line in textwrap.wrap(session.notice,width=45)[:4]:
            draw.text((18,y),line,font=font(12),fill=main);y+=17
        if session.pending:
            draw.text((18,620),'Next: '+', '.join(LABELS[d] for d in list(session.pending)[:2]),font=font(11),fill=muted)
        hit=session.telemetry.events[-1] if session.telemetry.events else None
        draw.text((18,653),'LAST HIT',font=font(12),fill=muted)
        if hit:
            color=session.telemetry.color(hit['peak_force_N'])
            draw.text((18,676),f"{hit['peak_force_N']:.1f} N",font=font(27),fill=color)
            draw.text((190,684),f"{hit['normal_impulse_Ns']:.3f} N s",font=font(17),fill=main)
            draw.text((18,716),f"{LABELS[hit['drum']]} | {len(session.telemetry.events)} total hits",font=font(12),fill=muted)
        else:draw.text((18,684),'Waiting for the first hit',font=font(15),fill=main)
        if self.height>810:
            draw.text((18,788),'Click while busy to queue another drum.',font=font(11),fill=muted)
            draw.text((18,807),'Keys 1-8: drums | Space: stop / clear',font=font(11),fill=muted)
        return image
