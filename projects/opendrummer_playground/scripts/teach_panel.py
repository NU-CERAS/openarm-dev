"""Mouse-driven native teaching panel, drawn beside MuJoCo without a web UI."""
import textwrap

import glfw
import numpy as np
from PIL import Image, ImageDraw

from telemetry import LABELS, ORDER, font


class JogPanel:
    width = 370
    minimum_height = 760

    def __init__(self, controls):
        self.controls = controls
        self.drag_joint = None
        self.regions = []
        self.height = self.minimum_height

    @property
    def session(self):
        return self.controls.session

    def layout(self, height):
        self.height = max(self.minimum_height, height)
        self.regions = [('arm_left',(18,65,180,96)),('arm_right',(190,65,352,96)),
                        ('drum_prev',(18,105,52,138)),('drum_next',(318,105,352,138))]
        for i in range(4):
            self.regions.append((f'step_{i}',(18+i*85,151,97+i*85,177)))
        for i in range(7):
            y = 208+i*43
            self.regions.extend([(f'minus_{i}',(43,y-7,68,y+19)),
                                 (f'slider_{i}',(79,y-8,258,y+20)),
                                 (f'plus_{i}',(267,y-7,292,y+19))])
        self.regions.extend([
            ('ready',(18,515,180,549)),('strike',(190,515,352,549)),
            ('load',(18,558,180,588)),('edit',(190,558,352,588)),
            ('test',(18,598,180,636)),('stop',(190,598,352,636))])

    def enabled(self, action):
        running = bool(self.session.routine and not self.session.routine.finished)
        if running:
            return action == 'stop'
        if action == 'stop':
            return False
        if action.startswith(('slider_','minus_','plus_')):
            return not self.session.wrist_only or action.endswith('_6')
        return True

    def invoke(self, action):
        session, controls = self.session, self.controls
        if not self.enabled(action):
            return
        if action.startswith('arm_'):
            arm = action[4:]
            if session.arm != arm:
                controls.key(glfw.KEY_F8)
        elif action.startswith('drum_'):
            offset = 1 if action == 'drum_next' else -1
            drum = ORDER[(ORDER.index(session.drum)+offset) % len(ORDER)]
            session.command({'action':'select','arm':session.arm,'drum':drum})
        elif action.startswith('step_'):
            controls.step_index = int(action[-1])
        elif action.startswith(('minus_','plus_')):
            controls.joint = int(action[-1])
            controls.key(glfw.KEY_LEFT if action.startswith('minus') else glfw.KEY_RIGHT)
        else:
            controls.key({'ready':glfw.KEY_F10,'strike':glfw.KEY_F11,'load':glfw.KEY_HOME,
                          'edit':glfw.KEY_END,'test':glfw.KEY_F12,'stop':glfw.KEY_SPACE}[action])

    def slide(self, joint, x):
        if not self.enabled(f'slider_{joint}'):
            return
        self.controls.joint = joint
        index = int(self.session.selected[joint])
        bounds = np.degrees(self.session.controller.bounds[index])
        value = bounds[0]+float(np.clip((x-79)/(258-79),0,1))*(bounds[1]-bounds[0])
        self.session.command({'action':'jog','joint':joint,'degrees':value})

    def press(self, x, y):
        self.drag_joint = None
        for action, (left,top,right,bottom) in self.regions:
            if left <= x <= right and top <= y <= bottom:
                if not self.enabled(action):
                    return
                if action.startswith('slider_'):
                    self.drag_joint = int(action[-1])
                    self.slide(self.drag_joint,x)
                else:
                    self.invoke(action)
                return

    def motion(self, x):
        if self.drag_joint is not None:
            self.slide(self.drag_joint,x)

    def release(self):
        self.drag_joint = None

    def draw(self, height):
        self.layout(height)
        state = self.session.state()
        image = Image.new('RGB',(self.width,self.height),(17,23,34))
        draw = ImageDraw.Draw(image)
        main,muted = (235,242,249),(151,171,194)
        draw.text((18,15),'POSE TEACHING',font=font(20),fill=main)
        draw.text((18,41),'Physical test running' if state['running'] else 'Pose preview | physics paused',font=font(12),fill=(100,212,200))
        titles = {'arm_left':'LEFT ARM','arm_right':'RIGHT ARM','drum_prev':'<','drum_next':'>',
                  'ready':'1. Save ready pose','strike':'2. Save J7 strike',
                  'load':'Load ready','edit':'Edit ready pose','test':'3. Test one hit','stop':'Stop test'}
        for action,rect in self.regions:
            if action.startswith(('slider_','minus_','plus_')):
                continue
            active = action == 'arm_'+state['arm'] or action == f'step_{self.controls.step_index}'
            fill = (28,85,77) if active or action in ('ready','strike','test') else (29,42,61)
            if action=='stop':fill=(100,45,56)
            color = main if self.enabled(action) else (91,105,125)
            draw.rounded_rectangle(rect,radius=5,fill=fill,outline=(53,72,96))
            text = f'{self.controls.steps[int(action[-1])]:g} deg' if action.startswith('step_') else titles[action]
            box = draw.textbbox((0,0),text,font=font(12))
            draw.text(((rect[0]+rect[2]-(box[2]-box[0]))/2,rect[1]+7),text,font=font(12),fill=color)
        label = LABELS[state['drum']]
        draw.text((68,112),label,font=font(16),fill=main)
        for i in range(7):
            y = 208+i*43
            index = int(self.session.selected[i])
            low,high = np.degrees(self.session.controller.bounds[index])
            value = state['degrees'][i]
            enabled = self.enabled(f'slider_{i}')
            color = (102,211,198) if enabled else (69,82,101)
            draw.text((18,y-2),f'J{i+1}',font=font(13),fill=main if enabled else muted)
            for text,left in (('-',43),('+',267)):
                draw.rounded_rectangle((left,y-7,left+25,y+19),radius=4,fill=(29,42,61))
                draw.text((left+7,y-3),text,font=font(14),fill=main if enabled else muted)
            draw.rounded_rectangle((79,y+3,258,y+9),radius=3,fill=(47,61,80))
            x = 79+(value-low)/(high-low)*(258-79)
            draw.ellipse((x-6,y,x+6,y+12),fill=color)
            draw.text((300,y-2),f'{value:+.1f}',font=font(12),fill=main if enabled else muted)
        draw.text((18,496),'J7 ONLY | arm held' if state['wrist_only'] else 'Angles in degrees | select either arm',font=font(12),fill=muted)
        y = 648
        for line in textwrap.wrap(state['notice'].replace('→','->'),width=48)[:3]:
            draw.text((18,y),line,font=font(12),fill=main);y+=16
        draw.text((18,y+5),f"Tip height {state['height_mm']:.1f} mm | offset {state['radial_mm']:.1f} mm",font=font(11),fill=muted)
        hit = state['hits'][-1] if state['hits'] else None
        text = 'No test hits yet' if not hit else f"{len(state['hits'])} hit(s) | {hit['peak_force_N']:.1f} N | {hit['normal_impulse_Ns']:.3f} N s"
        draw.text((18,y+24),text,font=font(13),fill=(100,212,200))
        if self.height > 810:
            draw.text((18,790),'Drag scene: orbit | right drag: pan',font=font(11),fill=muted)
            draw.text((18,809),'Scroll: zoom | arrow keys: fine jog',font=font(11),fill=muted)
        return image
