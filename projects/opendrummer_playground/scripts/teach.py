"""Native MuJoCo pose teaching: paused joint jogging and physical hit tests."""
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.taught import PoseStore, TaughtStrike, PRESETS, DRUM_ARMS
from openarm_drummer.telemetry import DrumTelemetry, LABELS, ORDER


class TeachingSession:
    def __init__(self, model, robot, scene, path=None, demo_path=None, original_path=None):
        self.model,self.robot,self.scene = model,robot,scene
        self.controller = HoldController(model,robot)
        self.data = mujoco.MjData(model)
        self.controller.reset(self.data)
        self.park = self.controller.target.copy()
        self.pose = self.park.copy()
        self.store = PoseStore(model,robot,scene,**({'path':path} if path else {}))
        self.demo_path = demo_path if demo_path is not None else PRESETS.with_name('demo_poses.json')
        self.original_path = original_path if original_path is not None else PRESETS.with_name('original_demo_poses.json')
        self.arm,self.drum = 'left','snare'
        self.wrist_only = False
        self.routine = None
        self.telemetry = DrumTelemetry(model,scene)
        self.notice = 'Select an arm and drum. Jog the arm, then save the ready pose.'

    @property
    def selected(self):
        return np.arange(7) if self.arm=='left' else np.arange(7,14)

    def preview(self):
        self.controller.target = self.pose.copy()
        self.controller.reset(self.data)

    def command(self, message):
        action = message['action']
        if self.routine is not None and not self.routine.finished and action != 'stop':
            raise ValueError('Stop or finish the physical test before editing')
        if action == 'select':
            self.store.key(message['arm'],message['drum'])
            self.drum = message['drum']
            self.arm = DRUM_ARMS.get(self.drum,message['arm'])
            self.wrist_only = False
            self.notice = 'Selection changed. Load its preset or jog a new ready pose.'
        elif action == 'jog':
            index = int(message['joint'])
            angle = float(message['degrees'])*np.pi/180
            if index not in range(7) or (self.wrist_only and index != 6):
                raise ValueError('Only J7 is unlocked. Use End (edit ready) to move the arm.')
            q = self.pose[self.selected].copy(); q[index] = angle
            bounds = self.controller.bounds[self.selected[index]]
            # Accept rounded degree endpoints (e.g. 90° for a 1.570796 rad
            # URDF bound), while retaining the exact model limit internally.
            if bounds[0]-1e-6 <= angle <= bounds[1]+1e-6:
                q[index] = np.clip(angle, *bounds)
            self.store.validate(self.arm,q)
            self.pose[self.selected] = q
            self.preview()
        elif action == 'edit':
            self.wrist_only = False
            self.notice = 'Selected arm unlocked. Save ready again after changing the arm.'
        elif action == 'save_ready':
            # Preview collisions are visible but cannot be saved as a ready pose.
            if any(c.dist < -1e-5 for c in self.data.contact):
                raise ValueError('Ready pose intersects geometry. Raise/move the arm first.')
            surface = self.telemetry.surfaces[self.drum]
            normal = np.array(surface['normal'])
            delta = self.data.site(self.arm+'_stick_tip').xpos-np.array(surface['center'])
            height = float(delta@normal)
            if height < .01 or np.linalg.norm(delta-height*normal) > surface['radius']-.005:
                raise ValueError('Place the stick tip over the selected head with at least 10 mm clearance.')
            self.store.save_ready(self.arm,self.drum,self.pose[self.selected])
            self.wrist_only = True
            self.notice = 'Ready saved. Joints 1–6 locked; lower only J7 to the head, then save strike.'
        elif action == 'save_strike':
            self.store.save_strike(self.arm,self.drum,self.pose[self.selected])
            self.notice = 'Strike angle saved. Test one hit checks the wrist arc and contact physics.'
        elif action == 'publish':
            if self.drum == 'kick':
                raise ValueError('The live demo has no kick pedal controller')
            preset = self.store.get(self.arm,self.drum)
            # Check the saved wrist arc, not the current unsaved preview pose.
            TaughtStrike(self.model,self.robot,self.scene,preset)
            destination = PoseStore(self.model,self.robot,self.scene,self.demo_path)
            # The live UI has one button per drum, so replace its arm assignment too.
            destination.document['presets'] = {
                key: item for key,item in destination.document['presets'].items()
                if item.get('drum') != self.drum}
            destination.document['presets'][self.store.key(self.arm,self.drum)] = preset
            destination.document.get('unavailable', {}).pop(self.drum, None)
            destination.write()
            self.notice = f'{self.arm.title()} {LABELS[self.drum]} saved for live demo. Restart run_demo.py to use it.'
        elif action in ('show_demo', 'show_original'):
            path = self.demo_path if action == 'show_demo' else self.original_path
            source = PoseStore(self.model,self.robot,self.scene,path)
            owner = DRUM_ARMS.get(self.drum)
            matches = [p for p in source.document['presets'].values()
                       if p.get('drum') == self.drum and p.get('arm') == owner]
            if not matches:
                raise ValueError('No '+('demo' if action == 'show_demo' else 'original')+' ready pose for this drum')
            # Match the live demo's last assignment if a file contains duplicate drums.
            item = source.get(matches[-1]['arm'],self.drum,complete=False)
            self.arm = item['arm']
            self.pose[self.selected] = item['ready_rad']
            self.wrist_only = False
            self.preview()
            label = 'Demo ready (restart demo to refresh)' if action == 'show_demo' else 'Original demo ready'
            self.notice = f'{label}: {self.arm} arm. Preview only; Save ready to teach from here.'
        elif action == 'load':
            latest = PoseStore(self.model,self.robot,self.scene,self.store.path)
            item = latest.get(self.arm,self.drum,complete=False)
            self.store = latest
            self.pose[self.selected] = item['ready_rad']
            self.wrist_only = True
            self.preview()
            self.notice = 'Saved ready pose loaded. Jog J7 to adjust the strike angle.'
        elif action == 'test':
            preset = self.store.get(self.arm,self.drum)
            ready = self.pose.copy(); ready[self.selected] = preset['ready_rad']
            routine = TaughtStrike(self.model,self.robot,self.scene,preset,initial=ready)
            routine.reset(self.data)
            self.pose = ready
            self.routine = routine
            self.telemetry = DrumTelemetry(self.model,self.scene)
            self.notice = 'Physical test running: ready → J7 downstroke → rebound.'
        elif action == 'stop':
            if self.routine and not self.routine.finished:
                self.routine.abort('Stopped by user')
                self.routine.save()
            self.pose = self.data.qpos[self.controller.q].copy()
            self.preview()
            self.notice = 'Stopped. Physics paused; load ready to return to the saved pose.'
        else:
            raise ValueError('Unknown teaching command')

    def step(self):
        if self.routine and not self.routine.finished:
            self.routine.step(self.data)
            self.telemetry.sample(self.data)
            if self.routine.finished:
                self.routine.save(); self.telemetry.save()
                self.pose = self.data.qpos[self.controller.q].copy()
                self.notice = ('Test passed. One hit and stable recovery.' if not self.routine.failure
                               else 'Test stopped: '+self.routine.failure)
        self.telemetry.update_colors()

    def state(self):
        c = self.controller
        item = self.store.document['presets'].get(self.store.key(self.arm,self.drum))
        delta = self.data.site(self.arm+'_stick_tip').xpos-np.array(self.telemetry.surfaces[self.drum]['center'])
        n = np.array(self.telemetry.surfaces[self.drum]['normal'])
        height = float(delta@n)
        return {'arm':self.arm,'drum':self.drum,'degrees':np.degrees(self.pose[self.selected]).tolist(),
                'bounds':np.degrees(c.bounds[self.selected]).tolist(),'wrist_only':self.wrist_only,
                'running':bool(self.routine and not self.routine.finished),
                'phase':self.routine.state if self.routine and not self.routine.finished else 'Pose preview · physics paused',
                'notice':self.notice,'saved':item,'presets':list(self.store.document['presets']),
                'height_mm':height*1000,'radial_mm':float(np.linalg.norm(delta-height*n))*1000,
                'contacts':[self.model.geom(x.geom1).name+' / '+self.model.geom(x.geom2).name for x in self.data.contact if x.dist<0],
                'hits':self.telemetry.events,'surfaces':LABELS}



class NativeTeachingControls:
    """Teaching keyboard commands shared by the native mouse panel."""
    steps = (.1, .5, 1., 5.)

    def __init__(self, session):
        self.session = session
        self.joint = 0
        self.step_index = 2

    def key(self, key):
        session = self.session
        if key == glfw.KEY_SPACE:
            session.command({'action': 'stop'})
            return
        if session.routine and not session.routine.finished:
            session.notice = 'Physical test running. Space stops the test; editing is disabled.'
            return
        if key == glfw.KEY_F8:
            arm = 'right' if session.arm == 'left' else 'left'
            drum = next(d for d in ORDER if DRUM_ARMS.get(d) == arm)
            session.command({'action':'select','arm':arm,'drum':drum})
            self.joint = 0
        elif key == glfw.KEY_F9:
            drum = ORDER[(ORDER.index(session.drum)+1) % len(ORDER)]
            session.command({'action': 'select', 'arm': session.arm, 'drum': drum})
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
            self.joint = 6 if session.wrist_only else (self.joint+(1 if key == glfw.KEY_DOWN else -1)) % 7
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            index = int(session.selected[self.joint])
            degrees = np.degrees(session.pose[index])+(1 if key == glfw.KEY_RIGHT else -1)*self.steps[self.step_index]
            degrees = float(np.clip(degrees, *np.degrees(session.controller.bounds[index])))
            session.command({'action': 'jog', 'joint': self.joint, 'degrees': degrees})
        elif key in (glfw.KEY_MINUS, glfw.KEY_EQUAL):
            self.step_index = int(np.clip(self.step_index+(1 if key == glfw.KEY_EQUAL else -1), 0, len(self.steps)-1))
        else:
            action = {glfw.KEY_F10: 'save_ready', glfw.KEY_F11: 'save_strike',
                      glfw.KEY_F12: 'test', glfw.KEY_HOME: 'load', glfw.KEY_END: 'edit'}.get(key)
            if action:
                session.command({'action': action})
                if action in ('save_ready', 'load'):
                    self.joint = 6


def native_app(name):
    import os
    from pathlib import Path
    import sys
    executable_name = 'opendrummer-'+name.lower().replace(' ','-')+'-python'
    # Give the custom GLFW window a real app identity on macOS and run Python
    # on the OS main thread (mjpython's script thread cannot create NSWindows).
    if sys.platform == 'darwin' and Path(sys.executable).name != executable_name:
        import plistlib
        import shutil
        from openarm_drummer.model import OUTPUT
        bundle = OUTPUT/f'OpenDrummer {name}.app'
        contents = bundle/'Contents'
        executable = contents/'MacOS'/executable_name
        executable.parent.mkdir(parents=True, exist_ok=True)
        # Replace the executable inode rather than overwriting a Mach-O binary;
        # macOS can otherwise retain a stale code-signature cache on relaunch.
        temporary=executable.with_suffix('.new')
        shutil.copy2(Path(sys.executable).resolve(), temporary)
        temporary.replace(executable)
        libraries = contents/'lib'
        if not libraries.exists():
            libraries.symlink_to(Path(sys.prefix)/'lib', target_is_directory=True)
        info = {'CFBundleExecutable':executable_name, 'CFBundleIdentifier':'org.opendrummer.'+name.lower().replace(' ','-'),
                'CFBundleName':'OpenDrummer '+name, 'CFBundlePackageType':'APPL',
                'NSHighResolutionCapable':True}
        (contents/'Info.plist').write_bytes(plistlib.dumps(info))
        environment = os.environ.copy()
        environment['PYTHONHOME'] = sys.prefix
        os.execve(str(executable), [str(executable), *sys.argv], environment)


def run_teaching(model, robot, scene):
    from openarm_drummer.teach_panel import JogPanel
    native_app('Teaching')
    session = TeachingSession(model,robot,scene)
    controls = NativeTeachingControls(session)
    panel = JogPanel(controls)
    run_native_window(model,session,panel,controls,'OpenDrummer | Pose teaching')


def run_native_window(model, session, panel, controls, title):
    from PIL import Image
    from openarm_drummer.run import frame_camera
    panel.layout(850)
    if not glfw.init():
        raise RuntimeError('Could not initialize the native teaching window')
    window = glfw.create_window(1320, 850, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError('Could not create the native teaching window')
    glfw.set_window_size_limits(window, 800, 600, glfw.DONT_CARE, glfw.DONT_CARE)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera = mujoco.MjvCamera()
    frame_camera(model, session.data, camera)
    options = mujoco.MjvOption()
    options.geomgroup[4] = 0
    options.sitegroup[3] = 1
    options.sitegroup[5] = 1
    render_scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    backdrop = mujoco.MjvFigure()
    backdrop.figurergba[:] = [17/255, 23/255, 34/255, 1]
    mouse = {'button': None, 'position': (0., 0.)}

    def apply(action):
        try:
            action()
        except (ValueError, OSError) as error:
            session.notice = 'Cannot apply: '+str(error)
            print(session.notice, flush=True)

    def panel_coordinates(x, y):
        width, height = glfw.get_window_size(window)
        return x-(width-panel.width), y*max(panel.minimum_height,height)/max(1,height)

    def key_callback(handle, key, scancode, action, mods):
        if action in (glfw.PRESS, glfw.REPEAT):
            apply(lambda: controls.key(key))

    def button_callback(handle, button, action, mods):
        x,y = glfw.get_cursor_pos(window)
        width,_ = glfw.get_window_size(window)
        if action == glfw.RELEASE:
            mouse['button'] = None
            panel.release()
        elif action == glfw.PRESS:
            if x >= width-panel.width:
                if button == glfw.MOUSE_BUTTON_LEFT:
                    apply(lambda: panel.press(*panel_coordinates(x,y)))
            else:
                mouse['button'] = button
                mouse['position'] = (x,y)

    def cursor_callback(handle, x, y):
        if panel.drag_joint is not None:
            apply(lambda: panel.motion(panel_coordinates(x,y)[0]))
            return
        if mouse['button'] is None:
            return
        dx,dy = x-mouse['position'][0],y-mouse['position'][1]
        mouse['position'] = (x,y)
        _,height = glfw.get_window_size(window)
        shift = glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(window,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS
        if mouse['button'] == glfw.MOUSE_BUTTON_LEFT:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif mouse['button'] == glfw.MOUSE_BUTTON_RIGHT:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(model,action,dx/max(height,1),dy/max(height,1),render_scene,camera)

    def scroll_callback(handle, dx, dy):
        mujoco.mjv_moveCamera(model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*dy,render_scene,camera)

    glfw.set_key_callback(window,key_callback)
    glfw.set_mouse_button_callback(window,button_callback)
    glfw.set_cursor_pos_callback(window,cursor_callback)
    glfw.set_scroll_callback(window,scroll_callback)
    print(title+' | Close the window to exit.',flush=True)
    try:
        while not glfw.window_should_close(window):
            start = time.monotonic()
            glfw.poll_events()
            for _ in range(max(1,round(1/60/model.opt.timestep))):
                session.step()
            width,height = glfw.get_window_size(window)
            frame_width,frame_height = glfw.get_framebuffer_size(window)
            if width <= 0 or height <= 0 or frame_height <= 0:
                glfw.wait_events_timeout(.05)
                continue
            panel_width = round(panel.width*frame_width/width)
            viewport = mujoco.MjrRect(0,0,frame_width-panel_width,frame_height)
            mujoco.mjv_updateScene(model,session.data,options,None,camera,
                                  mujoco.mjtCatBit.mjCAT_ALL,render_scene)
            mujoco.mjr_render(viewport,render_scene,context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,'Left drag: orbit | Right drag: pan | Scroll: zoom\n'
                'Click the panel controls on the right.','',context)
            panel_image = panel.draw(height).resize((panel_width,frame_height),Image.Resampling.LANCZOS)
            pixels = np.ascontiguousarray(np.asarray(panel_image)[::-1])
            rect = mujoco.MjrRect(frame_width-panel_width,0,panel_width,frame_height)
            # Establish 2D state before drawing pixels (mjr_drawPixels inherits
            # the active depth test from the 3D scene otherwise).
            mujoco.mjr_figure(rect,backdrop,context)
            mujoco.mjr_drawPixels(pixels.ravel(),None,rect,context)
            glfw.swap_buffers(window)
            time.sleep(max(0,1/60-(time.monotonic()-start)))
    finally:
        if hasattr(session, 'close'):
            session.close()
        session.telemetry.save()
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
