"""Read-only per-head hit measurement and a native-viewer right sidebar."""
from functools import lru_cache
import json
import time

import mujoco
import numpy as np

from openarm_drummer.model import CONFIG, OUTPUT
from openarm_drummer.strike import HitDetector, STRIKE_CONFIG

LABELS = {'snare': 'Snare', 'floor_tom': 'Floor tom', 'rack_tom_left': 'Rack tom · L',
          'rack_tom_right': 'Rack tom · R', 'cymbal_left': 'Cymbal · L',
          'cymbal_right': 'Cymbal · R', 'hi_hat': 'Hi-hat', 'kick': 'Kick'}
ORDER = ('snare', 'rack_tom_left', 'rack_tom_right', 'floor_tom', 'hi_hat', 'cymbal_left', 'cymbal_right', 'kick')


@lru_cache(maxsize=32)
def font(size):
    from PIL import ImageFont
    for name in ('/System/Library/Fonts/Helvetica.ttc', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


class DrumTelemetry:
    def __init__(self, model, scene_report, clock=time.monotonic):
        self.model, self.clock = model, clock
        self.cfg = json.loads(CONFIG.with_name('telemetry.json').read_text())
        if not 0 < self.cfg['blue_below_N'] < self.cfg['red_at_N'] or self.cfg['flash_seconds'] <= 0:
            raise ValueError('Invalid telemetry force bands or flash duration')
        self.detector_cfg = json.loads(STRIKE_CONFIG.read_text())
        self.surfaces = {s['name']: s for s in scene_report['surfaces']}
        self.heads = {model.geom(name+'_head_collision').id: name for name in self.surfaces}
        self.sticks = {i: model.geom(i).name for i in range(model.ngeom)
                       if model.geom(i).name.startswith(('left_stick_', 'right_stick_'))}
        self.tip_sites = [model.site(side+'_stick_tip').id for side in ('left', 'right')]
        self.flash_sites = {name: model.site(name+'_hit_flash').id for name in self.surfaces}
        self.detectors = {name: HitDetector(self.detector_cfg) for name in self.surfaces}
        self.events = []
        self.latest = {}
        self.flash_start = {}
        self.last_time = 0.
        self.backdrop = mujoco.MjvFigure()
        self.backdrop.figurergba[:] = [17/255, 23/255, 34/255, 1]

    def color(self, force):
        if force < self.cfg['blue_below_N']:
            return (78, 169, 255)
        if force < self.cfg['red_at_N']:
            return (255, 191, 80)
        return (255, 102, 111)

    def sample(self, data):
        """Called exactly once after each physics step, never at display rate."""
        t = float(data.time)
        if t <= self.last_time:
            return
        dt = t-self.last_time
        self.last_time = t
        forces = dict.fromkeys(self.surfaces, 0.)
        parts = {name: set() for name in self.surfaces}
        for i, contact in enumerate(data.contact):
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.heads and second in self.sticks:
                head, stick = first, second
            elif second in self.heads and first in self.sticks:
                head, stick = second, first
            else:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, wrench)
            value = max(0., float(wrench[0]))
            name = self.heads[head]
            forces[name] += value
            if value > self.detector_cfg['release_force_N']:
                parts[name].add(self.sticks[stick])
        for name, surface in self.surfaces.items():
            normal, center = np.array(surface['normal']), np.array(surface['center'])
            gaps = []
            for site in self.tip_sites:
                delta = data.site_xpos[site]-center
                gap = float(delta@normal)
                if np.linalg.norm(delta-gap*normal) <= surface['radius']+.03:
                    gaps.append(gap)
            event = self.detectors[name].update(t, forces[name], min(gaps, default=1.), dt)
            if event is not None:
                event.update(drum=name, hit_number=len(self.detectors[name].events),
                             contact_parts=sorted(parts[name]))
                # Keep the episode object: peak and impulse continue updating
                # until separation, rather than freezing at first contact.
                self.events.append(event)
                self.latest[name] = event
                self.flash_start[name] = self.clock()
            episode = self.detectors[name].episode
            if episode is not None and 'contact_parts' in episode:
                episode['contact_parts'] = sorted(set(episode['contact_parts']) | parts[name])

    def flash_alpha(self, name):
        age = self.clock()-self.flash_start.get(name, -1e9)
        duration = self.cfg['flash_seconds']
        return float(.85*max(0., 1-age/duration)**.6) if age < duration else 0.

    def update_colors(self):
        """Display-only site colors; no mass, contact, or controller changes."""
        for name, site in self.flash_sites.items():
            force = self.latest.get(name, {}).get('peak_force_N', 0.)
            self.model.site_rgba[site] = [*(np.array(self.color(force))/255), self.flash_alpha(name)]

    def save(self):
        payload = {'hit_count': len(self.events), 'hits': self.events,
                   'latest_by_drum': self.latest, 'display_config': self.cfg,
                   'measurement': 'Summed normal stick/head contact force; impulse integrated above release threshold. Same-head simultaneous contacts form one episode.'}
        (OUTPUT/'hit-telemetry.json').write_text(json.dumps(payload, indent=2)+'\n')

    def panel_width(self, width):
        return min(max(int(width*self.cfg['panel_width_fraction']), 280), max(180, width//2))

    def draw_panel(self, width, height, status='Ready'):
        from PIL import Image, ImageDraw
        # Render at a fixed logical width, scaling for framebuffer/Retina size.
        scale = width/330
        h = max(420, int(height/scale))
        image = Image.new('RGB', (330, h), (17, 23, 34))
        draw = ImageDraw.Draw(image)
        main, muted = (235, 242, 249), (148, 166, 186)
        draw.line((0, 0, 0, h), fill=(54, 70, 90), width=2)
        draw.text((22, 19), 'DRUM HITS', font=font(18), fill=main)
        draw.text((22, 44), status[:42], font=font(12), fill=(98, 210, 203))
        last = self.events[-1] if self.events else None
        color = self.color(last['peak_force_N']) if last else muted
        draw.rounded_rectangle((18, 73, 312, 195), radius=10, fill=(27, 36, 50))
        label = f"{LABELS[last['drum']]} · hit {last['hit_number']}" if last else 'Waiting for a hit'
        draw.text((31, 84), label, font=font(14), fill=color)
        peak = f"{last['peak_force_N']:.1f}" if last else '—'
        impulse = f"{last['normal_impulse_Ns']:.3f}" if last else '—'
        draw.text((31, 108), peak, font=font(30), fill=main)
        draw.text((181, 113), impulse, font=font(23), fill=main)
        draw.text((31, 145), 'Peak force · N', font=font(11), fill=muted)
        draw.text((181, 145), 'Impulse · N·s', font=font(11), fill=muted)
        draw.rounded_rectangle((31, 171, 298, 177), radius=3, fill=(47, 59, 76))
        if last:
            bar = min(last['peak_force_N']/self.cfg['force_bar_max_N'], 1)*267
            if bar > 1:
                draw.rounded_rectangle((31, 171, 31+bar, 177), radius=3, fill=color)
        y = 214
        draw.text((22, y), 'SURFACE / HITS', font=font(10), fill=muted)
        draw.text((173, y), 'PEAK N', font=font(10), fill=muted)
        draw.text((241, y), 'N·s', font=font(10), fill=muted)
        row_height = max(22, min(47, (h-310)//8))
        y += 24
        for name in ORDER:
            event = self.latest.get(name)
            count = len(self.detectors[name].events)
            if event:
                active = self.flash_alpha(name)/.85
                tint = np.array(self.color(event['peak_force_N']))
                background = tuple((np.array([17, 23, 34])*(1-active*.18)+tint*active*.18).astype(int))
                draw.rounded_rectangle((14, y-4, 316, y+row_height-7), radius=5, fill=background)
            draw.text((22, y), f'{LABELS[name]}  {count}', font=font(12), fill=main if event else muted)
            draw.text((173, y), f"{event['peak_force_N']:.1f}" if event else '—', font=font(12), fill=self.color(event['peak_force_N']) if event else muted)
            draw.text((241, y), f"{event['normal_impulse_Ns']:.3f}" if event else '—', font=font(12), fill=main if event else muted)
            y += row_height
        footer = h-65
        draw.line((22, footer-12, 308, footer-12), fill=(45, 58, 76))
        draw.text((22, footer), f'{len(self.events)} hits · flashes show peak force', font=font(11), fill=main)
        draw.text((22, footer+18), f"Blue <{self.cfg['blue_below_N']:g} N   Amber <{self.cfg['red_at_N']:g} N   Red ≥{self.cfg['red_at_N']:g} N", font=font(10), fill=muted)
        draw.text((22, footer+35), 'Simulation estimates · not calibrated loudness', font=font(10), fill=muted)
        return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS))

    def present(self, viewer, status='Ready'):
        viewport = viewer.viewport
        if viewport is None or viewport.width < 100 or viewport.height < 100:
            return
        width = self.panel_width(viewport.width)
        panel = self.draw_panel(width, viewport.height, status)
        rect = mujoco.MjrRect(viewport.left+viewport.width-width, viewport.bottom, width, viewport.height)
        # MuJoCo 3.6 image overlays inherit the scene's depth test. A native
        # figure establishes 2D rendering first so foreground geometry cannot
        # obscure the sidebar. The RGB panel covers the figure completely.
        viewer.set_figures((rect, self.backdrop))
        viewer.set_images((rect, panel))

    def offset_camera(self, viewer):
        """Frame the scene left of the opaque sidebar on initial launch."""
        viewport = viewer.viewport
        if viewport is None or viewport.height <= 0:
            return
        width = self.panel_width(viewport.width)
        azimuth = np.deg2rad(viewer.cam.azimuth)
        right = np.array([np.sin(azimuth), -np.cos(azimuth), 0.])
        shift = viewer.cam.distance*np.tan(np.deg2rad(self.model.vis.global_.fovy)/2)*width/viewport.height
        viewer.cam.lookat[:] += right*shift
