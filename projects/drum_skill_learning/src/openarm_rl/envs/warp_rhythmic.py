import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco_warp as mjw
import numpy as np
from openarm_drummer.strike import STRIKE_CONFIG
import torch
import warp as wp
from stable_baselines3.common.vec_env import VecEnv

from ..music.corpus import read_manifest
from ..music.score import CONFIRMATION_GRACE_S, MATCH_HORIZON_S, MATCH_TOLERANCE_S
from ..skills.registry import SKILLS
from .rhythmic import RhythmicDrummingEnv
from .warp_pickup import simulation_stream


PAIRS = 2*len(SKILLS)
HISTORY = 4
LOOKAHEAD = 8


@wp.kernel
def servo_torque(applied: wp.array2d[float], velocity: wp.array2d[float], gain: wp.array[float],
                 qpos: wp.array2d[float], qvel: wp.array2d[float], qids: wp.array[int], vids: wp.array[int],
                 aids: wp.array[int], kp: wp.array[float], kd: wp.array[float], effort: wp.array[float],
                 control: wp.array2d[float]):
    world, joint = wp.tid()
    torque = (kp[joint]*gain[world]*(applied[world, joint]-qpos[world, qids[joint]])
              + kd[joint]*wp.sqrt(gain[world])*(velocity[world, joint]-qvel[world, vids[joint]]))
    control[world, aids[joint]] = wp.clamp(torque, -effort[joint], effort[joint])


@wp.kernel
def clear_forces(ticks: wp.array[int], force: wp.array2d[float]):
    world, pair = wp.tid()
    force[world, pair] = 0.
    if pair == 0:
        ticks[world] = ticks[world]+1


@wp.kernel
def scan_contacts(count: wp.array[int], geoms: wp.array[wp.vec2i], worlds: wp.array[int], distances: wp.array[float],
                  addresses: wp.array2d[int], efc_force: wp.array2d[float], tip_arm: wp.array[int], head: wp.array[int],
                  force_stop: float, penetration_stop: float, targets: int,
                  force: wp.array2d[float], peak: wp.array[float], penetration: wp.array[float], unsafe: wp.array[int]):
    index = wp.tid()
    if index >= count[0]:
        return
    world = worlds[index]
    pair = geoms[index]
    normal = float(0.)
    for row in range(addresses.shape[1]):
        address = addresses[index, row]
        if address >= 0:
            normal += efc_force[world, address]
    normal = wp.max(normal, 0.)
    arm = wp.max(tip_arm[pair[0]], tip_arm[pair[1]])
    drum = wp.max(head[pair[0]], head[pair[1]])
    depth = -distances[index]
    if arm >= 0 and drum >= 0:
        wp.atomic_add(force, world, arm*targets+drum, normal)
        wp.atomic_max(peak, world, normal)
        wp.atomic_max(penetration, world, depth)
        if normal > force_stop or depth > penetration_stop:
            wp.atomic_max(unsafe, world, 1)
    elif normal > .1:
        wp.atomic_max(unsafe, world, 2)


@wp.kernel
def detect_hits(force: wp.array2d[float], sites: wp.array2d[wp.vec3], tip_sites: wp.array[int], centers: wp.array2d[wp.vec3],
                normals: wp.array[wp.vec3], ticks: wp.array[int], started: wp.array[float], preroll: wp.array[float],
                clips: wp.array[int], first: wp.array[int], last: wp.array[int], goal_time: wp.array2d[float],
                goal_target: wp.array2d[int], unsafe: wp.array[int], frozen: wp.array[int],
                release: float, hit_force: float, hit_impulse: float, rearm_gap: float, rearm_time: float,
                tolerance: float, dt: float, targets: int,
                armed: wp.array2d[int], clear_time: wp.array2d[float], episode: wp.array2d[int], onset: wp.array2d[float],
                impulse: wp.array2d[float], duration: wp.array2d[float], longest: wp.array[float],
                matched: wp.array2d[int], missed: wp.array2d[int], match_error: wp.array2d[float],
                correct: wp.array[int], actual: wp.array[int]):
    world, pair = wp.tid()
    if frozen[world] != 0:
        return
    arm = pair/targets
    drum = pair-arm*targets
    value = force[world, pair]
    time_s = started[world]+float(ticks[world])*dt-preroll[world]
    tip = sites[world, tip_sites[arm]]
    gap = wp.dot(tip-centers[world, drum], normals[drum])
    if value > release:
        clear_time[world, pair] = 0.
        if episode[world, pair] == 0:
            episode[world, pair] = 1
            onset[world, pair] = time_s
            impulse[world, pair] = 0.
            duration[world, pair] = 0.
        impulse[world, pair] = impulse[world, pair]+value*dt
        duration[world, pair] = duration[world, pair]+dt
        if armed[world, pair] != 0 and value >= hit_force and impulse[world, pair] >= hit_impulse:
            armed[world, pair] = 0
            wp.atomic_add(actual, world, 1)
            start = onset[world, pair]
            clip = clips[world]
            best = int(-1)
            best_error = float(1e9)
            for goal in range(first[world], last[world]):
                error = wp.abs(goal_time[clip, goal]-start)
                if (goal_target[clip, goal] == drum and error <= tolerance and error < best_error
                        and matched[world, goal] == 0 and missed[world, goal] == 0):
                    best, best_error = goal, error
            if best >= 0 and wp.atomic_add(matched, world, best, 1) == 0:
                match_error[world, best] = start-goal_time[clip, best]
                wp.atomic_add(correct, world, 1)
    elif gap >= rearm_gap:
        clear_time[world, pair] = clear_time[world, pair]+dt
        if clear_time[world, pair] >= rearm_time:
            armed[world, pair] = 1
            episode[world, pair] = 0
    else:
        clear_time[world, pair] = 0.
    if episode[world, pair] != 0:
        wp.atomic_max(longest, world, duration[world, pair])


@wp.kernel
def finish_substep(force: wp.array2d[float], control: wp.array2d[float], qpos: wp.array2d[float], target: wp.array2d[float],
                   qids: wp.array[int], aids: wp.array[int], force_stop: float, dt: float, period: float,
                   unsafe: wp.array[int], frozen: wp.array[int], peak: wp.array[float],
                   force_cost: wp.array[float], energy: wp.array[float]):
    world = wp.tid()
    if frozen[world] != 0:
        return
    cost = float(0.)
    for pair in range(force.shape[1]):
        value = force[world, pair]
        peak[world] = wp.max(peak[world], value)
        cost += (value/force_stop)*(value/force_stop)
        if value > force_stop:
            unsafe[world] = wp.max(unsafe[world], 3)
    force_cost[world] = force_cost[world]+cost*dt/period
    effort = float(0.)
    for joint in range(qids.shape[0]):
        if wp.abs(target[world, joint]-qpos[world, qids[joint]]) > .2:
            unsafe[world] = wp.max(unsafe[world], 4)
        effort += control[world, aids[joint]]*control[world, aids[joint]]
    energy[world] = energy[world]+effort*dt
    if unsafe[world] != 0:
        frozen[world] = 1


class WarpRhythmicVecEnv(VecEnv):
    """Batched score-only drumming on MuJoCo Warp; RhythmicDrummingEnv remains the semantic reference."""
    render_mode = None
    period = .01

    def __init__(self, num_envs, scene_directory, profiles, midi_files=None, corpus_manifest=None, corpus_split="train",
                 mapping=None, tempo_scale=1., domain_range=0., position_range=0., segment_seconds=None,
                 force_penalty=.05, seed=0, contacts_per_world=24, constraints_per_world=128, sync_forward=True):
        if (midi_files is None) == (corpus_manifest is None):
            raise ValueError("Select MIDI files or one corpus manifest")
        if not 0 <= domain_range <= .2 or not 0 <= position_range <= .05:
            raise ValueError("Invalid rhythmic domain randomization ranges")
        self.domain_range, self.position_range, self.segment_seconds = domain_range, position_range, segment_seconds
        self.force_penalty, self.sync_forward = float(force_penalty), sync_forward
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.cfg = json.loads(STRIKE_CONFIG.read_text())
        options = {"scene_directory": scene_directory, "profiles": profiles, "segment_seconds": segment_seconds, "score_only": True}
        if corpus_manifest is not None:
            document = read_manifest(corpus_manifest)
            parent = Path(corpus_manifest).parent
            midi_files = [parent/clip["midi"] for clip in document["clips"] if clip["split"] == corpus_split]
            mapping = parent/document["mapping"]
        self._load_clips(midi_files, {**options, "mapping": mapping, "tempo_scale": tempo_scale})
        wp.init()
        torch.cuda.synchronize()
        self.stream = wp.Stream(f"cuda:{torch.cuda.current_device()}")
        self.torch_stream = wp.stream_to_torch(self.stream)
        observation = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (269,), np.float32),
                                   "critic": spaces.Box(-np.inf, np.inf, (344,), np.float32),
                                   "nominal": spaces.Box(-10, 10, (14,), np.float32)})
        super().__init__(num_envs, observation, spaces.Box(-1, 1, (14,), np.float32))
        self._initialize_gpu(contacts_per_world, constraints_per_world)

    def _load_clips(self, midi_files, options):
        source, rows = None, []
        for midi in midi_files:
            if source is not None:
                source.restore_nominal()
            env = RhythmicDrummingEnv(**options, midi_file=midi, source=source)
            source = source or env
            starts = env.settled_starts() if self.segment_seconds is not None else [0.]
            rows.append({"env": env, "starts": starts})
        self.reference_env = source
        self.model, self.controller, self.targets = source.model, source.controller, source.targets
        self.clip_sources = [str(midi) for midi in midi_files]
        size = max(len(row["env"].score.goals) for row in rows)
        self.goal_time = np.full((len(rows), size), 1e9, np.float32)
        self.goal_target = np.full((len(rows), size), -1, np.int32)
        self.goal_velocity = np.zeros((len(rows), size), np.float32)
        self.clip_goals, bank = np.zeros(len(rows), np.int32), []
        self.clip_preroll, self.clip_duration = np.zeros(len(rows), np.float32), np.zeros(len(rows), np.float32)
        for index, row in enumerate(rows):
            env = row["env"]
            goals = env.score.goals
            self.clip_goals[index] = len(goals)
            self.goal_time[index, :len(goals)] = [goal.time_s for goal in goals]
            self.goal_target[index, :len(goals)] = [env.target_index[goal.target] for goal in goals]
            self.goal_velocity[index, :len(goals)] = [goal.velocity/127 for goal in goals]
            self.clip_preroll[index], self.clip_duration[index] = env.reference.preroll, env.reference.duration
            for start in row["starts"]:
                positions, velocities = env.reference.sample(start)
                end = env.reference.duration if self.segment_seconds is None else min(env.reference.duration, start+self.segment_seconds)
                window = [goal.event_id for goal in goals if start < goal.time_s+env.reference.preroll <= end-MATCH_HORIZON_S]
                bank.append((index, start, end, window[0] if window else 0, window[-1]+1 if window else 0, *positions, *velocities))
        self.bank = np.asarray(bank)

    def _tensor(self, values, dtype=torch.float32):
        return torch.as_tensor(values, dtype=dtype, device="cuda")

    @simulation_stream
    def _initialize_gpu(self, contacts_per_world, constraints_per_world):
        worlds, model, c = self.num_envs, self.model, self.controller
        self.reference_env.restore_nominal()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.wmodel = mjw.put_model(model)
        self.data = mjw.put_data(model, data, nworld=worlds, nconmax=contacts_per_world, njmax=constraints_per_world)
        self.gravity_model = mjw.put_model(model)
        self.gravity_data = mjw.put_data(model, data, nworld=worlds, nconmax=1, njmax=1)
        self.batched = {}
        for name, source, dtype in (("body_mass", model.body_mass, float), ("body_inertia", model.body_inertia, wp.vec3),
                                    ("geom_friction", model.geom_friction, wp.vec3), ("geom_pos", model.geom_pos, wp.vec3),
                                    ("site_pos", model.site_pos, wp.vec3), ("dof_invweight0", model.dof_invweight0, float),
                                    ("body_invweight0", model.body_invweight0, wp.vec2)):
            array = wp.array(np.tile(source, (worlds, *([1]*source.ndim))), dtype=dtype, device="cuda")
            setattr(self.wmodel, name, array)
            self.batched[name] = wp.to_torch(array)
        self.nominal = {name: value[0].clone() for name, value in self.batched.items()}
        env = self.reference_env
        self.kit_geoms, self.kit_sites = self._tensor(env.kit_geoms, torch.long), self._tensor(env.kit_sites, torch.long)
        self.qids, self.vids, self.aids = (self._tensor(ids, torch.long) for ids in (c.q, c.v, c.a))
        self.qids_wp, self.vids_wp, self.aids_wp = (wp.array(ids.astype(np.int32), dtype=int, device="cuda") for ids in (c.q, c.v, c.a))
        self.kp, self.kd, self.effort = self._tensor(c.kp), self._tensor(c.kd), self._tensor(c.effort)
        self.bounds, self.limit = self._tensor(c.bounds), self._tensor(.8*c.velocity)
        self.centers0, self.normals = self._tensor(env.centers), self._tensor(env.normals)
        self.qpos, self.qvel = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel)
        self.geom_xpos = wp.to_torch(self.data.geom_xpos)
        self.kit_xpos = self._tensor(data.geom_xpos[env.kit_geoms])
        self.sites = wp.to_torch(self.data.site_xpos)
        self.gravity_qpos, self.gravity_bias = wp.to_torch(self.gravity_data.qpos), wp.to_torch(self.gravity_data.qfrc_bias)
        self.tip_site_ids = self._tensor(list(env.tip_sites.values()), torch.long)
        self.goal_time_t, self.goal_target_t = self._tensor(self.goal_time), self._tensor(self.goal_target, torch.int32)
        self.goal_velocity_t, self.bank_t = self._tensor(self.goal_velocity), self._tensor(self.bank)
        self.preroll_t, self.duration_t = self._tensor(self.clip_preroll), self._tensor(self.clip_duration)
        floats = ("started", "preroll", "end", "peak", "penetration", "force_cost", "energy", "longest", "gain", "episode_return")
        ints = ("ticks", "unsafe", "frozen", "clip", "first", "last", "correct", "actual", "steps", "feedback_period", "feedback_delay")
        self.state = {name: torch.zeros(worlds, device="cuda") for name in floats}
        self.state |= {name: torch.zeros(worlds, dtype=torch.int32, device="cuda") for name in ints}
        pairs = (worlds, PAIRS)
        self.detector = {name: torch.zeros(pairs, device="cuda") for name in ("force", "clear_time", "onset", "impulse", "duration")}
        self.detector |= {name: torch.zeros(pairs, dtype=torch.int32, device="cuda") for name in ("armed", "episode")}
        goals = (worlds, self.goal_time.shape[1])
        self.matched, self.missed = (torch.zeros(goals, dtype=torch.int32, device="cuda") for _ in range(2))
        self.match_error = torch.zeros(goals, device="cuda")
        self.centers = self.centers0.repeat(worlds, 1, 1).contiguous()
        self.shift = torch.zeros((worlds, 3), device="cuda")
        self.parameters = torch.ones((worlds, 3), device="cuda")
        self.applied, self.velocity, self.target = (torch.zeros((worlds, 14), device="cuda") for _ in range(3))
        self.previous_target, self.previous_velocity = torch.zeros((worlds, 14), device="cuda"), torch.zeros((worlds, 14), device="cuda")
        self.last_action, self.anchor = torch.zeros((worlds, 14), device="cuda"), torch.zeros((worlds, 14), device="cuda")
        self.history = torch.zeros((worlds, HISTORY, 42), device="cuda")
        self.samples = torch.zeros((worlds, 4, 29), device="cuda")
        self.cache = torch.zeros((worlds, 29), device="cuda")
        roles_tip, roles_head = -np.ones(self.model.ngeom, np.int32), -np.ones(self.model.ngeom, np.int32)
        for geom, arm in env.tips.items():
            roles_tip[geom] = 0 if arm == "left" else 1
        for geom, target in env.heads.items():
            roles_head[geom] = env.target_index[target]
        self.tip_arm, self.head = (wp.array(values, dtype=int, device="cuda") for values in (roles_tip, roles_head))
        wrap = lambda value, dtype=None: wp.from_torch(value, dtype=dtype) if dtype else wp.from_torch(value)
        self.w = {name: wrap(value) for name, value in {**self.state, **self.detector}.items()}
        self.w.update(applied=wrap(self.applied), velocity=wrap(self.velocity), target=wrap(self.target), matched=wrap(self.matched),
                      missed=wrap(self.missed), match_error=wrap(self.match_error), goal_time=wrap(self.goal_time_t),
                      goal_target=wrap(self.goal_target_t), centers=wrap(self.centers, wp.vec3), normals=wrap(self.normals, wp.vec3),
                      tip_sites=wp.array(self.tip_site_ids.cpu().numpy().astype(np.int32), dtype=int, device="cuda"),
                      kp=wrap(self.kp), kd=wrap(self.kd), effort=wrap(self.effort))
        self._reset_worlds(torch.arange(worlds, device="cuda"))
        self._gravity()
        self._physics()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(round(self.period/model.opt.timestep)):
                self._physics()
        self.graph = capture.graph
        with wp.ScopedCapture() as capture:
            self._gravity()
        self.gravity_graph = capture.graph
        self.reset()

    def _gravity(self):
        mjw.kinematics(self.gravity_model, self.gravity_data)
        mjw.com_pos(self.gravity_model, self.gravity_data)
        mjw.com_vel(self.gravity_model, self.gravity_data)
        mjw.rne(self.gravity_model, self.gravity_data)

    def _physics(self):
        w, data, cfg, dt = self.w, self.data, self.cfg, float(self.model.opt.timestep)
        worlds = self.num_envs
        wp.launch(servo_torque, dim=(worlds, 14),
                  inputs=[w["applied"], w["velocity"], w["gain"], data.qpos, data.qvel, self.qids_wp, self.vids_wp, self.aids_wp,
                          w["kp"], w["kd"], w["effort"]], outputs=[data.ctrl])
        mjw.step(self.wmodel, data)
        if self.sync_forward:
            mjw.forward(self.wmodel, data)
        wp.launch(clear_forces, dim=(worlds, PAIRS), inputs=[w["ticks"]], outputs=[w["force"]])
        wp.launch(scan_contacts, dim=data.naconmax,
                  inputs=[data.nacon, data.contact.geom, data.contact.worldid, data.contact.dist, data.contact.efc_address,
                          data.efc.force, self.tip_arm, self.head, float(cfg["force_stop_N"]), float(cfg["penetration_stop_m"]),
                          len(self.targets)], outputs=[w["force"], w["peak"], w["penetration"], w["unsafe"]])
        wp.launch(detect_hits, dim=(worlds, PAIRS),
                  inputs=[w["force"], data.site_xpos, w["tip_sites"], w["centers"], w["normals"], w["ticks"], w["started"], w["preroll"],
                          w["clip"], w["first"], w["last"], w["goal_time"], w["goal_target"], w["unsafe"], w["frozen"],
                          float(cfg["release_force_N"]), float(cfg["hit_force_N"]), float(cfg["hit_impulse_Ns"]),
                          float(cfg["rearm_clearance_m"]), float(cfg["rearm_seconds"]), MATCH_TOLERANCE_S, dt, len(self.targets)],
                  outputs=[w["armed"], w["clear_time"], w["episode"], w["onset"], w["impulse"], w["duration"], w["longest"],
                           w["matched"], w["missed"], w["match_error"], w["correct"], w["actual"]])
        wp.launch(finish_substep, dim=worlds,
                  inputs=[w["force"], data.ctrl, data.qpos, w["target"], self.qids_wp, self.aids_wp, float(cfg["force_stop_N"]), dt, self.period],
                  outputs=[w["unsafe"], w["frozen"], w["peak"], w["force_cost"], w["energy"]])

    def _reset_worlds(self, ids, rows=None, parameters=None, shift=None):
        count = len(ids)
        if count == 0:
            return
        if rows is None:
            rows = torch.randint(len(self.bank_t), (count,), device="cuda", generator=self.generator)
        row = self.bank_t[rows]
        uniform = lambda low, high, size: low+(high-low)*torch.rand(size, device="cuda", generator=self.generator)
        if parameters is None:
            parameters = uniform(1-self.domain_range, 1+self.domain_range, (count, 3))
        if shift is None:
            shift = torch.zeros((count, 3), device="cuda")
            shift[:, :2] = uniform(-self.position_range, self.position_range, (count, 2))
        s = self.state
        clip = row[:, 0].long()
        self.parameters[ids], self.shift[ids] = parameters, shift
        mass = parameters[:, 0]
        b = self.batched
        # Armature keeps these constants from scaling exactly with mass; the error stays below the float32 parity bound.
        b["body_mass"][ids] = self.nominal["body_mass"]*mass[:, None]
        b["body_inertia"][ids] = self.nominal["body_inertia"]*mass[:, None, None]
        b["body_invweight0"][ids] = self.nominal["body_invweight0"]/mass[:, None, None]
        b["dof_invweight0"][ids] = self.nominal["dof_invweight0"]/mass[:, None]
        b["geom_friction"][ids] = self.nominal["geom_friction"]*parameters[:, 1, None, None]
        geom_pos, site_pos = self.nominal["geom_pos"].repeat(count, 1, 1), self.nominal["site_pos"].repeat(count, 1, 1)
        geom_pos[:, self.kit_geoms] += shift[:, None]
        site_pos[:, self.kit_sites] += shift[:, None]
        b["geom_pos"][ids], b["site_pos"][ids] = geom_pos, site_pos
        self.centers[ids] = self.centers0+shift[:, None]
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda")
        mask[ids] = True
        mjw.reset_data(self.wmodel, self.data, wp.from_torch(mask))
        # MuJoCo Warp computes world-attached geom poses once, so moving the kit needs its world positions written here.
        self.geom_xpos[ids[:, None], self.kit_geoms] = self.kit_xpos+shift[:, None]
        q, dq = row[:, 5:19], row[:, 19:33]
        noisy = q+uniform(-.003, .003, (count, 14))*self.domain_range/.1
        noisy = torch.minimum(torch.maximum(noisy, self.bounds[:, 0]), self.bounds[:, 1])
        self.qpos[ids[:, None], self.qids] = noisy
        self.qvel[ids[:, None], self.vids] = dq
        mjw.forward(self.wmodel, self.data)
        for name in ("ticks", "unsafe", "frozen", "correct", "actual", "steps"):
            s[name][ids] = 0
        for name in ("peak", "penetration", "force_cost", "energy", "longest", "episode_return"):
            s[name][ids] = 0
        s["started"][ids], s["end"][ids], s["clip"][ids] = row[:, 1], row[:, 2], clip.int()
        s["preroll"][ids], s["first"][ids], s["last"][ids] = self.preroll_t[clip], row[:, 3].int(), row[:, 4].int()
        s["gain"][ids] = parameters[:, 2]
        self.detector["armed"][ids] = 1
        for name in ("episode", "force", "clear_time", "onset", "impulse", "duration"):
            self.detector[name][ids] = 0
        self.matched[ids], self.missed[ids], self.match_error[ids] = 0, 0, 0
        random = self.domain_range > 0
        s["feedback_period"][ids] = torch.randint(1, 3, (count,), device="cuda", dtype=torch.int32, generator=self.generator) if random else 1
        s["feedback_delay"][ids] = torch.randint(0, 3, (count,), device="cuda", dtype=torch.int32, generator=self.generator) if random else 0
        self.last_action[ids] = self.encode(q)
        self.anchor[ids] = noisy
        self.previous_target[ids], self.previous_velocity[ids] = q, dq
        sample = torch.cat((row[:, 1:2], noisy, dq), dim=1)
        self.samples[ids] = sample[:, None]
        self.cache[ids] = sample
        self.history[ids] = torch.cat((noisy, dq, self.last_action[ids]), dim=1)[:, None]

    def encode(self, positions):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return torch.clamp(2*(positions-low)/(high-low)-1, -.999, .999)

    def elapsed(self):
        return self.state["started"]+self.state["ticks"].float()*float(self.model.opt.timestep)

    def encoders(self):
        q, dq = self.cache[:, 1:15], self.cache[:, 15:29]
        return q+dq*(self.elapsed()-self.cache[:, 0])[:, None], dq

    def future_goals(self, score_time):
        clip = self.state["clip"].long()
        times = self.goal_time_t[clip]
        first = torch.sum(times < (score_time-MATCH_TOLERANCE_S)[:, None], dim=1)
        index = first[:, None]+torch.arange(LOOKAHEAD, device="cuda")
        valid = index < self._tensor(self.clip_goals, torch.long)[clip][:, None]
        index = index.clamp(max=times.shape[1]-1)
        rows = torch.zeros((self.num_envs, LOOKAHEAD, len(self.targets)+3), device="cuda")
        drum = self.goal_target_t[clip].gather(1, index).long().clamp(min=0)
        rows.scatter_(2, drum[:, :, None], valid[:, :, None].float())
        rows[:, :, -3] = torch.where(valid, times.gather(1, index)-score_time[:, None], 0.)
        rows[:, :, -2] = torch.where(valid, self.goal_velocity_t[clip].gather(1, index), 0.)
        rows[:, :, -1] = valid.float()
        return rows.flatten(1)

    def observation(self):
        elapsed = self.elapsed()
        centers = self.centers0.flatten().expand(self.num_envs, -1)
        actor = torch.cat((self.history.flatten(1), centers, self.future_goals(elapsed-self.state["preroll"])), dim=1)
        tips = self.sites[:, self.tip_site_ids].flatten(1)
        critic = torch.cat((actor, self.qpos[:, self.qids], self.qvel[:, self.vids], self.detector["force"], tips,
                            self.centers.flatten(1), self.parameters, self.shift[:, :2], elapsed[:, None]), dim=1)
        return {"actor": actor, "critic": critic, "nominal": torch.atanh(self.encode(self.anchor))}

    def _numpy(self, observation):
        return {name: value.cpu().numpy() for name, value in observation.items()}

    def metrics(self):
        s = self.state
        tp, actual = s["correct"].float(), s["actual"].float()
        count = (s["last"]-s["first"]).float()
        precision, recall = tp/actual.clamp(min=1), tp/count.clamp(min=1)
        f1 = 2*precision*recall/(precision+recall).clamp(min=1e-12)
        speed = torch.abs(self.qvel[:, self.vids]).amax(dim=1)
        recovered = ((speed < .08) & (self.detector["armed"] != 0).all(dim=1)
                     & (self.detector["force"].amax(dim=1) <= self.cfg["release_force_N"]))
        full = s["end"] == self.duration_t[s["clip"].long()]
        success = (f1 >= .95) & (s["unsafe"] == 0) & (s["longest"] <= .12) & (~full | recovered)
        return {"f1": f1, "precision": precision, "recall": recall, "recovered": recovered, "is_success": success}

    def control(self, actions):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        desired = low+(actions+1)*.5*(high-low)
        velocity = torch.minimum(torch.maximum((desired-self.previous_target)/self.period, -self.limit), self.limit)
        velocity = torch.minimum(torch.maximum(velocity, self.previous_velocity-120*self.period), self.previous_velocity+120*self.period)
        target = torch.minimum(torch.maximum(self.previous_target+velocity*self.period, low), high)
        measured, _ = self.encoders()
        self.gravity_qpos[:, self.qids] = measured
        wp.capture_launch(self.gravity_graph)
        self.applied.copy_(target+self.gravity_bias[:, self.vids]/self.kp)
        self.velocity.copy_(velocity)
        self.target.copy_(target)
        self.previous_target.copy_(target)
        self.previous_velocity.copy_(velocity)

    def advance(self, actions):
        """One control step for every world; returns rewards and the terminal mask."""
        s = self.state
        previous_action = self.last_action.clone()
        self.last_action.copy_(actions)
        self.control(actions)
        before_correct, before_extra = s["correct"].clone(), s["actual"]-s["correct"]
        s["force_cost"].zero_()
        s["energy"].zero_()
        wp.capture_launch(self.graph)
        if self.data.nacon.numpy()[0] > self.data.naconmax or bool(wp.to_torch(self.data.overflow).any().item()):
            raise RuntimeError("MuJoCo Warp contact or constraint capacity exceeded")
        elapsed = self.elapsed()
        score_time = elapsed-s["preroll"]
        clip = s["clip"].long()
        index = torch.arange(self.goal_time.shape[1], device="cuda")
        times = self.goal_time_t[clip]
        window = (index >= s["first"][:, None]) & (index < s["last"][:, None])
        open_goals = window & (self.matched == 0) & (self.missed == 0)
        expired = open_goals & (times+MATCH_TOLERANCE_S+CONFIRMATION_GRACE_S < score_time[:, None])
        self.missed[expired] = 1
        missed = expired.sum(dim=1).float()
        correct, extra = s["correct"]-before_correct, (s["actual"]-s["correct"])-before_extra
        reward = correct.float()-.5*extra.float()-2*missed-self.force_penalty*s["force_cost"]
        active = window & (torch.abs(times-score_time[:, None]) <= MATCH_TOLERANCE_S)
        tips = self.sites[:, self.tip_site_ids]
        drums = self.goal_target_t[clip].long().clamp(min=0)
        centers = self.centers.gather(1, drums[:, :, None].expand(-1, -1, 3))
        distance = torch.linalg.vector_norm(tips[:, None]-centers[:, :, None], dim=3).amin(dim=2)
        reward = reward-(distance*active).sum(dim=1)
        reward = reward-.001*((actions-previous_action)**2).sum(dim=1)-1e-6*s["energy"]
        long_contact = ((self.detector["episode"] != 0) & (self.detector["duration"] > .12)).sum(dim=1).float()
        reward = reward-.2*long_contact
        unsafe = s["unsafe"] != 0
        remaining = (window & (self.matched == 0) & (self.missed == 0)).sum(dim=1).float()
        reward = reward-torch.where(unsafe, 10+2*remaining, 0.)
        self.missed[unsafe[:, None] & window & (self.matched == 0)] = 1
        s["steps"] += 1
        s["episode_return"] += reward
        sample = torch.cat((elapsed[:, None], self.qpos[:, self.qids], self.qvel[:, self.vids]), dim=1)
        self.samples = torch.cat((self.samples[:, 1:], sample[:, None]), dim=1)
        refresh = s["steps"] % s["feedback_period"] == 0
        selected = self.samples[torch.arange(self.num_envs, device="cuda"), 3-s["feedback_delay"].long()]
        self.cache = torch.where(refresh[:, None], selected, self.cache)
        measured, speed = self.encoders()
        self.history = torch.cat((self.history[:, 1:], torch.cat((measured, speed, self.last_action), dim=1)[:, None]), dim=1)
        return reward, (elapsed >= s["end"]) | unsafe

    @simulation_stream
    def reset(self):
        if self._seeds[0] is not None:
            self.generator.manual_seed(self._seeds[0])
        self._reset_worlds(torch.arange(self.num_envs, device="cuda"))
        self._reset_seeds()
        self._reset_options()
        return self._numpy(self.observation())

    @simulation_stream
    def step_async(self, actions):
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.num_envs, 14) or not np.isfinite(values).all():
            raise ValueError("Rhythm actions require fourteen finite normalized joint targets per world")
        self.actions = torch.as_tensor(values, device="cuda").clamp(-1, 1)

    @simulation_stream
    def step_wait(self):
        reward, done = self.advance(self.actions)
        observation = self._numpy(self.observation())
        rewards, dones = reward.cpu().numpy(), done.cpu().numpy()
        infos = [{} for _ in range(self.num_envs)]
        finished = np.flatnonzero(dones)
        if len(finished):
            metrics = {name: value.cpu().numpy() for name, value in self.metrics().items()}
            unsafe, returns = self.state["unsafe"].cpu().numpy(), self.state["episode_return"].cpu().numpy()
            steps = self.state["steps"].cpu().numpy()
            for index in finished:
                infos[index] = {**{name: float(value[index]) for name, value in metrics.items()},
                                "unsafe_contact": bool(unsafe[index]), "TimeLimit.truncated": False,
                                "terminal_observation": {name: value[index].copy() for name, value in observation.items()},
                                "episode": {"r": float(returns[index]), "l": int(steps[index])}}
            ids = torch.as_tensor(finished, device="cuda")
            self._reset_worlds(ids)
            fresh = self._numpy(self.observation())
            for name in observation:
                observation[name][finished] = fresh[name][finished]
        return observation, rewards, dones, infos

    def close(self):
        wp.synchronize_stream(self.stream)
        self.reference_env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Batched model attributes require explicit GPU updates")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("Use the batched environment API")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
