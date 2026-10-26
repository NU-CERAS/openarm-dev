from pathlib import Path
from functools import wraps

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from stable_baselines3.common.vec_env import VecEnv

from .pickup import StickPickupEnv
from .pickup_reward import PickupReward, REWARD_VERSION


def simulation_stream(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        caller = torch.cuda.current_stream()
        changed = caller.cuda_stream != self.torch_stream.cuda_stream
        if changed:
            self.torch_stream.wait_stream(caller)
        with wp.ScopedStream(self.stream, sync_enter=False, sync_exit=False), torch.cuda.stream(self.torch_stream):
            result = method(self, *args, **kwargs)
        if changed:
            caller.wait_stream(self.torch_stream)
        return result
    return run


@wp.kernel
def servo_control(desired: wp.array2d[float], bias: wp.array2d[float],
                  dofs: wp.array[wp.int32], gains: wp.array[float], limits: wp.array2d[float],
                  control: wp.array2d[float]):
    world, actuator = wp.tid()
    control[world, actuator] = wp.clamp(desired[world, actuator] + bias[world, dofs[actuator]] / gains[actuator],
                                       limits[actuator, 0], limits[actuator, 1])


@wp.kernel
def clear_finger_contacts(flags: wp.array2d[int]):
    world = wp.tid()
    flags[world, 0] = 0
    flags[world, 1] = 0
    flags[world, 3] = 0


@wp.kernel
def collect_contacts(count: wp.array[int], pairs: wp.array[wp.vec2i], worlds: wp.array[int],
                     distances: wp.array[float], robot_geom: wp.array[bool],
                     stick: int, finger0: int, finger1: int, flags: wp.array2d[int]):
    index = wp.tid()
    if index < count[0]:
        pair = pairs[index]
        world = worlds[index]
        touches_stick = pair[0] == stick or pair[1] == stick
        if touches_stick:
            other = pair[1] if pair[0] == stick else pair[0]
            if other != finger0 and other != finger1:
                wp.atomic_max(flags, world, 3, 1)
            if pair[0] == finger0 or pair[1] == finger0:
                wp.atomic_max(flags, world, 0, 1)
            if pair[0] == finger1 or pair[1] == finger1:
                wp.atomic_max(flags, world, 1, 1)
        elif distances[index] < -0.002 and (robot_geom[pair[0]] or robot_geom[pair[1]]):
            wp.atomic_max(flags, world, 2, 1)


def build_reset_bank(path: Path, count: int, seed: int, position_range: float = 0.05) -> None:
    rng = np.random.default_rng(seed)
    states, commands, drum_positions = [], [], []
    with StickPickupEnv() as env:
        sim = env.sim
        address = sim.model.jnt_qposadr[env.stick_joint_id]
        drum_id = sim.model.geom("drumhead").id
        nominal_drum = sim.model.geom_pos[drum_id].copy()
        attempts = 0
        while len(states) < count:
            attempts += 1
            if attempts > count * 20:
                raise RuntimeError("Could not sample enough collision-free pickup starting states")
            sim.reset()
            sim.data.qpos[address:address + 2] += rng.uniform(-position_range, position_range, 2)
            drum = nominal_drum + rng.uniform(-position_range, position_range, 3)
            if drum[2] - 0.015 < sim.table_height:
                continue
            sim.model.geom_pos[drum_id] = drum
            mujoco.mj_forward(sim.model, sim.data)
            target = env.stick_position() + rng.uniform(-position_range, position_range, 3)
            target[2] += 0.112
            goal = sim._solve("right", target, env.tool_orientation)
            if goal is None:
                continue
            sim.data.qpos[:] = goal
            sim.data.qvel[:] = 0
            mujoco.mj_forward(sim.model, sim.data)
            if sim._collision(sim.data):
                continue
            sim._desired = sim.data.qpos[sim.model.jnt_qposadr[sim.model.actuator_trnid[:, 0]]].copy()
            for _ in range(100):
                sim.step()
            if env.unsafe_contacts() or np.linalg.norm(sim.tcp_position("right") - target) > 0.01:
                continue
            states.append(sim.data.qpos.copy())
            commands.append(sim._desired.copy())
            drum_positions.append(drum)
            if len(states) % 64 == 0:
                print(f"Validated reset states: {len(states)}/{count}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, qpos=states, commands=commands, drum_positions=drum_positions,
                 seed=seed, position_range=position_range, attempts=attempts)


class WarpPickupVecEnv(VecEnv):
    render_mode = None

    def __init__(self, num_envs: int, bank: Path, seed: int = 0, sequential_resets: bool = False):
        self.cpu = StickPickupEnv()
        self.sim = self.cpu.sim
        self.model = self.sim.model
        self.rng = np.random.default_rng(seed)
        self.sequential_resets = sequential_resets
        self.reset_cursor = 0
        with np.load(bank) as source:
            self.bank_qpos = torch.as_tensor(source["qpos"], dtype=torch.float32, device="cuda")
            self.bank_commands = torch.as_tensor(source["commands"], dtype=torch.float32, device="cuda")
            self.bank_drums = torch.as_tensor(source["drum_positions"], dtype=torch.float32, device="cuda")
        if sequential_resets and num_envs > len(self.bank_qpos):
            raise ValueError("Independent evaluation needs at least one reset state per world")
        self.device = "cuda"
        wp.init()
        torch.cuda.synchronize()
        self.stream = wp.Stream(f"cuda:{torch.cuda.current_device()}")
        self.torch_stream = wp.stream_to_torch(self.stream)
        self._initialize_gpu(num_envs)

    @simulation_stream
    def _initialize_gpu(self, num_envs):
        self.wmodel = mjw.put_model(self.model)
        self.data = mjw.put_data(self.model, self.sim.data, nworld=num_envs, nconmax=64, njmax=256)
        self.wmodel.geom_pos = wp.array(np.tile(self.model.geom_pos, (num_envs, 1, 1)), dtype=wp.vec3, device="cuda")
        self.geom_positions = wp.to_torch(self.wmodel.geom_pos)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.bias = wp.to_torch(self.data.qfrc_bias)
        self.xpos = wp.to_torch(self.data.xpos)
        self.xquat = wp.to_torch(self.data.xquat)
        self.cvel = wp.to_torch(self.data.cvel)
        self.sites = wp.to_torch(self.data.site_xpos)
        self.arm_qids = torch.as_tensor(self.sim.qpos_ids["right"], device="cuda")
        self.arm_vids = torch.as_tensor(self.model.jnt_dofadr[self.sim.joint_ids["right"]], device="cuda")
        self.arm_aids = torch.as_tensor(self.sim.actuator_ids["right"], device="cuda")
        self.actuated_dofs = torch.as_tensor(self.sim.actuated_dofs, device="cuda")
        self.kp = torch.as_tensor(self.sim.kp, dtype=torch.float32, device="cuda")
        self.limits = torch.as_tensor(self.model.actuator_ctrlrange, dtype=torch.float32, device="cuda")
        self.robot_geom = torch.as_tensor(np.isin(self.model.geom_bodyid, list(self.sim.robot_bodies)), device="cuda")
        self.tcp_body = self.model.body("openarm_right_hand_tcp").id
        self.drum_id = self.model.geom("drumhead").id
        self.desired = torch.zeros((num_envs, self.model.nu), device="cuda")
        self.last_action = torch.zeros((num_envs, 8), device="cuda")
        self.elapsed = torch.zeros(num_envs, device="cuda")
        self.reset_samples = torch.zeros(num_envs, dtype=torch.long, device="cuda")
        self.initial_height = torch.zeros(num_envs, device="cuda")
        self.returns = np.zeros(num_envs)
        self.lengths = np.zeros(num_envs, dtype=int)
        self.reward_state = PickupReward(num_envs, device="cuda")
        self.flags_wp = wp.zeros((num_envs, 4), dtype=int, device="cuda")
        self.flags = wp.to_torch(self.flags_wp)
        self.desired_wp = wp.from_torch(self.desired)
        self.dofs_wp = wp.from_torch(self.actuated_dofs)
        self.gains_wp = wp.from_torch(self.kp)
        self.limits_wp = wp.from_torch(self.limits)
        self.robot_geom_wp = wp.from_torch(self.robot_geom)
        super().__init__(num_envs, gym.spaces.Box(-np.inf, np.inf, (57,), dtype=np.float32),
                         gym.spaces.Box(-1, 1, (8,), dtype=np.float32))
        self._reset_indices(np.arange(num_envs))
        self._physics()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(round(0.02 / self.model.opt.timestep)):
                self._physics()
        self.graph = capture.graph
        self.reset()

    def _physics(self):
        wp.launch(servo_control, dim=(self.num_envs, self.model.nu),
                  inputs=[self.desired_wp, self.data.qfrc_bias, self.dofs_wp, self.gains_wp, self.limits_wp],
                  outputs=[self.data.ctrl])
        mjw.step(self.wmodel, self.data)
        mjw.forward(self.wmodel, self.data)
        self._scan_contacts()

    def _scan_contacts(self):
        wp.launch(clear_finger_contacts, dim=self.num_envs, inputs=[self.flags_wp])
        fingers = sorted(self.cpu.finger_geom_ids)
        wp.launch(collect_contacts, dim=self.data.naconmax,
                  inputs=[self.data.nacon, self.data.contact.geom, self.data.contact.worldid,
                          self.data.contact.dist, self.robot_geom_wp, self.cpu.stick_geom_id, *fingers],
                  outputs=[self.flags_wp])

    @simulation_stream
    def _reset_indices(self, indices):
        if len(indices) == 0:
            return
        ids = torch.as_tensor(indices, device="cuda")
        if self.sequential_resets:
            samples = torch.as_tensor((np.arange(len(indices)) + self.reset_cursor) % len(self.bank_qpos), device="cuda")
            self.reset_cursor += len(indices)
        else:
            samples = torch.as_tensor(self.rng.integers(len(self.bank_qpos), size=len(indices)), device="cuda")
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device="cuda")
        mask[ids] = True
        mjw.reset_data(self.wmodel, self.data, wp.from_torch(mask))
        self.qpos[ids] = self.bank_qpos[samples]
        self.reset_samples[ids] = samples
        self.qvel[ids] = 0
        self.desired[ids] = self.bank_commands[samples]
        self.geom_positions[ids, self.drum_id] = self.bank_drums[samples]
        self.ctrl[ids] = self.desired[ids]
        self.last_action[ids] = 0
        self.elapsed[ids] = 0
        self.returns[indices] = 0
        self.lengths[indices] = 0
        address = self.model.jnt_qposadr[self.cpu.stick_joint_id]
        self.initial_height[ids] = self.qpos[ids, address + 2]
        mjw.forward(self.wmodel, self.data)
        self._scan_contacts()
        distance = torch.linalg.vector_norm(self.xpos[ids, self.cpu.stick_body_id]
                                            - self.sites[ids, self.sim.site_ids["right"]], dim=1)
        self.reward_state.reset(ids, distance)

    def _observation(self):
        stick = self.cpu.stick_body_id
        tcp = self.sites[:, self.sim.site_ids["right"]]
        return torch.cat((self.qpos[:, self.arm_qids], self.qvel[:, self.arm_vids],
                          self.qpos[:, self.cpu.grip_qpos_id, None], self.qvel[:, self.cpu.grip_dof_id, None],
                          self.xpos[:, stick] - tcp, self.xquat[:, stick], self.cvel[:, stick],
                          self.xquat[:, self.tcp_body],
                          (self.initial_height + 0.10 - self.xpos[:, stick, 2])[:, None], self.last_action,
                          self.reward_state.features(), self.flags[:, :2].sum(dim=1)[:, None] / 2,
                          self.flags[:, 3, None].float(), self.geom_positions[:, self.drum_id] - tcp), dim=1)

    def _contacts(self):
        count = int(wp.to_torch(self.data.nacon)[0].item())
        if count > self.data.naconmax or bool(wp.to_torch(self.data.overflow).any().item()):
            raise RuntimeError("MuJoCo Warp contact or constraint capacity exceeded")
        return self.flags[:, :2].sum(dim=1), self.flags[:, 2].bool()

    @simulation_stream
    def reset(self):
        if self._seeds[0] is not None:
            self.rng = np.random.default_rng(self._seeds[0])
        if self.sequential_resets:
            offset = self._options[0].get("bank_offset", 0)
            if type(offset) is not int or offset < 0 or offset + self.num_envs > len(self.bank_qpos):
                raise ValueError("Sequential reset offset exceeds the reset bank")
            self.reset_cursor = offset
        self._reset_indices(np.arange(self.num_envs))
        self._reset_seeds()
        self._reset_options()
        return self._observation().cpu().numpy()

    @simulation_stream
    def step_async(self, actions):
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.num_envs, 8) or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
            raise ValueError("Invalid batched pickup actions")
        self.actions = torch.as_tensor(values, device="cuda")

    @simulation_stream
    def step_wait(self):
        self.desired[:, self.arm_aids] = torch.clamp(self.desired[:, self.arm_aids] + 0.025 * self.actions[:, :7],
                                                   self.limits[self.arm_aids, 0], self.limits[self.arm_aids, 1])
        self.desired[:, self.cpu.grip_actuator_id] = (self.actions[:, 7] + 1) * 0.022
        self.flags.zero_()
        wp.capture_launch(self.graph)
        contacts, unsafe = self._contacts()
        stick = self.cpu.stick_body_id
        lift = self.xpos[:, stick, 2] - self.initial_height
        distance = torch.linalg.vector_norm(self.xpos[:, stick] - self.sites[:, self.sim.site_ids["right"]], dim=1)
        rewards, success, failure, timeout, terms = self.reward_state.step(
            distance, contacts == 2, lift, self.flags[:, 3].bool(),
            torch.linalg.vector_norm(self.cvel[:, stick, 3:], dim=1), unsafe,
            self.xpos[:, stick, 2] < self.sim.table_height - 0.05, self.actions, self.last_action,
        )
        self.elapsed.copy_(self.reward_state.elapsed)
        terminated = success | failure | timeout
        self.last_action.copy_(self.actions)
        obs = self._observation().cpu().numpy()
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite MuJoCo Warp state")
        reward = rewards.cpu().numpy()
        done = terminated.cpu().numpy()
        success_np, unsafe_np = success.cpu().numpy(), unsafe.cpu().numpy()
        self.returns += reward
        self.lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        done_indices = np.flatnonzero(done)
        if len(done_indices):
            terminal_positions = self.qpos.cpu().numpy().copy()
            diagnostics = torch.stack((self.reward_state.best_height, self.reward_state.awarded[:, 0],
                                        self.reward_state.awarded[:, 1], self.reward_state.grasp_steps /
                                        torch.clamp(self.reward_state.elapsed / 0.02, min=1),
                                        timeout.float(), (failure | timeout).float(),
                                        *self.reward_state.term_totals.values()), dim=1).cpu().numpy()
        for i in np.flatnonzero(done):
            infos[i] = {"is_success": bool(success_np[i]), "unsafe_contact": bool(unsafe_np[i]),
                        "reset_index": int(self.reset_samples[i].item()),
                        "terminal_qpos": terminal_positions[i],
                        "TimeLimit.truncated": False, "terminal_observation": obs[i].copy(),
                        "max_lift_m": float(diagnostics[i, 0]), "milestone_2cm": bool(diagnostics[i, 1]),
                        "milestone_5cm": bool(diagnostics[i, 2]), "grasp_fraction": float(diagnostics[i, 3]),
                        "timeout": bool(diagnostics[i, 4]), "failed": bool(diagnostics[i, 5]),
                        "reward_version": REWARD_VERSION,
                        "reward_terms": {name: float(diagnostics[i, 6 + j])
                                         for j, name in enumerate(self.reward_state.term_totals)},
                        "episode": {"r": float(self.returns[i]), "l": int(self.lengths[i])}}
        self._reset_indices(np.flatnonzero(done))
        if done.any():
            obs[done] = self._observation()[torch.as_tensor(done, device="cuda")].cpu().numpy()
        return obs, reward, done, infos

    @simulation_stream
    def snapshot(self, world=0):
        mjw.get_data_into(self.sim.data, self.model, self.data, world_id=world)
        self.model.geom_pos[self.drum_id] = self.geom_positions[world, self.drum_id].cpu().numpy()
        mujoco.mj_forward(self.model, self.sim.data)
        return self.sim.data

    @simulation_stream
    def render_snapshot(self, world=0):
        self.sim.data.qpos[:] = self.qpos[world].cpu().numpy()
        self.sim.data.qvel[:] = self.qvel[world].cpu().numpy()
        self.sim.data.time = float(self.elapsed[world].item())
        self.model.geom_pos[self.drum_id] = self.geom_positions[world, self.drum_id].cpu().numpy()
        mujoco.mj_kinematics(self.model, self.sim.data)
        mujoco.mj_camlight(self.model, self.sim.data)
        return self.sim.data

    def close(self):
        wp.synchronize_stream(self.stream)
        self.cpu.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Batched model attributes require explicit GPU updates")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("Use the batched environment API")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
