from gymnasium.envs.registration import register

from .simulation import ReachSimulation

register(id="OpenArm-DrumStrike-v0", entry_point="openarm_sim.drumming:DrumStrikeEnv", max_episode_steps=100)
register(id="OpenArm-StickPickup-v0", entry_point="openarm_sim.pickup:StickPickupEnv", max_episode_steps=300)
register(id="OpenArm-ResidualDrumStrike-v0", entry_point="openarm_sim.residual_drumming:ResidualDrumStrikeEnv", max_episode_steps=100)

__all__ = ["ReachSimulation"]
