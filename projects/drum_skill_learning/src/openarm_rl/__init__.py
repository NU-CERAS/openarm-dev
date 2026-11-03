
from gymnasium.envs.registration import register

register(id="OpenArm-DrumStrike-v0", entry_point="openarm_rl.envs.drumming:DrumStrikeEnv", max_episode_steps=100)
register(id="OpenArm-StickPickup-v0", entry_point="openarm_rl.envs.pickup:StickPickupEnv", max_episode_steps=300)
register(id="OpenArm-ResidualDrumStrike-v0", entry_point="openarm_rl.envs.residual_drumming:ResidualDrumStrikeEnv", max_episode_steps=100)
register(id="OpenArm-OpenDrummerStrike-v0", entry_point="openarm_rl.envs.opendrummer:OpenDrummerStrikeEnv")
