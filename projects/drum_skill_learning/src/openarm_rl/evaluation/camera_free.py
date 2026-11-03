from contextlib import ExitStack
import json

import imageio.v2 as imageio
import numpy as np

from ..envs.camera_free_strike import CameraFreeStrikeEnv, REWARD_VERSION


def evaluate(options, output, episodes, seed, policy=None, normalizer=None, video=False, exported=None):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ExitStack() as resources:
        env = resources.enter_context(CameraFreeStrikeEnv(**options, render_mode="rgb_array" if video else None))
        writer = resources.enter_context(imageio.get_writer(output / "execution.mp4", fps=50, codec="libx264")) if video else None
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total = 0.0
            for step in range(1000):
                if exported is not None:
                    import torch

                    with torch.no_grad():
                        action = exported(torch.tensor(observation["actor"]).unsqueeze(0)).numpy()[0]
                    if policy is not None:
                        expected, _ = policy.predict(normalizer.normalize_obs(observation), deterministic=True)
                        np.testing.assert_allclose(action, expected, atol=1e-5, rtol=1e-5)
                elif policy is None:
                    action = env.reference_action()
                else:
                    action, _ = policy.predict(normalizer.normalize_obs(observation), deterministic=True)
                observation, reward, done, truncated, info = env.step(action)
                total += reward
                if episode == 0 and writer is not None and step % 2 == 0:
                    writer.append_data(env.render())
                if done or truncated:
                    results.append({**info, "seed": seed + episode, "episode_return": total})
                    break
            else:
                raise RuntimeError("Timed strike exceeded its episode bound")
            if episode == 0:
                (output / "trace.json").write_text(json.dumps(env.trace, indent=2) + "\n")
    report = {"reward_version": REWARD_VERSION, "episodes": episodes,
              "success_rate": float(np.mean([r["is_success"] for r in results])),
              "contact_rate": float(np.mean([r["hit_count"] > 0 for r in results])),
              "recovery_rate": float(np.mean([r["recovered"] for r in results])),
              "unsafe_rate": float(np.mean([r["unsafe_contact"] for r in results])),
              "mean_return": float(np.mean([r["episode_return"] for r in results])), "results": results}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
