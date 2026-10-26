import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import wandb

from .act_data import EpisodeDataset, command_action
from .act_model import ACTConfig, ACTController, ActionChunkTransformer, normalization
from .warp_pickup import WarpPickupVecEnv
from .warp_train import evaluate_and_record


class ACTExecutor:
    def __init__(self, model, env, query_interval=5):
        self.env = env
        self.controller = ACTController(model, env.num_envs, query_interval)

    def predict(self, observation, deterministic=True):
        command = self.controller.command(observation)
        return command_action(self.env, command).cpu().numpy(), None


def physical_evaluation(model, bank: Path, output: Path, steps: int, worlds=32, query_interval=5):
    if model.config.vision:
        raise ValueError("Visual ACT evaluation requires the camera student evaluator")
    model.eval()
    training_stream = torch.cuda.current_stream()
    env = WarpPickupVecEnv(worlds, bank, seed=100000)
    try:
        executor = ACTExecutor(model, env, query_interval)
        report = evaluate_and_record(executor, env, output, steps)
    finally:
        env.close()
        torch.cuda.synchronize()
        torch.cuda.set_stream(training_stream)
    return report


def main():
    parser = argparse.ArgumentParser(description="Train an oracle or visual ACT pickup policy")
    parser.add_argument("--data", type=Path, default=Path(".work/act/data"))
    parser.add_argument("--output", type=Path, default=Path(".work/act/teacher"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--query-interval", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=1000)
    parser.add_argument("--eval-worlds", type=int, default=32)
    parser.add_argument("--student", action="store_true")
    parser.add_argument("--relative-commands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--physical-oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-clip", type=float, default=10.0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.chunk, args.hidden, args.layers, args.evaluate_every, args.eval_worlds) <= 0:
        raise ValueError("ACT training sizes must be positive")
    if not 1 <= args.query_interval <= args.chunk or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Invalid ACT query interval or learning rate")
    if not np.isfinite(args.kl_weight) or args.kl_weight < 0:
        raise ValueError("ACT KL weight must be finite and nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train = EpisodeDataset(args.data / "train.h5", args.chunk, args.student)
    validation = EpisodeDataset(args.data / "eval.h5", args.chunk, args.student)
    import h5py

    with h5py.File(train.path) as train_file, h5py.File(validation.path) as validation_file:
        if train_file.attrs["bank_sha256"] == validation_file.attrs["bank_sha256"]:
            raise ValueError("Training and validation reset distributions must have distinct seeds")
    settings = {**vars(args), "algorithm": "ACT", "state_dim": train.state.shape[1],
                "reference_source_commit": "742c753c0d4a5d87076c8f69e5628c79a8cc5488",
                "train_sha256": hashlib.sha256(train.path.read_bytes()).hexdigest(),
                "eval_sha256": hashlib.sha256(validation.path.read_bytes()).hexdigest(),
                "train_episodes": len(train.episodes), "eval_episodes": len(validation.episodes)}
    (args.output / "run_config.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
    model = ActionChunkTransformer(ACTConfig(state_dim=train.state.shape[1], chunk=args.chunk,
                                           hidden=args.hidden, layers=args.layers, vision=args.student,
                                           relative_commands=args.relative_commands, physical_oracle=args.physical_oracle and not args.student,
                                           observation_clip=args.observation_clip),
                                   *normalization(train, args.relative_commands)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    run = wandb.init(project="openarm-teachers", name=args.output.name, dir=str(args.output),
                     config=json.loads(json.dumps(settings, default=str)), mode="online", force=True) if args.wandb else None
    if run:
        run.define_metric("updates")
        run.define_metric("*", step_metric="updates")
        (args.output / "wandb_url.txt").write_text(run.url + "\n")
        print(f"W&B run: {run.url}", flush=True)
    best_loss = float("inf")
    history = []
    try:
        for step in range(1, args.steps + 1):
            model.train()
            state, actions, padding, images = train.batch(rng.integers(len(train), size=args.batch_size))
            loss, reconstruction, kl = model.loss(state, actions, padding, images, args.kl_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite ACT training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 20 == 0:
                metrics = {"updates": step, "train/loss": float(loss.detach()),
                           "train/reconstruction": float(reconstruction.detach()), "train/kl": float(kl.detach())}
                history.append(metrics)
                if run:
                    run.log(metrics)
            if step % 200 == 0 or step == args.steps:
                model.eval()
                state, actions, padding, images = validation.batch(rng.integers(len(validation), size=512))
                with torch.no_grad():
                    predicted = model.commands(state, images)
                    valid = (~padding).unsqueeze(-1)
                    prior_l1 = float((((predicted - actions) / model.action_std).abs() * valid).sum() /
                                     (valid.sum() * 8))
                model.save(args.output / "latest.pt", updates=step, settings=json.loads(json.dumps(settings, default=str)))
                if prior_l1 < best_loss:
                    best_loss = prior_l1
                    model.save(args.output / "best.pt", updates=step, validation_prior_l1=prior_l1)
                print(json.dumps({"updates": step, "validation/prior_l1": prior_l1}), flush=True)
                if run:
                    run.log({"updates": step, "validation/prior_l1": prior_l1})
            if step % args.evaluate_every == 0 or step == args.steps:
                evaluation_dir = args.output / f"evaluation_{step:06d}"
                if args.student:
                    from .act_distill import evaluate_student

                    training_stream = torch.cuda.current_stream()
                    model.eval()
                    report = evaluate_student(model, args.data / "eval_resets.npz", evaluation_dir, args.eval_worlds)
                    torch.cuda.synchronize()
                    torch.cuda.set_stream(training_stream)
                else:
                    report = physical_evaluation(model, args.data / "eval_resets.npz", evaluation_dir,
                                                 step, args.eval_worlds, args.query_interval)
                model.save(evaluation_dir / "act.pt", updates=step, success_rate=report["success_rate"])
                if run:
                    run.log({"updates": step, "eval/success_rate": report["success_rate"], "eval/mean_return": report["mean_return"],
                             "eval/execution": wandb.Video(str(args.output / f"evaluation_{step:06d}/execution.mp4"), format="mp4")})
                    artifact = wandb.Artifact(f"act-{run.id}", type="model", metadata={"updates": step})
                    artifact.add_file(str(args.output / "latest.pt"))
                    run.log_artifact(artifact)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
