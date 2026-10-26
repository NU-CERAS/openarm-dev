import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run_command(command, log, compute=True):
    with log.open("a") as output:
        guarded = [sys.executable, "-m", "openarm_sim.compute_guard", *command] if compute else command
        process = subprocess.Popen(guarded,
                                   stdout=output, stderr=subprocess.STDOUT)
        log.with_suffix(".pid").write_text(str(process.pid) + "\n")
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def student_stage(args):
    args.output.mkdir(parents=True, exist_ok=args.resume)
    teacher = args.pickup_run / "best/act.pt"
    residual = args.pickup_run / "best"
    data = args.output / "initial_data"
    data.mkdir(exist_ok=args.resume)
    (args.output / "state.json").write_text(json.dumps({"phase": "teacher_collection", "attempts": args.demonstrations}) + "\n")
    collect = [sys.executable, "-m", "openarm_sim.act_distill", "--teacher", str(teacher), "--residual", str(residual)]
    continuation = ["--resume"] if args.resume else []
    run_command(collect + ["--bank", str(args.train_bank), "--output", str(data / "train.h5"),
                          "--episodes", str(args.demonstrations), "--seed", "850000", *continuation],
                args.output / "collect_train.log")
    run_command(collect + ["--bank", str(args.validation_bank), "--output", str(data / "eval.h5"),
                          "--episodes", "64", "--seed", "950000", *continuation],
                args.output / "collect_eval.log")
    datasets = [data / "train.h5"]
    prefix = "student_clock" if args.student_state == "position-clock" else "student"
    checkpoint = None
    for iteration in range(3):
        if iteration:
            query = args.output / f"{prefix}_query_{iteration}.h5"
            run_command(collect + ["--bank", str(args.train_bank), "--output", str(query), "--episodes", "128",
                                  "--seed", str(850000 + iteration), "--student", str(checkpoint), *continuation],
                        args.output / f"query_{iteration}.log")
            datasets.append(query)
            next_data = args.output / f"{prefix}_data_{iteration}"
            next_data.mkdir(exist_ok=args.resume)
            run_command([sys.executable, "-m", "openarm_sim.act_distill", "--merge", *map(str, datasets),
                         "--output", str(next_data / "train.h5")], args.output / f"merge_{iteration}.log")
            shutil.copyfile(data / "eval.h5", next_data / "eval.h5")
            data = next_data
        shutil.copyfile(args.train_bank, data / "train_resets.npz")
        shutil.copyfile(args.validation_bank, data / "eval_resets.npz")
        training = args.output / f"{prefix}_{iteration}"
        if (training / "history.json").exists() and (training / "best_task.pt").exists():
            checkpoint = training / "best_task.pt"
            continue
        command = [sys.executable, "-m", "openarm_sim.training.act_train", "--student", "--data", str(data),
                   "--output", str(training), "--steps", "8000" if not iteration else "4000",
                   "--batch-size", "32", "--evaluate-every", "1000", "--eval-worlds", "64",
                   "--student-markers", "hidden",
                   "--student-state", args.student_state]
        if args.resume and (training / "latest.pt").exists():
            checkpoint = training / "latest.pt"
        if checkpoint is not None:
            command += ["--initialize", str(checkpoint)]
        (args.output / "state.json").write_text(json.dumps({"phase": "training", "iteration": iteration,
                                                           "output": str(training)}, indent=2) + "\n")
        run_command(command, args.output / f"{prefix}_{iteration}.log")
        checkpoint = training / "best_task.pt"
    run_command([sys.executable, "-m", "openarm_sim.act_distill", "--evaluate-student", str(checkpoint),
                 "--bank", str(args.test_bank), "--output", str(args.output / "heldout"), "--episodes", "128",
                 "--worlds", "16", "--markers", "hidden"],
                args.output / "heldout.log")
    report = json.loads((args.output / "heldout/evaluation.json").read_text())
    (args.output / "state.json").write_text(json.dumps({"phase": "assessment_completed", "checkpoint": str(checkpoint),
                                                       "success_rate": report["success_rate"],
                                                       "qualified": report["success_rate"] >= 0.9,
                                                       "runtime_inputs": ["RGB", "joint_positions", "gripper_position", "controller_clock"]
                                                       if args.student_state == "position-clock" else ["RGB", "joint_feedback", "command_history"]}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Assess pickup, then run visual distillation and strike within one compute slot")
    parser.add_argument("--pickup-run", type=Path, required=True)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--validation-bank", type=Path, required=True)
    parser.add_argument("--test-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "student"], default="all")
    parser.add_argument("--demonstrations", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--student-state", choices=["feedback", "position-clock"], default="position-clock")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise ValueError("Use a new pipeline output directory")
    if args.stage == "student":
        student_stage(args)
        return
    if not (args.pickup_run / "final/evaluation.json").exists():
        raise ValueError("Pickup training has not completed its final validation")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if not (args.resume and (args.output / "pickup_assessment/comparison.json").exists()):
        run_command([sys.executable, "-m", "openarm_sim.training.pickup_evaluation", "--teacher", str(args.pickup_run / "best/act.pt"),
                     "--residual", str(args.pickup_run / "best"), "--bank", str(args.test_bank),
                     "--output", str(args.output / "pickup_assessment")], args.output / "pickup_assessment.log")
    reports = json.loads((args.output / "pickup_assessment/comparison.json").read_text())
    if reports["provenance"]["teacher_sha256"] != hashlib.sha256((args.pickup_run / "best/act.pt").read_bytes()).hexdigest():
        raise ValueError("Qualified ACT checkpoint differs from the requested teacher")
    if reports["provenance"]["bank_sha256"] != hashlib.sha256(args.test_bank.read_bytes()).hexdigest():
        raise ValueError("Qualified assessment uses a different reset bank")
    original, candidate, paired = reports["act"], reports["residual"], reports["paired"]
    unsafe_original = original["failure_counts"].get("unsafe_contact", 0)
    unsafe_candidate = candidate["failure_counts"].get("unsafe_contact", 0)
    passed = (candidate["success_rate"] >= 0.9 and paired["success_difference"] > 0
              and paired["paired_binomial_p"] < 0.05 and unsafe_candidate <= unsafe_original)
    (args.output / "gate.json").write_text(json.dumps({"passed": passed, "paired": paired,
                                                      "act_success_rate": original["success_rate"],
                                                      "candidate_success_rate": candidate["success_rate"]}, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Pickup robustness assessment did not pass; downstream training was not started")
    student_command = [sys.executable, "-m", "openarm_sim.training.skill_pipeline", "--stage", "student",
                       "--pickup-run", str(args.pickup_run), "--train-bank", str(args.train_bank),
                       "--validation-bank", str(args.validation_bank), "--test-bank", str(args.test_bank),
                       "--output", str(args.output / "visual_student"), "--demonstrations", str(args.demonstrations),
                       "--student-state", args.student_state]
    strike_command = [sys.executable, "-m", "openarm_sim.training.strike_train", "--output", str(args.output / "strike_teacher")]
    if args.resume:
        student_command += ["--resume"]
        strike_command += ["--resume-checkpoint", str(args.output / "strike_teacher/best")]
    (args.output / "jobs.json").write_text(json.dumps({"student": student_command, "strike": strike_command}, indent=2) + "\n")
    for name, command in (("visual_student", student_command), ("strike_teacher", strike_command)):
        (args.output / "status.json").write_text(json.dumps({"active_stage": name, "mode": "serial"}) + "\n")
        run_command(command, args.output / f"{name}.log", compute=name != "visual_student")
    student_result = json.loads((args.output / "visual_student/state.json").read_text())
    strike_result = json.loads((args.output / "strike_teacher/heldout/evaluation.json").read_text())
    (args.output / "complete.json").write_text(json.dumps({"execution_completed": True,
                                                          "student_qualified": student_result["qualified"],
                                                          "strike_qualified": strike_result["success_rate"] >= 0.9}, indent=2) + "\n")


if __name__ == "__main__":
    main()
