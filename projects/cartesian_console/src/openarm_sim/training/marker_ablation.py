import argparse
import hashlib
import json
from pathlib import Path

from ..act_distill import evaluate_student
from ..act_model import ActionChunkTransformer
from ..compute_guard import budgeted
from .pickup_evaluation import paired_comparison


@budgeted("openarm_sim.training.marker_ablation")
def main():
    parser = argparse.ArgumentParser(description="Evaluate visual ACT with different site marker visibility")
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", default=64, type=int)
    parser.add_argument("--worlds", default=16, type=int)
    args = parser.parse_args()
    model = ActionChunkTransformer.load(args.student)
    reports = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for markers in ("visible", "no_tcp", "hidden"):
        reports[markers] = evaluate_student(model, args.bank, args.output / markers,
                                           args.episodes, markers, args.worlds)
    report = {
        "student_sha256": hashlib.sha256(args.student.read_bytes()).hexdigest(),
        "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        "episodes": args.episodes,
        "success_rates": {name: value["success_rate"] for name, value in reports.items()},
        "no_tcp": paired_comparison(reports["visible"], reports["no_tcp"]),
        "hidden": paired_comparison(reports["visible"], reports["hidden"]),
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
