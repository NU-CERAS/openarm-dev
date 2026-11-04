import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ..envs.camera_free_strike import REWARD_VERSION


def summarize(path):
    document = json.loads(Path(path).read_text())
    rows = document["results"]
    if document["reward_version"] != REWARD_VERSION or len(rows) != document["episodes"] or not rows:
        raise ValueError("Evaluation version or episode count differs")
    counts = Counter()
    speeds = {"low": [], "middle": [], "high": []}
    for row in rows:
        hits = row["hits"]
        if len(hits) != row["hit_count"]:
            raise ValueError("Recorded hit count differs from the event list")
        failed = {"no_hit": not hits, "extra_hit": len(hits) > 1, "not_recovered": not row["recovered"],
                  "unsafe_contact": row["unsafe_contact"], "encoder_failure": bool(row["failure"])}
        if hits:
            hit = hits[0]
            failed.update(center=hit["radial_offset_m"] > .01,
                          timing=abs(hit["onset_time_s"] - row["target_impact_time_s"]) > .06,
                          speed=abs(hit["downward_speed_m_s"] - row["target_speed_m_s"]) > .3,
                          long_contact=hit["contact_duration_s"] > .12)
        if bool(row["is_success"]) != (not any(failed.values())):
            raise ValueError(f"Stored success and recorded measurements disagree for seed {row['seed']}")
        counts.update(name for name, value in failed.items() if value)
        speed = row["target_speed_m_s"]
        if not .35 <= speed <= .85:
            raise ValueError("Requested speed is outside the evaluated tool range")
        group = "low" if speed < .5 else "middle" if speed < .7 else "high"
        speeds[group].append(bool(row["is_success"]))
    success = sum(row["is_success"] for row in rows)
    if not np.isclose(success / len(rows), document["success_rate"]):
        raise ValueError("Stored aggregate success differs from individual episodes")
    return {"episodes": len(rows), "successes": success, "success_rate": success / len(rows),
            "criterion_failure_counts": dict(counts), "counts_overlap": True,
            "speed_groups": {name: {"episodes": len(values), "success_rate": float(np.mean(values)) if values else None}
                             for name, values in speeds.items()},
            "seeds": [row["seed"] for row in rows]}


def main():
    parser = argparse.ArgumentParser(description="Classify recorded strike evaluation failures and verify success measurements")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "openarm.camera_free_tools.v1":
        raise ValueError("Unsupported tools manifest")
    report = {"reward_version": REWARD_VERSION, "targets": {}, "speed_bins_m_s": {"low": [.35, .5], "middle": [.5, .7], "high": [.7, .85]}}
    for target, row in manifest["targets"].items():
        if row["status"] == "unavailable":
            continue
        directory = args.manifest.parent / target
        reference = summarize(directory / "heldout_reference/evaluation.json")
        policy = summarize(directory / "heldout/evaluation.json")
        if reference.pop("seeds") != policy.pop("seeds"):
            raise ValueError("Reference and policy evaluation seeds differ")
        report["targets"][target] = {"reference": reference, "policy": policy}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
