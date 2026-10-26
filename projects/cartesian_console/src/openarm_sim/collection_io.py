import hashlib
import json

import h5py


def collection_start(teacher, residual, bank, output, episodes, seed, student, width, height, resume):
    if episodes <= 0:
        raise ValueError("Collection attempt count must be positive")
    if not output.exists():
        return 0, 0
    if not resume:
        raise ValueError("Collection output exists; use --resume to continue it")
    with h5py.File(output) as existing:
        expected = {"teacher_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
                    "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(), "bank_seed": seed,
                    "source": "ACT+PPO" if residual is not None else "ACT",
                    "supervision": "teacher_query" if student is not None else "executed"}
        if residual is not None:
            expected["residual_sha256"] = hashlib.sha256((residual / "policy.zip").read_bytes()).hexdigest()
        if student is not None:
            expected["student_sha256"] = hashlib.sha256(student.read_bytes()).hexdigest()
        if any(existing.attrs[key] != value for key, value in expected.items()):
            raise ValueError("Resume configuration differs from the saved collection")
        if tuple(existing.attrs["image_resolution"]) != (width, height):
            raise ValueError("Resume image resolution differs from the saved collection")
        previous_successes = 0
        completed = int(existing.attrs.get("attempts_completed", 0))
        for item in existing["episodes"].values():
            steps = int(item.attrs["steps"])
            if any(len(item[key]) != steps for key in ("command", "qpos", "qvel", "image", "observation")):
                raise ValueError("Saved collection contains an incomplete episode")
            previous_successes += bool(item.attrs["success"])
            completed = max(completed, int(item.attrs["attempt"]) + 1)
    if completed > episodes:
        raise ValueError("Saved collection already exceeds the requested attempt count")
    return completed, previous_successes


def save_progress(file, output, completed, successes, previous_completed, results):
    file.attrs["attempts_completed"] = completed
    file.attrs["successful"] = successes
    file.flush()
    report = {"attempts": completed, "successes": successes,
              "previous_completed_attempts": previous_completed, "results": results}
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output.with_suffix(".json"))
