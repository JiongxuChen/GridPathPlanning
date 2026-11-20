"""Equal-budget step-marker alignment ablation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import importlib.metadata
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"),
                str(HERE)]

import aligned_representation
import evaluate_dev
import train as local_train
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments import training
from gridpath.experiments.data import load
from gridpath.planning import protocol
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

boundary_spec = spec_from_file_location("late_collision_v2_boundary", HERE / "boundary_loss.py")
boundary = module_from_spec(boundary_spec)
boundary_spec.loader.exec_module(boundary)
late = boundary.late
fresh = boundary.fresh

NAME = "late_collision_aligned_v2"
RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"
OUT = ROOT / "results/late_collision_v2"
FROZEN = OUT / "aligned_holdout.json"
SEED = 20261124
GROUPS = fresh.GROUPS


def source_hashes():
    return {"script": sha256(Path(__file__)),
            "protocol": sha256(HERE / "aligned_protocol.md"),
            "representation": sha256(HERE / "aligned_representation.py"),
            "boundary_holdout": sha256(boundary.FROZEN)}


def forbidden():
    seen = boundary.forbidden()
    seen.update(r["layout_id"] for pair in read_json(boundary.FROZEN)["pairs"]
                for r in pair.values())
    return seen


def audit(data):
    assert data["sources"] == source_hashes()
    excluded = forbidden()
    used = set()
    assert len(data["pairs"]) == 40
    for pair in data["pairs"]:
        assert set(pair) == set(GROUPS)
        lid = pair["short_valid"]["layout_id"]
        assert lid not in excluded and lid not in used
        assert {r["layout_id"] for r in pair.values()} == {lid}
        assert layout_split(lid, 42) == "test"
        used.add(lid)
        for name, row in pair.items():
            truth = execute(grid_from_record(row), row["candidate_plan"])
            assert truth.verification_label == row["target_text"]
            assert (truth.success if "valid" in name
                    else truth.error_reason == ErrorReason.OBSTACLE_COLLISION)
            if name == "early_collision":
                assert truth.first_error_step <= 7
            if name == "late_collision":
                assert 11 <= truth.first_error_step <= 17
    return {"layouts": len(used), "candidates": len(used) * 4,
            "excluded_layouts": len(excluded)}


def prepare():
    if FROZEN.exists():
        data = read_json(FROZEN)
        audit(data)
        return data
    used = forbidden()
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    pairs = []
    attempts = 0
    while len(pairs) < 40:
        attempts += 1
        if attempts > 300000:
            raise RuntimeError("Cannot fill aligned holdout")
        blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
        free = [cell for cell in cells if cell not in blocked]
        start, goal = rng.sample(free, 2)
        grid = Grid(6, 6, blocked, start, goal)
        lid = layout_id(grid)
        if lid in used or layout_split(lid, 42) != "test":
            continue
        path = shortest_path(grid)
        if not path.reachable or not 2 <= path.distance <= 7:
            continue
        try:
            pair = late.candidates(grid, path.actions, "late_v2_aligned_test")
        except ValueError:
            continue
        if pair is None:
            continue
        pairs.append(pair)
        used.add(lid)
    data = {"seed": SEED, "sources": source_hashes(),
            "attempts": attempts, "pairs": pairs}
    audit(data)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def binding():
    cfg, rows, dev, baseline = local_train.binding()
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {
        "experiment": NAME, "baseline": baseline["fingerprint"],
        "config": cfg, "train_sample_ids": digest([r["sample_id"] for r in rows]),
        "representation": sha256(HERE / "aligned_representation.py"),
        "entry": sha256(Path(__file__)),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("torch", "transformers", "numpy")},
    }
    return cfg, rows, dev, {"fingerprint": digest(payload), **payload}


def train():
    prepare()
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        assert done["fingerprint"] == identity["fingerprint"]
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    old = protocol.render, training.binding_for, training.budget
    protocol.render = aligned_representation.render
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        local_train.LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        protocol.render, training.binding_for, training.budget = old


def score(stage):
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier

    cfg, _, _, identity = binding()
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    target = OUT / ("aligned_" + stage + ".json")
    if target.exists():
        saved = read_json(target)
        assert saved["fingerprint"] == identity["fingerprint"]
        return saved
    rows = (read_json(fresh.FROZEN)["panels"]["dev"] if stage == "dev"
            else prepare()["pairs"])
    original = protocol.render
    protocol.render = aligned_representation.render
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], f"{NAME}:{stage}") as budget:
            tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
            network = StepVerifier.from_pretrained(RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(network, tok, cfg, budget)
            groups = {name: evaluate_dev.evaluate_group(
                [(pair[name], pair[name]["candidate_plan"]) for pair in rows], verifier)
                for name in GROUPS}
            if stage == "dev":
                fixed, natural = evaluate_dev.dev_items()
                groups["fixed_V80"] = evaluate_dev.evaluate_group(fixed, verifier)
                groups["natural_G60"] = evaluate_dev.evaluate_group(natural, verifier)
                source = load(["test_id_verify", "test_ood_verify"])
                for split in ("test_id_verify", "test_ood_verify"):
                    groups[split] = evaluate_dev.evaluate_group(
                        [(r, r["candidate_plan"]) for r in source[split]], verifier)
            calls = verifier.calls
    finally:
        protocol.render = original
    summary = {name: {"correct": sum(x["prediction"]["text"] == x["row"]["target_text"]
                                    for x in group["cases"]),
                      "count": len(group["cases"]),
                      "failures": group["metrics"]["interface_failure_count"],
                      "max_input_tokens": max(x["detail"].get("input_tokens", 0)
                                              for x in group["cases"])}
               for name, group in groups.items()}
    result = {"fingerprint": identity["fingerprint"],
              "challenge_sha256": sha256(fresh.FROZEN if stage == "dev" else FROZEN),
              "summary": summary, "groups": groups, "calls": calls}
    write_json(target, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "dev", "test"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif stage == "train":
        print(train()["state"]["global_step"])
    else:
        print(stage, score(stage)["summary"])
