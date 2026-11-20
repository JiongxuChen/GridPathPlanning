"""Equal-quota early/late collision pairs using the original full verifier."""
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import importlib.metadata
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1")]

import evaluate_dev
import representation
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

late_spec = spec_from_file_location(
    "late_collision_v1_experiment", ROOT / "analysis/late_collision_v1/experiment.py")
late = module_from_spec(late_spec)
late_spec.loader.exec_module(late)
fresh_spec = spec_from_file_location("late_collision_v2_fresh", HERE / "fresh_eval.py")
fresh = module_from_spec(fresh_spec)
fresh_spec.loader.exec_module(fresh)

NAME = "late_collision_balanced_v2"
RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"
OUT = ROOT / "results/late_collision_v2"
FROZEN = OUT / "balanced_holdout.json"
SEED = 20261122
GROUPS = fresh.GROUPS


def training_rows():
    previous = late.prepare()
    rows = list(previous["training_rows"])
    original_ids = {r["sample_id"] for r in load(["train_verify"])["train_verify"]}
    indices = [i for i, row in enumerate(rows)
               if row["sample_id"] not in original_ids
               and row["error_reason"] == "obstacle_collision"]
    assert len(indices) == 20
    for index in indices[:10]:
        old = rows[index]
        grid = grid_from_record(old)
        pair = late.candidates(grid, old["reference_plan"], "train")
        assert pair["late_collision"]["sample_id"] == old["sample_id"]
        rows[index] = pair["early_collision"]
    assert len(rows) == len({r["sample_id"] for r in rows}) == 1000
    assert Counter(r["is_valid"] for r in rows) == {True: 500, False: 500}
    assert Counter(r["error_reason"] for r in rows if not r["is_valid"]) == {
        reason.value: 125 for reason in ErrorReason}
    assert sum(r["is_valid"] and len(r["candidate_plan"]) > r["shortest_distance"]
               for r in rows) == 200
    assert sum(r["error_reason"] == "obstacle_collision"
               and r["unchecked_endpoint_is_goal"] for r in rows) == 83
    assert all(execute(grid_from_record(r), r["candidate_plan"]).verification_label
               == r["target_text"] for r in rows)
    return rows


def holdout_sources():
    return {"script": sha256(Path(__file__)),
            "protocol": sha256(HERE / "balanced_protocol.md"),
            "previous_late": sha256(late.FROZEN),
            "previous_fresh": sha256(fresh.FROZEN)}


def audit(data):
    assert data["sources"] == holdout_sources()
    excluded = fresh.forbidden_layouts()
    seen_fresh = read_json(fresh.FROZEN)
    excluded.update(row["layout_id"] for panel in seen_fresh["panels"].values()
                    for pair in panel for row in pair.values())
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
            grid = grid_from_record(row)
            truth = execute(grid, row["candidate_plan"])
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
    excluded = fresh.forbidden_layouts()
    already = fresh.prepare()
    excluded.update(row["layout_id"] for panel in already["panels"].values()
                    for pair in panel for row in pair.values())
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    pairs = []
    attempts = 0
    while len(pairs) < 40:
        attempts += 1
        if attempts > 300000:
            raise RuntimeError("Cannot fill balanced holdout")
        blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
        free = [cell for cell in cells if cell not in blocked]
        start, goal = rng.sample(free, 2)
        grid = Grid(6, 6, blocked, start, goal)
        lid = layout_id(grid)
        if lid in excluded or layout_split(lid, 42) != "test":
            continue
        path = shortest_path(grid)
        if not path.reachable or not 2 <= path.distance <= 7:
            continue
        try:
            pair = late.candidates(grid, path.actions, "late_v2_balanced_test")
        except ValueError:
            continue
        if pair is None:
            continue
        pairs.append(pair)
        excluded.add(lid)
    data = {"seed": SEED, "sources": holdout_sources(),
            "attempts": attempts, "pairs": pairs}
    audit(data)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def binding():
    cfg, _, dev, baseline = local_train.binding()
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    rows = training_rows()
    payload = {
        "experiment": NAME, "baseline": baseline["fingerprint"],
        "config": cfg, "train_sample_ids": digest([r["sample_id"] for r in rows]),
        "representation": sha256(Path(representation.__file__)),
        "training_entry": sha256(Path(__file__)),
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
    protocol.render = representation.render
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
    target = OUT / ("balanced_" + stage + ".json")
    if target.exists():
        saved = read_json(target)
        assert saved["fingerprint"] == identity["fingerprint"]
        return saved
    if stage == "dev":
        rows = read_json(fresh.FROZEN)["panels"]["dev"]
    elif stage == "test":
        rows = prepare()["pairs"]
    else:
        raise ValueError(stage)
    original = protocol.render
    protocol.render = representation.render
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
                      "failures": group["metrics"]["interface_failure_count"]}
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
