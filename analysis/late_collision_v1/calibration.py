"""Independent dev-bucket calibration panel for an optional collision gate."""
from collections import Counter
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import evaluate_dev
import experiment
import representation
import train as local_train
from gridpath.bfs import shortest_path
from gridpath.data_schema import layout_id, layout_split
from gridpath.experiments.data import load
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import Grid
from transformers import AutoTokenizer

OUT = experiment.OUT
FROZEN = OUT / "calibration_frozen.json"
PREDICTIONS = OUT / "calibration_predictions.json"
DECISION = OUT / "calibration_decision.json"
SEED = 20261118
THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.99)
GROUPS = ("short_valid", "long_valid", "early_collision", "late_collision")


def prepare():
    sources = {"script": sha256(Path(__file__)), "protocol": sha256(HERE / "calibration.md"),
               "previous": sha256(experiment.FROZEN)}
    if FROZEN.exists():
        data = read_json(FROZEN)
        if data["sources"] != sources:
            raise ValueError("Calibration source changed")
        return data
    existing = load(["train_generate", "train_verify", "dev_samples",
                     "test_id_samples", "test_ood_samples"])
    forbidden = {r["layout_id"] for rows in existing.values() for r in rows}
    for name in ("fresh_holdout", "robustness_v1_challenge"):
        data = read_json(local_train.OUT / (name + ".json"))
        if name == "fresh_holdout":
            forbidden.update(r["layout_id"] for group in
                             ("fixed_positive", "fixed_hard_collision", "generate")
                             for r in data[group])
        else:
            forbidden.update(item["row"]["layout_id"] for group in data["groups"].values()
                             for item in group)
    prepared = experiment.prepare()
    forbidden.update(r["layout_id"] for pairs in prepared["panels"].values()
                     for pair in pairs for r in pair.values())
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    pairs = []
    attempts = 0
    while len(pairs) < 40:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("Cannot fill calibration panel")
        blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
        free = [p for p in cells if p not in blocked]
        start, goal = rng.sample(free, 2)
        grid = Grid(6, 6, blocked, start, goal)
        lid = layout_id(grid)
        if lid in forbidden or layout_split(lid, 42) != "dev":
            continue
        path = shortest_path(grid)
        if not path.reachable or not 2 <= path.distance <= 7:
            continue
        try:
            pair = experiment.candidates(grid, path.actions, "calibration")
        except ValueError:
            continue
        if pair is None:
            continue
        pairs.append(pair)
        forbidden.add(lid)
    data = {"seed": SEED, "sources": sources, "attempts": attempts, "pairs": pairs}
    write_json(FROZEN, data)
    return data


def evaluate():
    data = prepare()
    cfg, _, _, identity = local_train.binding()
    validate_checkpoint(local_train.RUN / "latest", identity["fingerprint"])
    if PREDICTIONS.exists():
        result = read_json(PREDICTIONS)
        if (result["frozen_sha256"] != sha256(FROZEN) or
                result["checkpoint_fingerprint"] != identity["fingerprint"]):
            raise ValueError("Calibration predictions mismatch")
        return result
    original = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], "late_collision_v1:calibration") as budget:
            tok = AutoTokenizer.from_pretrained(local_train.RUN / "latest", local_files_only=True)
            model = StepVerifier.from_pretrained(local_train.RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(model, tok, cfg, budget)
            groups = {name: evaluate_dev.evaluate_group(
                [(pair[name], pair[name]["candidate_plan"]) for pair in data["pairs"]], verifier)
                for name in GROUPS}
            result = {"frozen_sha256": sha256(FROZEN),
                      "checkpoint_fingerprint": identity["fingerprint"],
                      "groups": groups, "gpu_calls": verifier.calls}
            write_json(PREDICTIONS, result)
            return result
    finally:
        protocol.render = original


def gated_text(case, threshold=None):
    tags = list(case["detail"]["tags"])
    if threshold is not None:
        for i, tag in enumerate(tags[:-1]):
            if tag == 1 and case["detail"]["probabilities"][i][1] < threshold:
                tags[i] = 0
    return protocol.aggregate(tags, len(case["row"]["candidate_plan"]))


def counts(groups, threshold):
    return {name: sum(gated_text(case, threshold) == case["row"]["target_text"]
                      for case in groups[name]["cases"]) for name in groups}


def decide():
    result = evaluate()
    original_dev = read_json(local_train.OUT / "verification_dev.json")
    guards = {"fixed_V80": original_dev["fixed_V80"],
              "natural_G60": original_dev["natural_G60"]}
    baseline = counts(result["groups"], None)
    guard_baseline = counts(guards, None)
    assert baseline["short_valid"] == baseline["long_valid"] == 40
    assert guard_baseline == {"fixed_V80": 80, "natural_G60": 60}
    table = []
    selected = None
    for threshold in THRESHOLDS:
        current = counts(result["groups"], threshold)
        regression = counts(guards, threshold)
        eligible = (current["late_collision"] > baseline["late_collision"] and
                    current["early_collision"] >= baseline["early_collision"] and
                    current["short_valid"] == baseline["short_valid"] and
                    current["long_valid"] == baseline["long_valid"] and
                    regression == guard_baseline)
        table.append({"threshold": threshold, "calibration": current,
                      "original_dev": regression, "eligible": eligible})
        if selected is None and eligible:
            selected = threshold
    decision = {"calibration_sha256": sha256(PREDICTIONS),
                "protocol_sha256": sha256(HERE / "calibration.md"),
                "baseline": baseline, "original_dev_baseline": guard_baseline,
                "thresholds": table, "selected": selected}
    if DECISION.exists() and read_json(DECISION) != decision:
        raise ValueError("Calibration decision changed")
    write_json(DECISION, decision)
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "decide"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print("calibration layouts", len(prepare()["pairs"]), sha256(FROZEN))
    elif stage == "evaluate":
        print("calibration calls", evaluate()["gpu_calls"])
    else:
        print(decide())
