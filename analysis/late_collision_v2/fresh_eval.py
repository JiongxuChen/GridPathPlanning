"""Freeze a fresh paired panel, then compare fixed verifier routes."""
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"),
                str(ROOT / "analysis/representation_attribution_v1")]

import evaluate_dev
import representation
import train as local_train
import variants
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments.data import MANIFEST, load
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import Grid
from transformers import AutoTokenizer

import experiment as attribution

late_spec = spec_from_file_location(
    "late_collision_v1_experiment", ROOT / "analysis/late_collision_v1/experiment.py")
late = module_from_spec(late_spec)
late_spec.loader.exec_module(late)

OUT = ROOT / "results/late_collision_v2"
FROZEN = OUT / "fresh_frozen.json"
SEED = 20261121
QUOTA = 40
GROUPS = ("short_valid", "long_valid", "early_collision", "late_collision")
THRESHOLD = 0.7


def sources():
    paths = {
        "script": HERE / "fresh_eval.py",
        "protocol": HERE / "protocol.md",
        "dataset": MANIFEST,
        "late_constructor": ROOT / "analysis/late_collision_v1/experiment.py",
        "prior_late": ROOT / "results/late_collision_v1/prepared.json",
        "calibration": ROOT / "results/late_collision_v1/calibration_frozen.json",
        "attribution": ROOT / "results/representation_attribution_v1/challenge.json",
        "sealed": ROOT / "results/sealed_verifier_v1/challenge.json",
        "natural": ROOT / "analysis/natural_collision_v1/pool.json",
    }
    return {name: sha256(path) for name, path in paths.items()}


def forbidden_layouts():
    prior = load(["train_generate", "train_verify", "dev_samples",
                  "test_id_samples", "test_ood_samples"])
    seen = {row["layout_id"] for rows in prior.values() for row in rows}
    fresh = read_json(local_train.OUT / "fresh_holdout.json")
    seen.update(row["layout_id"] for name in
                ("fixed_positive", "fixed_hard_collision", "generate")
                for row in fresh[name])
    robust = read_json(local_train.OUT / "robustness_v1_challenge.json")
    seen.update(item["row"]["layout_id"] for rows in robust["groups"].values()
                for item in rows)
    late_data = read_json(ROOT / "results/late_collision_v1/prepared.json")
    seen.update(row["layout_id"] for panel in late_data["panels"].values()
                for pair in panel for row in pair.values())
    calibration = read_json(ROOT / "results/late_collision_v1/calibration_frozen.json")
    seen.update(row["layout_id"] for pair in calibration["pairs"]
                for row in pair.values())
    attribution_data = read_json(ROOT / "results/representation_attribution_v1/challenge.json")
    seen.update(row["layout_id"] for panel in attribution_data["panels"].values()
                for rows in panel.values() for row in rows)
    sealed = read_json(ROOT / "results/sealed_verifier_v1/challenge.json")
    seen.update(item["row"]["layout_id"] for rows in sealed["groups"].values()
                for item in rows)
    natural = read_json(ROOT / "analysis/natural_collision_v1/pool.json")
    seen.update(record["source"]["layout_id"] for record in natural["records"])
    return seen


def audit(data):
    assert data["sources"] == sources()
    excluded = forbidden_layouts()
    used = set()
    for split in ("dev", "test"):
        pairs = data["panels"][split]
        assert len(pairs) == QUOTA
        for pair in pairs:
            assert set(pair) == set(GROUPS)
            lids = {row["layout_id"] for row in pair.values()}
            assert len(lids) == 1
            lid = lids.pop()
            assert lid not in excluded and lid not in used
            assert layout_split(lid, 42) == split
            used.add(lid)
            for name, row in pair.items():
                grid = grid_from_record(row)
                assert grid.rows == grid.cols == 6
                truth = execute(grid, row["candidate_plan"])
                assert truth.verification_label == row["target_text"]
                assert (truth.success if "valid" in name
                        else truth.error_reason.value == "obstacle_collision")
                if name == "early_collision":
                    assert truth.first_error_step <= 7
                if name == "late_collision":
                    assert 11 <= truth.first_error_step <= 17
    return {"layouts": len(used), "excluded_layouts": len(excluded),
            "candidates": len(used) * len(GROUPS)}


def prepare():
    if FROZEN.exists():
        data = read_json(FROZEN)
        audit(data)
        return data
    used = forbidden_layouts()
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    panels = {"dev": [], "test": []}
    attempts = Counter()
    for split in panels:
        while len(panels[split]) < QUOTA:
            attempts[split] += 1
            if attempts[split] > 300000:
                raise RuntimeError("Cannot fill new panel " + split)
            blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
            free = [cell for cell in cells if cell not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(6, 6, blocked, start, goal)
            lid = layout_id(grid)
            if lid in used or layout_split(lid, 42) != split:
                continue
            path = shortest_path(grid)
            if not path.reachable or not 2 <= path.distance <= 7:
                continue
            try:
                pair = late.candidates(grid, path.actions, "late_v2_" + split)
            except ValueError:
                continue
            if pair is None:
                continue
            panels[split].append(pair)
            used.add(lid)
    data = {"seed": SEED, "sources": sources(),
            "attempts": dict(attempts), "panels": panels}
    audit(data)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def gate(case):
    tags = list(case["detail"]["tags"])
    for index, tag in enumerate(tags[:-1]):
        if tag == 1 and case["detail"]["probabilities"][index][1] < THRESHOLD:
            tags[index] = 0
    return protocol.aggregate(tags, len(case["row"]["candidate_plan"]))


def score(split):
    frozen = prepare()
    full_cfg, _, _, full = local_train.binding()
    center_cfg, _, _, center = attribution.binding("center_only")
    models = {
        "full": (full_cfg, local_train.RUN / "latest", full["fingerprint"],
                 representation.render),
        "center_only": (center_cfg, attribution.run_dir("center_only") / "latest",
                        center["fingerprint"],
                        lambda grid, actions: variants.render(grid, actions, "center_only")),
    }
    calibration_decision = ROOT / "results/late_collision_v1/calibration_decision.json"
    decision = read_json(calibration_decision)
    assert decision["selected"] == THRESHOLD
    identities = {name: identity for name, (_, path, identity, _) in models.items()}
    for cfg, path, identity, renderer in models.values():
        validate_checkpoint(path, identity)
    output = OUT / (split + "_fresh_predictions.json")
    if output.exists():
        report = read_json(output)
        assert report["challenge_sha256"] == sha256(FROZEN)
        assert report["identities"] == identities
        assert report["calibration_decision_sha256"] == sha256(calibration_decision)
        return report
    original = protocol.render
    results = {}
    try:
        with GPUBudget(local_train.LEDGER, full_cfg["gpu_hours_cap"],
                       full_cfg["session_wall_seconds"],
                       "late_collision_v2:fresh_" + split) as budget:
            for name, (cfg, path, identity, render) in models.items():
                protocol.render = render
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                network = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
                verifier = Verifier(network, tokenizer, cfg, budget)
                results[name] = {group: evaluate_dev.evaluate_group(
                    [(pair[group], pair[group]["candidate_plan"])
                     for pair in frozen["panels"][split]], verifier)
                    for group in GROUPS}
                results[name]["model_calls"] = verifier.calls
                del verifier, network
    finally:
        protocol.render = original
    summary = {}
    for group in GROUPS:
        summary[group] = {}
        for route in ("full", "full_gate_0.7", "center_only"):
            cases = results["center_only" if route == "center_only" else "full"][group]["cases"]
            summary[group][route] = {
                "joint_correct": sum((gate(case) if route == "full_gate_0.7"
                                      else case["prediction"]["text"]) == case["row"]["target_text"]
                                     for case in cases),
                "count": len(cases),
                "interface_failures": sum(case["prediction"]["failure"] is not None
                                          for case in cases),
            }
    report = {"challenge_sha256": sha256(FROZEN), "identities": identities,
              "calibration_decision_sha256": sha256(calibration_decision),
              "split": split, "summary": summary, "models": results}
    write_json(output, report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "dev", "test"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif stage == "audit":
        print(audit(read_json(FROZEN)), sha256(FROZEN))
    else:
        report = score(stage)
        print(stage, report["summary"])
