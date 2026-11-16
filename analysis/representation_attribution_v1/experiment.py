"""Frozen feature-factor experiment for the 6x6 local occupancy verifier."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import random
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import evaluate_dev
import representation
import train as local_train
import variants
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments import training
from gridpath.experiments.config import completed_model
from gridpath.experiments.data import MANIFEST, load, new_row
from gridpath.planning import protocol
from gridpath.planning.events import Event, apply_events, choices, goal_endpoint, safe_loop
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ACTION_DELTAS, ErrorReason, Grid

NAME = "representation_attribution_v1"
OUT = ROOT / "results" / NAME
FROZEN = OUT / "challenge.json"
GROUPS = ("valid_short", "valid_loop", "hard_collision",
          "out_of_bounds", "invalid_token", "goal_not_reached")
QUOTAS = {"dev": (10, 10, 20, 10, 10, 10),
          "test": (20, 20, 40, 20, 20, 20)}
SEED = 20261119


def choose(grid, base, group):
    if group == "valid_short":
        return new_row(grid, "attribution", "VERIFY", base)
    if group == "valid_loop":
        candidate, event = safe_loop(grid, base, layout_id(grid))
        return new_row(grid, "attribution", "VERIFY", candidate, positive_event=event)
    if group == "goal_not_reached":
        event = Event("truncate", len(base) - 1)
    elif group == "invalid_token":
        pool = choices(grid, base, ErrorReason.INVALID_ACTION_TOKEN,
                       "insert_invalid_token")
        event = min(pool, key=lambda e: digest([SEED, layout_id(grid), e.to_dict()]))
    else:
        reason = (ErrorReason.OBSTACLE_COLLISION if group == "hard_collision"
                  else ErrorReason.OUT_OF_BOUNDS)
        pool = [e for e in choices(grid, base, reason, "return_block")
                if goal_endpoint(grid, apply_events(base, [e]))]
        if not pool:
            return None
        event = min(pool, key=lambda e: digest([SEED, layout_id(grid), e.to_dict()]))
    candidate = apply_events(base, [event])
    row = new_row(grid, "attribution", "VERIFY", candidate, [event])
    expected = {"hard_collision": "obstacle_collision",
                "out_of_bounds": "out_of_bounds",
                "invalid_token": "invalid_action_token",
                "goal_not_reached": "goal_not_reached"}[group]
    assert row["error_reason"] == expected
    if group in ("hard_collision", "out_of_bounds"):
        assert row["unchecked_endpoint_is_goal"]
    return row


def forbidden_layouts():
    frozen = load(["train_generate", "train_verify", "dev_samples",
                   "test_id_samples", "test_ood_samples"])
    forbidden = {r["layout_id"] for rows in frozen.values() for r in rows}
    old = read_json(local_train.OUT / "fresh_holdout.json")
    forbidden.update(r["layout_id"] for group in
                     ("fixed_positive", "fixed_hard_collision", "generate") for r in old[group])
    robust = read_json(local_train.OUT / "robustness_v1_challenge.json")
    forbidden.update(x["row"]["layout_id"] for group in robust["groups"].values()
                     for x in group)
    late = read_json(ROOT / "results/late_collision_v1/prepared.json")
    forbidden.update(r["layout_id"] for panel in late["panels"].values()
                     for pair in panel for r in pair.values())
    calibration = read_json(ROOT / "results/late_collision_v1/calibration_frozen.json")
    forbidden.update(r["layout_id"] for pair in calibration["pairs"]
                     for r in pair.values())
    return forbidden


def prepare():
    sources = {"script": sha256(Path(__file__)), "protocol": sha256(HERE / "protocol.md"),
               "variants": sha256(HERE / "variants.py"), "dataset": sha256(MANIFEST)}
    if FROZEN.exists():
        data = read_json(FROZEN)
        if data["sources"] != sources:
            raise ValueError("Attribution challenge source changed")
        audit(data)
        return data
    forbidden = forbidden_layouts()
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    panels = {split: {group: [] for group in GROUPS} for split in QUOTAS}
    attempts = Counter()
    for split, quotas in QUOTAS.items():
        for group, quota in zip(GROUPS, quotas, strict=True):
            while len(panels[split][group]) < quota:
                attempts[f"{split}:{group}"] += 1
                if attempts[f"{split}:{group}"] > 100000:
                    raise RuntimeError(f"Cannot fill {split}:{group}")
                blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
                free = [p for p in cells if p not in blocked]
                start, goal = rng.sample(free, 2)
                grid = Grid(6, 6, blocked, start, goal)
                lid = layout_id(grid)
                if lid in forbidden or layout_split(lid, 42) != split:
                    continue
                ref = shortest_path(grid)
                if not ref.reachable or not 2 <= ref.distance <= 12:
                    continue
                try:
                    row = choose(grid, ref.actions, group)
                except ValueError:
                    continue
                if row is None:
                    continue
                row["split"] = "attribution_" + split
                panels[split][group].append(row)
                forbidden.add(lid)
    data = {"seed": SEED, "sources": sources, "attempts": dict(attempts),
            "panels": panels}
    audit(data)
    write_json(FROZEN, data)
    return data


def symbolic_reader(grid, candidate):
    actions = protocol.actions_of(candidate)
    lines = representation.render(grid, actions).splitlines()
    for index, action in enumerate(actions):
        if action not in ACTION_DELTAS:
            return f"invalid | step={index + 1} | reason=invalid_action_token"
        cell = re.search(r"\bCENTER (free|blocked|outside)\b", lines[index + 2])
        if cell is None:
            raise ValueError("Rendered CENTER status missing")
        if cell.group(1) != "free":
            reason = ("obstacle_collision" if cell.group(1) == "blocked"
                      else "out_of_bounds")
            return f"invalid | step={index + 1} | reason={reason}"
    end = re.search(r"goal minus end dr (-?\d+) dc (-?\d+)", lines[-1])
    if end is None:
        raise ValueError("Rendered goal offset missing")
    if end.groups() == ("0", "0"):
        return "valid"
    return f"invalid | step={len(actions) + 1} | reason=goal_not_reached"


def audit(data):
    seen = set()
    forbidden = forbidden_layouts()
    count = 0
    for split, quotas in QUOTAS.items():
        for group, quota in zip(GROUPS, quotas, strict=True):
            rows = data["panels"][split][group]
            assert len(rows) == quota
            for row in rows:
                grid = grid_from_record(row)
                lid = row["layout_id"]
                assert lid not in seen and lid not in forbidden
                assert layout_split(lid, 42) == split
                seen.add(lid)
                truth = execute(grid, row["candidate_plan"]).verification_label
                assert truth == row["target_text"]
                assert symbolic_reader(grid, row["candidate_plan"]) == truth
                if row["error_events"]:
                    from gridpath.planning.events import event_from_dict
                    events = [event_from_dict(e) for e in row["error_events"]]
                    assert len(events) == 1
                    assert tuple(row["candidate_plan"]) == apply_events(row["base_plan"], events)
                count += 1
    return {"rows": count, "layouts": len(seen),
            "groups": {s: {g: len(data["panels"][s][g]) for g in GROUPS}
                       for s in QUOTAS}}


def run_dir(mode):
    return ROOT / "checkpoints" / NAME / mode / "t5_base" / "verify"


def binding(mode):
    if mode not in variants.MODES:
        raise ValueError("Unknown representation")
    cfg, rows, dev, baseline = local_train.binding()
    cfg = {**cfg, "run_directory": str(run_dir(mode).relative_to(ROOT))}
    payload = {"experiment": NAME, "variant": mode,
               "baseline_binding": baseline["fingerprint"], "config": cfg,
               "variants_sha256": sha256(HERE / "variants.py"),
               "entry_sha256": sha256(Path(__file__)),
               "packages": {p: importlib.metadata.version(p)
                            for p in ("torch", "transformers", "numpy")}}
    return cfg, rows, dev, {"fingerprint": digest(payload), **payload}


def train(mode):
    cfg, _, _, identity = binding(mode)
    run = run_dir(mode)
    if (run / "completed.json").exists():
        done = read_json(run / "completed.json")
        if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
            raise ValueError("Variant checkpoint mismatch")
        validate_checkpoint(run / "latest", identity["fingerprint"])
        return done
    old = protocol.render, training.binding_for, training.budget
    protocol.render = lambda grid, candidate: variants.render(grid, candidate, mode)
    training.binding_for = lambda method, model, role, verify_weights=False: binding(mode)
    training.budget = lambda method, model, stage: GPUBudget(
        local_train.LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{mode}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        protocol.render, training.binding_for, training.budget = old


def score(split):
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier

    data = prepare()
    output = OUT / f"{split}_predictions.json"
    identities = {m: binding(m)[3]["fingerprint"] for m in variants.MODES}
    full_cfg, _, _, full_identity = local_train.binding()
    base_cfg, base_path, base_identity = completed_model(
        "improvement_verify_v1", "t5_base", "verify")
    identities.update(full=full_identity["fingerprint"],
                      structured=base_identity["fingerprint"])
    for mode in variants.MODES:
        validate_checkpoint(run_dir(mode) / "latest", identities[mode])
    validate_checkpoint(local_train.RUN / "latest", identities["full"])
    if output.exists():
        saved = read_json(output)
        if saved["identities"] != identities or saved["challenge_sha256"] != sha256(FROZEN):
            raise ValueError("Attribution predictions mismatch")
        return saved
    models = {
        "full": (full_cfg, local_train.RUN / "latest", representation.render),
        "center_only": (binding("center_only")[0], run_dir("center_only") / "latest",
                        lambda g, c: variants.render(g, c, "center_only")),
        "no_center": (binding("no_center")[0], run_dir("no_center") / "latest",
                      lambda g, c: variants.render(g, c, "no_center")),
        "structured": (base_cfg, base_path, protocol.render),
    }
    original = protocol.render
    results = {}
    try:
        with GPUBudget(local_train.LEDGER, full_cfg["gpu_hours_cap"],
                       full_cfg["session_wall_seconds"], f"{NAME}:{split}") as budget:
            for name, (cfg, path, renderer) in models.items():
                protocol.render = renderer
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                network = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
                verifier = Verifier(network, tokenizer, cfg, budget)
                results[name] = {group: evaluate_dev.evaluate_group(
                    [(r, r["candidate_plan"]) for r in data["panels"][split][group]], verifier)
                    for group in GROUPS}
                results[name]["gpu_calls"] = verifier.calls
                del network, verifier
    finally:
        protocol.render = original
    report = {"challenge_sha256": sha256(FROZEN), "identities": identities,
              "split": split, "models": results,
              "symbolic_probe": audit(data)}
    write_json(output, report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "dev", "test"))
    parser.add_argument("--mode", choices=variants.MODES)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif args.stage == "train":
        if not args.mode:
            parser.error("train requires --mode")
        print(train(args.mode))
    else:
        report = score(args.stage)
        for name, groups in report["models"].items():
            print(args.stage, name, {g: sum(
                x["prediction"]["text"] == x["row"]["target_text"]
                for x in groups[g]["cases"]) for g in GROUPS},
                "calls", groups["gpu_calls"])
