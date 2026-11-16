"""Freeze then evaluate the original full local-occupancy verifier once."""
from collections import Counter
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1")]

import evaluate_dev
import representation
import train
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments.data import MANIFEST, load, new_row
from gridpath.planning import protocol
from gridpath.planning.events import Event, apply_events, choices, goal_endpoint, safe_loop
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

OUT = ROOT / "results/sealed_verifier_v1"
FROZEN = OUT / "challenge.json"
RESULT = OUT / "predictions.json"
GROUPS = ("valid_short", "valid_loop", "valid_long", "hard_collision",
          "late_collision", "out_of_bounds", "invalid_token", "goal_not_reached")
SEED = 20261120
QUOTA = 20


def source_hashes():
    return {"entry": sha256(Path(__file__)),
            "protocol": sha256(HERE / "protocol.md"),
            "dataset": sha256(MANIFEST),
            "representation": sha256(Path(representation.__file__))}


def forbidden_layouts():
    source = load(["train_generate", "train_verify", "dev_samples",
                   "test_id_samples", "test_ood_samples"])
    seen = {r["layout_id"] for rows in source.values() for r in rows}
    local = train.OUT
    fresh = read_json(local / "fresh_holdout.json")
    seen.update(r["layout_id"] for group in
                ("fixed_positive", "fixed_hard_collision", "generate")
                for r in fresh[group])
    robust = read_json(local / "robustness_v1_challenge.json")
    seen.update(item["row"]["layout_id"] for rows in robust["groups"].values()
                for item in rows)
    late = read_json(ROOT / "results/late_collision_v1/prepared.json")
    seen.update(r["layout_id"] for panel in late["panels"].values()
                for pair in panel for r in pair.values())
    calibration = read_json(ROOT / "results/late_collision_v1/calibration_frozen.json")
    seen.update(r["layout_id"] for pair in calibration["pairs"]
                for r in pair.values())
    attribution = read_json(ROOT / "results/representation_attribution_v1/challenge.json")
    seen.update(r["layout_id"] for panel in attribution["panels"].values()
                for rows in panel.values() for r in rows)
    return seen


def make_candidate(grid, base, group):
    if group == "valid_short":
        return tuple(base), None
    if group == "valid_loop":
        candidate, event = safe_loop(grid, base, layout_id(grid))
        return candidate, event
    if group in ("valid_long", "late_collision"):
        _, loop = safe_loop(grid, base, layout_id(grid), at=0)
        prefix = tuple(loop.tokens) * 5
        if group == "valid_long":
            return prefix + tuple(base), {"loop": loop.to_dict()}
        pool = [e for e in choices(grid, base, ErrorReason.OBSTACLE_COLLISION,
                                   "return_block")
                if goal_endpoint(grid, apply_events(base, [e]))]
        if not pool:
            return None
        event = min(pool, key=lambda e: digest([SEED, layout_id(grid), e.to_dict()]))
        return prefix + apply_events(base, [event]), {"loop": loop.to_dict(),
                                                       "collision": event.to_dict()}
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
    return apply_events(base, [event]), event


def audit(data):
    assert data["sources"] == source_hashes()
    forbidden = forbidden_layouts()
    used = set()
    expected = {"hard_collision": "obstacle_collision",
                "late_collision": "obstacle_collision",
                "out_of_bounds": "out_of_bounds",
                "invalid_token": "invalid_action_token",
                "goal_not_reached": "goal_not_reached"}
    for group in GROUPS:
        assert len(data["groups"][group]) == QUOTA
        for item in data["groups"][group]:
            row = item["row"]
            grid = grid_from_record(row)
            lid = row["layout_id"]
            assert grid.rows == grid.cols == 6
            assert lid not in forbidden and lid not in used
            assert layout_split(lid, 42) == "test"
            used.add(lid)
            assert execute(grid, row["candidate_plan"]).verification_label == row["target_text"]
            made = make_candidate(grid, row["reference_plan"], group)
            assert made is not None and list(made[0]) == row["candidate_plan"]
            if group in expected:
                assert row["error_reason"] == expected[group]
            else:
                assert row["is_valid"]
            if group in ("hard_collision", "late_collision", "out_of_bounds"):
                assert row["unchecked_endpoint_is_goal"]
            if group == "late_collision":
                assert execute(grid, row["candidate_plan"]).first_error_step >= 11
            if group == "valid_long":
                assert len(row["candidate_plan"]) >= 12
    assert len(used) == len(GROUPS) * QUOTA
    return {"layouts": len(used), "counts": {g: QUOTA for g in GROUPS},
            "excluded_prior_layouts": len(forbidden)}


def prepare():
    if FROZEN.exists():
        data = read_json(FROZEN)
        audit(data)
        return data
    forbidden = forbidden_layouts()
    used = set(forbidden)
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    groups = {g: [] for g in GROUPS}
    attempts = Counter()
    for group in GROUPS:
        while len(groups[group]) < QUOTA:
            attempts[group] += 1
            if attempts[group] > 100000:
                raise RuntimeError("Cannot fill quota: " + group)
            blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
            free = [p for p in cells if p not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(6, 6, blocked, start, goal)
            lid = layout_id(grid)
            if lid in used or layout_split(lid, 42) != "test":
                continue
            ref = shortest_path(grid)
            if not ref.reachable or not 2 <= ref.distance <= 12:
                continue
            try:
                made = make_candidate(grid, ref.actions, group)
            except ValueError:
                continue
            if made is None:
                continue
            candidate, construction = made
            row = new_row(grid, "sealed_verifier_v1", "VERIFY", candidate)
            if group == "late_collision" and (
                    row["error_reason"] != "obstacle_collision"
                    or execute(grid, candidate).first_error_step < 11):
                continue
            if group == "valid_long" and (not row["is_valid"] or len(candidate) < 12):
                continue
            if group.startswith("valid") and not row["is_valid"]:
                continue
            if group in ("hard_collision", "out_of_bounds") and (
                    row["error_reason"] != ("obstacle_collision" if group == "hard_collision"
                                            else "out_of_bounds")):
                continue
            groups[group].append({"row": row, "construction": (
                construction.to_dict() if isinstance(construction, Event) else construction)})
            used.add(lid)
    data = {"seed": SEED, "sources": source_hashes(), "attempts": dict(attempts),
            "groups": groups}
    audit(data)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def evaluate():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier, metrics

    challenge = prepare()
    audit_summary = audit(challenge)
    cfg, _, _, identity = train.binding()
    done = read_json(train.RUN / "completed.json")
    if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Original full verifier checkpoint mismatch")
    validate_checkpoint(train.RUN / "latest", identity["fingerprint"])
    if RESULT.exists():
        saved = read_json(RESULT)
        if (saved["challenge_sha256"] != sha256(FROZEN)
                or saved["model_fingerprint"] != identity["fingerprint"]):
            raise ValueError("Existing sealed result belongs to another challenge/model")
        return saved
    original = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(train.LEDGER, 20, 3600, "sealed_verifier_v1:full:one_shot") as budget:
            path = train.RUN / "latest"
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            network = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = Verifier(network, tokenizer, cfg, budget)
            results = {group: evaluate_dev.evaluate_group(
                [(x["row"], x["row"]["candidate_plan"])
                 for x in challenge["groups"][group]], verifier)
                for group in GROUPS}
            cases = [case for group in GROUPS for case in results[group]["cases"]]
            result = {"challenge_sha256": sha256(FROZEN), "sources": source_hashes(),
                      "model_fingerprint": identity["fingerprint"],
                      "checkpoint_completed_sha256": sha256(train.RUN / "completed.json"),
                      "checkpoint_config_sha256": sha256(path / "config.json"),
                      "audit": audit_summary, "group_results": results,
                      "overall": metrics([x["row"] for x in cases], cases),
                      "model_calls": verifier.calls, "cache_hits": verifier.hits,
                      "model_seconds": verifier.seconds}
            assert verifier.calls == 160 and verifier.hits == 0
            write_json(RESULT, result)
            return result
    finally:
        protocol.render = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "audit", "evaluate"))
    args = parser.parse_args()
    if args.stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif args.stage == "audit":
        print(audit(read_json(FROZEN)), sha256(FROZEN))
    else:
        result = evaluate()
        print("challenge", result["challenge_sha256"], "fingerprint",
              result["model_fingerprint"], "calls", result["model_calls"])
        for group, panel in result["group_results"].items():
            print(group, panel["metrics"]["count"],
                  sum(x["prediction"]["text"] == x["row"]["target_text"]
                      for x in panel["cases"]),
                  panel["metrics"]["interface_failure_count"])
