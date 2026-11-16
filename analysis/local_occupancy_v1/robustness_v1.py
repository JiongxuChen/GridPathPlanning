"""Frozen post-development verifier robustness challenge and paired evaluation."""
from collections import Counter
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_dev
import representation
import train
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments.config import completed_model
from gridpath.experiments.data import MANIFEST, load, new_row
from gridpath.planning import protocol
from gridpath.planning.events import Event, apply_events, choices, safe_loop
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

HERE = Path(__file__).resolve().parent
FROZEN = train.OUT / "robustness_v1_challenge.json"
RESULT = train.OUT / "robustness_v1_predictions.json"
GROUPS = ("long_valid_6", "late_collision_6", "double_6",
          "sparse_8", "dense_8", "size_10", "token_boundary")
QUOTAS = {name: 20 for name in GROUPS[:-1]} | {"token_boundary": 12}


def legal_prefix(grid, base, loops=5):
    candidate, event = safe_loop(grid, base, layout_id(grid), at=0)
    prefix = tuple(event.tokens) * loops
    assert tuple(candidate) == tuple(event.tokens) + tuple(base)
    assert execute(grid, prefix + tuple(base)).success
    return prefix


def construct(grid, base, group, number):
    collision = choices(grid, base, ErrorReason.OBSTACLE_COLLISION, "return_block")
    if group == "long_valid_6":
        prefix = legal_prefix(grid, base)
        return prefix + tuple(base), {"kind": "legal_loops", "count": 5}
    if group == "late_collision_6":
        if not collision:
            return None
        prefix = legal_prefix(grid, base)
        event = collision[0]
        candidate = prefix + apply_events(base, [event])
        if execute(grid, candidate).first_error_step < 11:
            raise AssertionError("Late collision moved earlier")
        return candidate, {"kind": "late_collision", "events": [event.to_dict()]}
    if group == "double_6":
        for event in collision:
            invalid_index = len(base) if number % 2 == 0 else 0
            if event.index == invalid_index:
                continue
            other = Event("insert_invalid_token", invalid_index, ("jump",))
            combined = apply_events(base, [event, other])
            singles = [apply_events(base, [e]) for e in (event, other)]
            if (combined not in (tuple(base), *singles)
                    and all(not execute(grid, s).success for s in singles)
                    and not execute(grid, combined).success):
                return combined, {"kind": "two_independent_events",
                                  "events": [event.to_dict(), other.to_dict()],
                                  "single_labels": [execute(grid, s).verification_label for s in singles]}
        return None
    if group == "token_boundary":
        target = (16, 24, 32, 48, 64, 72)[number // 2]
        prefix = legal_prefix(grid, base, loops=1)[:2]
        count = max(0, (target - len(base) + 1) // 2)
        candidate = prefix * count + tuple(base)
        if not execute(grid, candidate).success or len(candidate) > 72:
            return None
        return candidate, {"kind": "source_limit", "target_actions": target}
    if number < 10:
        return tuple(base), {"kind": "shortest_valid"}
    if not collision:
        return None
    event = collision[0]
    return apply_events(base, [event]), {"kind": "collision",
                                         "events": [event.to_dict()]}


def prepare():
    if FROZEN.exists():
        data = read_json(FROZEN)
        if (data["source_sha256"] != sha256(Path(__file__))
                or data["protocol_sha256"] != sha256(HERE / "robustness_v1.md")
                or data["dataset_sha256"] != sha256(MANIFEST)):
            raise ValueError("Frozen robustness source/protocol/dataset changed")
        return data
    frozen = load(["train_generate", "train_verify", "dev_samples",
                   "test_id_samples", "test_ood_samples"])
    earlier = read_json(train.OUT / "fresh_holdout.json")
    forbidden = {r["layout_id"] for rows in frozen.values() for r in rows}
    forbidden.update(r["layout_id"] for name in
                     ("fixed_positive", "fixed_hard_collision", "generate")
                     for r in earlier[name])
    used = set(forbidden)
    rng = random.Random(20261116)
    groups = {name: [] for name in GROUPS}
    attempts = Counter()
    for group in GROUPS:
        size = 8 if group in ("sparse_8", "dense_8") else 10 if group == "size_10" else 6
        low, high = ((12, 18) if group == "dense_8" else
                     (2, 6) if group == "sparse_8" else
                     (3, 12) if group == "size_10" else (2, 5))
        cells = [(r, c) for r in range(size) for c in range(size)]
        while len(groups[group]) < QUOTAS[group]:
            attempts[group] += 1
            if attempts[group] > 50000:
                raise RuntimeError("Cannot fill frozen quota: " + group)
            blocked = frozenset(rng.sample(cells, rng.randint(low, high)))
            free = [p for p in cells if p not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(size, size, blocked, start, goal)
            lid = layout_id(grid)
            if lid in used or layout_split(lid, 42) != "test":
                continue
            ref = shortest_path(grid)
            if not ref.reachable or not 2 <= ref.distance <= 14:
                continue
            try:
                made = construct(grid, ref.actions, group, len(groups[group]))
            except ValueError:
                continue
            if made is None:
                continue
            candidate, construction = made
            row = new_row(grid, "robustness_v1", "VERIFY", candidate)
            truth = execute(grid, candidate)
            if group == "late_collision_6":
                assert truth.error_reason == ErrorReason.OBSTACLE_COLLISION
                assert row["unchecked_endpoint_is_goal"]
            if group == "double_6":
                assert len({e["index"] for e in construction["events"]}) == 2
            if group == "token_boundary":
                assert truth.success and len(candidate) >= construction["target_actions"] - 1
            groups[group].append({"row": row, "construction": construction})
            used.add(lid)
    data = {"seed": 20261116, "source_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(HERE / "robustness_v1.md"),
            "dataset_sha256": sha256(MANIFEST),
            "previous_challenge_sha256": sha256(train.OUT / "fresh_holdout.json"),
            "attempts": dict(attempts), "groups": groups}
    train.OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def audit(challenge):
    assert {name: len(items) for name, items in challenge["groups"].items()} == QUOTAS
    seen = set()
    for group, items in challenge["groups"].items():
        for item in items:
            row, construction = item["row"], item["construction"]
            grid = grid_from_record(row)
            assert row["layout_id"] not in seen
            seen.add(row["layout_id"])
            assert layout_split(row["layout_id"], 42) == "test"
            assert execute(grid, row["candidate_plan"]).verification_label == row["target_text"]
            if group == "double_6":
                events = [Event(e["kind"], e["index"], tuple(e["tokens"]))
                          for e in construction["events"]]
                base = row["reference_plan"]
                assert tuple(row["candidate_plan"]) == apply_events(base, events)
                assert len({e.index for e in events}) == 2
                for event, label in zip(events, construction["single_labels"], strict=True):
                    assert execute(grid, apply_events(base, [event])).verification_label == label
                    assert label != "valid"
            if group == "late_collision_6":
                assert execute(grid, row["candidate_plan"]).first_error_step >= 11
                assert row["unchecked_endpoint_is_goal"]
    return {"layouts": len(seen), "counts": {k: len(v) for k, v in challenge["groups"].items()}}


def score():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier

    challenge = prepare()
    audit(challenge)
    cfg, _, _, identity = train.binding()
    done = read_json(train.RUN / "completed.json")
    if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Local V checkpoint identity changed")
    validate_checkpoint(train.RUN / "latest", identity["fingerprint"])
    bcfg, bpath, bb = completed_model("improvement_verify_v1", "t5_base", "verify")
    if RESULT.exists():
        saved = read_json(RESULT)
        if saved["challenge_sha256"] != sha256(FROZEN):
            raise ValueError("Result/challenge mismatch")
        return saved
    with GPUBudget(train.LEDGER, 20, 3600, f"{train.NAME}:t5_base:robustness_v1") as budget:
        bt = AutoTokenizer.from_pretrained(bpath, local_files_only=True)
        bn = StepVerifier.from_pretrained(bpath, local_files_only=True).to("cuda").eval()
        baseline = Verifier(bn, bt, bcfg, budget)
        baseline_results = {group: evaluate_dev.evaluate_group(
            [(item["row"], item["row"]["candidate_plan"]) for item in items], baseline)
            for group, items in challenge["groups"].items()}
        base_calls = baseline.calls
        del baseline, bn
        lt = AutoTokenizer.from_pretrained(train.RUN / "latest", local_files_only=True)
        ln = StepVerifier.from_pretrained(train.RUN / "latest", local_files_only=True).to("cuda").eval()
        local = Verifier(ln, lt, cfg, budget)
        original = protocol.render
        protocol.render = representation.render
        try:
            local_results = {group: evaluate_dev.evaluate_group(
                [(item["row"], item["row"]["candidate_plan"]) for item in items], local)
                for group, items in challenge["groups"].items()}
        finally:
            protocol.render = original
        result = {"challenge_sha256": sha256(FROZEN),
                  "local_fingerprint": identity["fingerprint"],
                  "baseline_fingerprint": bb["fingerprint"],
                  "baseline_calls": base_calls, "local_calls": local.calls,
                  "baseline": baseline_results, "local": local_results}
        write_json(RESULT, result)
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "audit"))
    args = parser.parse_args()
    if args.stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif args.stage == "audit":
        print(audit(prepare()))
    else:
        report = score()
        for name in ("baseline", "local"):
            for group in GROUPS:
                m = report[name][group]["metrics"]
                print(name, group, "count", m["count"], "macro_f1", m["macro_f1"],
                      "joint", m["joint_diagnosis_accuracy"],
                      "failures", m["interface_failure_count"])
