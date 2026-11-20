"""Freeze and evaluate the original full 3x3 verifier with its fixed 0.7 gate."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"),
                str(ROOT / "analysis/sealed_verifier_v1")]

import evaluate as prior
import evaluate_dev
import representation
import train
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments.config import completed_model
from gridpath.experiments.data import MANIFEST, load, new_row
from gridpath.planning import protocol
from gridpath.planning.events import Event, apply_events, choices, goal_endpoint, safe_loop
from gridpath.planning.verification import metrics
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

OUT = ROOT / "results/sealed_gate_v1"
FROZEN = OUT / "challenge.json"
GENERATED = OUT / "generated_candidates.json"
RESULT = OUT / "predictions.json"
SEED = 20261123
QUOTA = 20
GROUPS = ("valid_short", "valid_long", "hard_collision", "late_collision",
          "out_of_bounds", "invalid_token", "goal_not_reached",
          "double_error", "token_boundary", "natural_G")


def sources():
    return {"entry": sha256(Path(__file__)), "protocol": sha256(HERE / "protocol.md"),
            "dataset": sha256(MANIFEST),
            "representation": sha256(Path(representation.__file__)),
            "prior_challenge": sha256(prior.FROZEN),
            "late_v2": sha256(ROOT / "results/late_collision_v2/fresh_frozen.json"),
            "natural_pool": sha256(ROOT / "analysis/natural_collision_v1/pool.json")}


def forbidden_layouts():
    seen = prior.forbidden_layouts()
    seen.update(x["row"]["layout_id"] for items in
                read_json(prior.FROZEN)["groups"].values() for x in items)
    late = read_json(ROOT / "results/late_collision_v2/fresh_frozen.json")
    seen.update(row["layout_id"] for pairs in late["panels"].values()
                for pair in pairs for row in pair.values())
    seen.update(item["source"]["layout_id"] for item in
                read_json(ROOT / "analysis/natural_collision_v1/pool.json")["records"])
    return seen


def make(grid, base, group, number):
    lid = layout_id(grid)
    if group == "natural_G":
        return None, None
    if group == "valid_short":
        return tuple(base), []
    if group in ("valid_long", "late_collision", "token_boundary"):
        loop, event = safe_loop(grid, base, lid, at=0)
        prefix = tuple(event.tokens)
        if group == "token_boundary":
            target = (16, 24, 32, 48, 64)[number // 4]
            count = max(0, (target - len(base) + 1) // 2)
        else:
            count = 5
        if group != "late_collision":
            return prefix * count + tuple(base), [{"legal_loop": event.to_dict(),
                                                     "repetitions": count}]
        pool = [e for e in choices(grid, base, ErrorReason.OBSTACLE_COLLISION,
                                   "return_block")
                if goal_endpoint(grid, apply_events(base, [e]))]
        if not pool:
            return None
        collision = min(pool, key=lambda e: digest([SEED, lid, e.to_dict()]))
        return prefix * count + apply_events(base, [collision]), [
            {"legal_loop": event.to_dict(), "repetitions": count},
            collision.to_dict()]
    if group == "goal_not_reached":
        event = Event("truncate", len(base) - 1)
    elif group == "double_error":
        collision = choices(grid, base, ErrorReason.OBSTACLE_COLLISION,
                            "return_block")
        pairs = []
        for e in collision:
            for index in (0, len(base)):
                if e.index == index:
                    continue
                other = Event("insert_invalid_token", index, ("jump",))
                combined = apply_events(base, [e, other])
                if combined not in (tuple(base), apply_events(base, [e]),
                                    apply_events(base, [other])):
                    pairs.append((e, other, combined))
        if not pairs:
            return None
        a, b, candidate = min(pairs, key=lambda x: digest(
            [SEED, lid, x[0].to_dict(), x[1].to_dict()]))
        return candidate, [a.to_dict(), b.to_dict()]
    else:
        reason = {"hard_collision": ErrorReason.OBSTACLE_COLLISION,
                  "out_of_bounds": ErrorReason.OUT_OF_BOUNDS,
                  "invalid_token": ErrorReason.INVALID_ACTION_TOKEN}[group]
        kind = ("return_block" if group in ("hard_collision", "out_of_bounds")
                else "insert_invalid_token" if group == "invalid_token" else None)
        pool = choices(grid, base, reason, kind)
        if group in ("hard_collision", "out_of_bounds"):
            pool = [e for e in pool if goal_endpoint(grid, apply_events(base, [e]))]
        if not pool:
            return None
        event = min(pool, key=lambda e: digest([SEED, lid, e.to_dict()]))
    return apply_events(base, [event]), [event.to_dict()]


def audit(data):
    assert data["sources"] == sources() and data["seed"] == SEED
    forbidden = forbidden_layouts()
    used = set()
    expected = {"hard_collision": "obstacle_collision",
                "late_collision": "obstacle_collision",
                "out_of_bounds": "out_of_bounds",
                "invalid_token": "invalid_action_token",
                "goal_not_reached": "goal_not_reached"}
    for group in GROUPS:
        assert len(data["groups"][group]) == QUOTA
        for number, item in enumerate(data["groups"][group]):
            row = item["row"]
            grid = grid_from_record(row)
            lid = row["layout_id"]
            assert grid.rows == grid.cols == 6 and lid not in forbidden | used
            assert layout_split(lid, 42) == "test"
            used.add(lid)
            if group == "natural_G":
                assert row["task"] == "GENERATE"
                continue
            made = make(grid, row["reference_plan"], group, number)
            assert made and list(made[0]) == row["candidate_plan"]
            assert made[1] == item["construction"]
            truth = execute(grid, row["candidate_plan"])
            assert truth.verification_label == row["target_text"]
            if group.startswith("valid") or group == "token_boundary":
                assert row["is_valid"]
            if group in expected:
                assert row["error_reason"] == expected[group]
            if group in ("hard_collision", "late_collision", "out_of_bounds"):
                assert row["unchecked_endpoint_is_goal"]
            if group == "late_collision":
                assert truth.first_error_step >= 11
            if group == "double_error":
                assert len(row["candidate_plan"]) <= 72
                assert not row["is_valid"]
                a, b = (Event(x["kind"], x["index"], tuple(x["tokens"]))
                        for x in item["construction"])
                assert a.index != b.index
                assert all(not execute(grid, apply_events(row["reference_plan"], [e])).success
                           for e in (a, b))
            if group == "token_boundary":
                assert len(row["candidate_plan"]) >= 16
    assert len(used) == len(GROUPS) * QUOTA
    return {"layouts": len(used), "excluded_prior_layouts": len(forbidden),
            "counts": {g: QUOTA for g in GROUPS}}


def prepare():
    if FROZEN.exists():
        data = read_json(FROZEN)
        audit(data)
        return data
    used = forbidden_layouts()
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    groups = {g: [] for g in GROUPS}
    attempts = Counter()
    for group in GROUPS:
        while len(groups[group]) < QUOTA:
            attempts[group] += 1
            if attempts[group] > 100000:
                raise RuntimeError("Cannot fill " + group)
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
                made = make(grid, ref.actions, group, len(groups[group]))
            except ValueError:
                continue
            if made is None and group != "natural_G":
                continue
            candidate, construction = made
            row = new_row(grid, "sealed_gate_v1", "GENERATE" if group == "natural_G"
                          else "VERIFY", candidate)
            if group == "late_collision" and (
                    row["error_reason"] != "obstacle_collision"
                    or execute(grid, candidate).first_error_step < 11):
                continue
            if group in ("valid_short", "valid_long", "token_boundary") and not row["is_valid"]:
                continue
            if group in ("hard_collision", "out_of_bounds", "invalid_token",
                         "goal_not_reached") and row["error_reason"] != {
                             "hard_collision": "obstacle_collision",
                             "out_of_bounds": "out_of_bounds",
                             "invalid_token": "invalid_action_token",
                             "goal_not_reached": "goal_not_reached"}[group]:
                continue
            groups[group].append({"row": row, "construction": construction})
            used.add(lid)
    data = {"seed": SEED, "sources": sources(), "attempts": dict(attempts),
            "groups": groups}
    audit(data)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def generate():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from gridpath.evaluation import Predictor

    challenge = prepare()
    cfg, path, binding = completed_model("v2", "t5_base", "generate")
    validate_checkpoint(path, binding["fingerprint"])
    if GENERATED.exists():
        saved = read_json(GENERATED)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generator_fingerprint"] == binding["fingerprint"]
        return saved
    with GPUBudget(train.LEDGER, 20, 3600, "sealed_gate_v1:G:one_shot") as budget:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        net = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to("cuda").eval()
        generator = Predictor(net, tok, cfg, budget)
        outputs = []
        for item in challenge["groups"]["natural_G"]:
            row = item["row"]
            prediction = generator(row["input_text"])
            candidate = prediction.text
            truth = (execute(grid_from_record(row), candidate).verification_label
                     if prediction.failure is None else None)
            outputs.append({"sample_id": row["sample_id"],
                            "prediction": asdict(prediction), "truth": truth})
        saved = {"challenge_sha256": sha256(FROZEN),
                 "generator_fingerprint": binding["fingerprint"],
                 "generator_config_sha256": sha256(path / "config.json"),
                 "model_calls": generator.calls, "outputs": outputs}
        write_json(GENERATED, saved)
        return saved


def gated_case(case):
    prediction = dict(case["prediction"])
    detail = case["detail"]
    if prediction["failure"] is None:
        tags = list(detail["tags"])
        for i, tag in enumerate(tags[:-1]):
            if tag == 1 and detail["probabilities"][i][1] < 0.7:
                tags[i] = 0
        prediction["text"] = protocol.aggregate(tags, len(case["row"]["candidate_plan"]))
    return {**case, "prediction": prediction}


def evaluate():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier

    challenge, generated = prepare(), generate()
    assert generated["challenge_sha256"] == sha256(FROZEN)
    cfg, _, _, binding = train.binding()
    done = read_json(train.RUN / "completed.json")
    assert done["fingerprint"] == binding["fingerprint"] and done["state"]["global_step"] == 750
    validate_checkpoint(train.RUN / "latest", binding["fingerprint"])
    if RESULT.exists():
        saved = read_json(RESULT)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generated_sha256"] == sha256(GENERATED)
        assert saved["model_fingerprint"] == binding["fingerprint"]
        return saved
    original = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(train.LEDGER, 20, 3600, "sealed_gate_v1:V:one_shot") as budget:
            path = train.RUN / "latest"
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            net = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = Verifier(net, tok, cfg, budget)
            panels = {}
            for group in GROUPS:
                if group == "natural_G":
                    items = []
                    failures = []
                    for item, output in zip(challenge["groups"][group],
                                            generated["outputs"], strict=True):
                        row = item["row"]
                        assert row["sample_id"] == output["sample_id"]
                        if output["prediction"]["failure"] is not None:
                            failures.append(output)
                        else:
                            items.append((row, output["prediction"]["text"]))
                else:
                    items = [(x["row"], x["row"]["candidate_plan"])
                             for x in challenge["groups"][group]]
                    failures = []
                scored = evaluate_dev.evaluate_group(items, verifier)
                panels[group] = {"cases": scored["cases"], "raw": scored["metrics"],
                                 "gate": metrics([gated_case(c)["row"] for c in scored["cases"]],
                                                 [gated_case(c) for c in scored["cases"]]),
                                 "generator_failures": failures,
                                 "denominator": QUOTA}
            all_cases = [c for p in panels.values() for c in p["cases"]]
            result = {"challenge_sha256": sha256(FROZEN),
                      "generated_sha256": sha256(GENERATED),
                      "sources": sources(), "audit": audit(challenge),
                      "model_fingerprint": binding["fingerprint"],
                      "checkpoint_completed_sha256": sha256(train.RUN / "completed.json"),
                      "checkpoint_config_sha256": sha256(path / "config.json"),
                      "checkpoint_weights_sha256": sha256(path / "model.safetensors"),
                      "panels": panels, "raw": metrics([c["row"] for c in all_cases], all_cases),
                      "gate": metrics([c["row"] for c in all_cases],
                                      [gated_case(c) for c in all_cases]),
                      "model_calls": verifier.calls, "cache_hits": verifier.hits,
                      "model_seconds": verifier.seconds}
            assert verifier.calls == len(all_cases) - result["raw"]["interface_failure_count"]
            assert verifier.hits == 0
            write_json(RESULT, result)
            return result
    finally:
        protocol.render = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "generate", "evaluate"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif stage == "audit":
        print(audit(read_json(FROZEN)), sha256(FROZEN))
    elif stage == "generate":
        print(generate()["model_calls"], sha256(GENERATED))
    else:
        report = evaluate()
        print("challenge", report["challenge_sha256"], "generated",
              report["generated_sha256"], "calls", report["model_calls"])
        for name, panel in report["panels"].items():
            cases = panel["cases"]
            print(name, "raw", sum(c["prediction"]["text"] == c["row"]["target_text"]
                                   for c in cases), "gate",
                  sum(gated_case(c)["prediction"]["text"] == c["row"]["target_text"]
                      for c in cases), "scored", len(cases),
                  "generator_failures", len(panel["generator_failures"]),
                  "V_failures", panel["raw"]["interface_failure_count"])
