"""Frozen 6x6 candidate comparison across structured, geometric, and set V."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import importlib.util
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/structured_pairs_v1"),
                str(ROOT / "analysis/relational_set_v1"),
                str(ROOT / "analysis/local_occupancy_v1")]

import experiment as paired
import representation
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments.config import LEDGER, binding_for, completed_model
from gridpath.experiments.data import MANIFEST, canonical_row, new_row
from gridpath.planning import protocol
from gridpath.planning.events import Event, apply_events, choices, goal_endpoint, safe_loop
from gridpath.planning.model import StepVerifier
from gridpath.planning.protocol import actions_of
from gridpath.planning.verification import Verifier, metrics
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

NAME = "relational_boundary_v1"
OUT = ROOT / "results" / NAME
FROZEN = OUT / "challenge.json"
GENERATED = OUT / "G_outputs.json"
RESULT = OUT / "predictions.json"
GEOMETRY = ROOT / "checkpoints" / NAME / "geometry_snapshot"
SEED = 20261130
GROUPS = ("valid_short", "valid_long", "hard_collision", "late_collision",
          "out_of_bounds", "invalid_token", "goal_not_reached", "token_boundary")
QUOTA = 10
NATURAL = 40


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relational = load_module("relational_boundary_set",
                         ROOT / "analysis/relational_set_v1/experiment.py")
multi = load_module("relational_boundary_multi",
                    ROOT / "analysis/relational_set_v1/multiseed.py")


def source_files():
    return [Path(__file__), HERE / "protocol.md", MANIFEST,
            ROOT / "results/relational_set_multiseed_v1/exclusions.json",
            ROOT / "results/relational_set_multiseed_v1/challenge.json",
            ROOT / "results/sealed_gate_v1/challenge.json"]


def sources():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in source_files()}


def forbidden():
    ids = set(read_json(source_files()[3])["layout_ids"])
    second = read_json(source_files()[4])
    ids.update(pair["valid"]["layout_id"] for pair in second["groups"]["paired"])
    ids.update(row["layout_id"] for row in second["groups"]["natural_G"])
    sealed = read_json(source_files()[5])
    ids.update(item["row"]["layout_id"] for group in sealed["groups"].values()
               for item in group)
    assert len(ids) == 2480
    return ids


def make(grid, base, group, index):
    lid = layout_id(grid)
    if group == "valid_short":
        return tuple(base), []
    if group in ("valid_long", "late_collision", "token_boundary"):
        loop, event = safe_loop(grid, base, lid, at=0)
        prefix = tuple(event.tokens)
        repetitions = (max(0, (32 if index < 5 else 64) - len(base) + 1) // 2
                       if group == "token_boundary" else 5)
        if group != "late_collision":
            return prefix * repetitions + tuple(base), [
                {"legal_loop": event.to_dict(), "repetitions": repetitions}]
        pool = [e for e in choices(grid, base, ErrorReason.OBSTACLE_COLLISION,
                                   "return_block")
                if goal_endpoint(grid, apply_events(base, [e]))]
        if not pool:
            return None
        collision = min(pool, key=lambda e: digest([SEED, lid, e.to_dict()]))
        return prefix * repetitions + apply_events(base, [collision]), [
            {"legal_loop": event.to_dict(), "repetitions": repetitions},
            collision.to_dict()]
    if group == "goal_not_reached":
        event = Event("truncate", len(base) - 1)
    else:
        reason = {"hard_collision": ErrorReason.OBSTACLE_COLLISION,
                  "out_of_bounds": ErrorReason.OUT_OF_BOUNDS,
                  "invalid_token": ErrorReason.INVALID_ACTION_TOKEN}[group]
        kind = ("return_block" if group != "invalid_token"
                else "insert_invalid_token")
        pool = choices(grid, base, reason, kind)
        if group in ("hard_collision", "out_of_bounds"):
            pool = [e for e in pool if goal_endpoint(grid, apply_events(base, [e]))]
        if not pool:
            return None
        event = min(pool, key=lambda e: digest([SEED, lid, e.to_dict()]))
    return apply_events(base, [event]), [event.to_dict()]


def audit(challenge):
    assert challenge["seed"] == SEED and challenge["sources"] == sources()
    seen = forbidden()
    for group in (*GROUPS, "natural_G"):
        expected = NATURAL if group == "natural_G" else QUOTA
        assert len(challenge["groups"][group]) == expected
        for i, item in enumerate(challenge["groups"][group]):
            row = item["row"]
            grid = grid_from_record(row)
            lid = layout_id(grid)
            assert lid == row["layout_id"] and lid not in seen
            assert grid.rows == grid.cols == 6 and layout_split(lid, 42) == "test"
            seen.add(lid)
            if group == "natural_G":
                assert row["task"] == "GENERATE"
                continue
            made = make(grid, row["reference_plan"], group, i)
            assert made is not None and list(made[0]) == row["candidate_plan"]
            assert made[1] == item["construction"]
            truth = execute(grid, row["candidate_plan"])
            assert truth.verification_label == row["target_text"]
            assert len(grid.obstacles) in (range(2, 6) if i < 5 else range(6, 10))
            if group in ("valid_short", "valid_long", "token_boundary"):
                assert truth.success
            if group in ("hard_collision", "late_collision"):
                assert row["error_reason"] == "obstacle_collision"
                assert row["unchecked_endpoint_is_goal"]
            if group == "late_collision":
                assert truth.first_error_step >= 11
            if group == "token_boundary":
                assert len(row["candidate_plan"]) >= (32 if i < 5 else 64)
    return {"layouts": len(seen) - len(forbidden()),
            "excluded_layouts": len(forbidden())}


def freeze():
    if FROZEN.exists():
        data = read_json(FROZEN)
        audit(data)
        return data
    rng = random.Random(SEED)
    seen = forbidden()
    cells = [(r, c) for r in range(6) for c in range(6)]
    groups = {g: [] for g in (*GROUPS, "natural_G")}
    attempts = Counter()
    for group, quota in ((g, QUOTA) for g in GROUPS):
        while len(groups[group]) < quota:
            index = len(groups[group])
            attempts[group] += 1
            if attempts[group] > 100000:
                raise RuntimeError(f"Cannot fill {group}")
            blocked = frozenset(rng.sample(cells, rng.randint(
                2, 5) if index < 5 else rng.randint(6, 9)))
            free = [p for p in cells if p not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(6, 6, blocked, start, goal)
            lid = layout_id(grid)
            if lid in seen or layout_split(lid, 42) != "test":
                continue
            base = shortest_path(grid)
            if not base.reachable or not 2 <= base.distance <= 12:
                continue
            try:
                made = make(grid, base.actions, group, index)
            except ValueError:
                continue
            if made is None:
                continue
            row = new_row(grid, NAME, "VERIFY", made[0])
            if group == "late_collision" and (
                    row["error_reason"] != "obstacle_collision"
                    or execute(grid, made[0]).first_error_step < 11):
                continue
            if group in ("valid_short", "valid_long", "token_boundary") and not row["is_valid"]:
                continue
            if group == "token_boundary" and len(row["candidate_plan"]) < (
                    32 if index < 5 else 64):
                continue
            groups[group].append({"row": row, "construction": made[1]})
            seen.add(lid)
    while len(groups["natural_G"]) < NATURAL:
        attempts["natural_G"] += 1
        if attempts["natural_G"] > 100000:
            raise RuntimeError("Cannot fill natural_G")
        blocked = frozenset(rng.sample(cells, rng.randint(2, 9)))
        free = [p for p in cells if p not in blocked]
        start, goal = rng.sample(free, 2)
        grid = Grid(6, 6, blocked, start, goal)
        lid = layout_id(grid)
        base = shortest_path(grid)
        if (lid in seen or layout_split(lid, 42) != "test"
                or not base.reachable or not 2 <= base.distance <= 12):
            continue
        groups["natural_G"].append({"row": new_row(grid, NAME, "GENERATE")})
        seen.add(lid)
    data = {"seed": SEED, "sources": sources(), "attempts": dict(attempts),
            "groups": groups}
    audit(data)
    write_json(FROZEN, data)
    return data


def generate():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from gridpath.evaluation import Predictor

    challenge = freeze()
    cfg, path, identity = completed_model("v2", "t5_base", "generate")
    validate_checkpoint(path, identity["fingerprint"])
    if GENERATED.exists():
        saved = read_json(GENERATED)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generator_weights_sha256"] == sha256(path / "model.safetensors")
        return saved
    with GPUBudget(LEDGER, 20, 3600, NAME + ":G") as budget:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        net = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to("cuda").eval()
        predictor = Predictor(net, tok, cfg, budget)
        outputs = []
        for item in challenge["groups"]["natural_G"]:
            row = item["row"]
            pred = predictor(row["input_text"])
            truth = (execute(grid_from_record(row), pred.text).verification_label
                     if pred.failure is None else None)
            outputs.append({"sample_id": row["sample_id"],
                            "prediction": asdict(pred), "truth": truth})
        result = {"challenge_sha256": sha256(FROZEN),
                  "generator_fingerprint": identity["fingerprint"],
                  "generator_weights_sha256": sha256(path / "model.safetensors"),
                  "model_calls": predictor.calls, "outputs": outputs}
        write_json(GENERATED, result)
        return result


def gated(case):
    pred = dict(case["prediction"])
    if pred["failure"] is None:
        tags = list(case["detail"]["tags"])
        for i, tag in enumerate(tags[:-1]):
            if tag == 1 and case["detail"]["probabilities"][i][1] < 0.7:
                tags[i] = 0
        pred["text"] = protocol.aggregate(tags, len(case["row"]["candidate_plan"]))
    return {**case, "prediction": pred}


def score():
    from transformers import AutoTokenizer

    challenge, generated = freeze(), generate()
    assert generated["challenge_sha256"] == sha256(FROZEN)
    if RESULT.exists():
        saved = read_json(RESULT)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generated_sha256"] == sha256(GENERATED)
        return saved
    fixed = {g: [(item["row"], item["row"]["candidate_plan"])
                 for item in challenge["groups"][g]] for g in GROUPS}
    natural = []
    failures = []
    for item, output in zip(challenge["groups"]["natural_G"],
                            generated["outputs"], strict=True):
        row = item["row"]
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            assert output["truth"] == execute(
                grid_from_record(row), output["prediction"]["text"]).verification_label
            natural.append((row, output["prediction"]["text"]))
        else:
            failures.append(output)
    fixed["natural_G"] = natural
    original_cfg, original_path, original_id = completed_model(
        "improvement_verify_v1", "t5_base", "verify")
    paths = [("original", original_cfg, original_path, original_id["fingerprint"],
              StepVerifier, Verifier)]
    assert sha256(GEOMETRY / "model.safetensors") == (
        "23d96f5d072e76cab17c147539d2f6586ad6b65a56529f5f2b56023a47067b09")
    geometry_done = read_json(GEOMETRY.parent / "completed.json")
    validate_checkpoint(GEOMETRY, geometry_done["fingerprint"])
    geometry_cfg, _, _, _ = binding_for(
        "improvement_verify_v1", "t5_base", "verify")
    paths.append(("geometry_gate_0.7", geometry_cfg, GEOMETRY,
                  geometry_done["fingerprint"], StepVerifier, Verifier))
    for seed in (42, 43, 44):
        if seed == 42:
            cfg, _, _, identity = relational.binding()
            run = relational.RUN
        else:
            old_name, old_run = multi.relational.NAME, multi.relational.RUN
            multi.relational.NAME, multi.relational.RUN = multi.NAME, multi.run_path(seed)
            try:
                cfg, _, _, identity = multi.binding(seed)
            finally:
                multi.relational.NAME, multi.relational.RUN = old_name, old_run
            run = multi.run_path(seed)
        assert read_json(run / "completed.json")["fingerprint"] == identity["fingerprint"]
        validate_checkpoint(run / "latest", identity["fingerprint"])
        paths.append((f"set_seed{seed}", cfg, run / "latest",
                      identity["fingerprint"], relational.set_model.SetStepVerifier,
                      relational.SetVerifier))
    models = {}
    with GPUBudget(LEDGER, 20, 3600, NAME + ":V") as budget:
        for name, cfg, path, fingerprint, model_type, verifier_type in paths:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            net = model_type.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = verifier_type(net, tok, cfg, budget)
            original_render = protocol.render
            if name == "geometry_gate_0.7":
                protocol.render = representation.render
            try:
                panels = {group: paired.score(items, verifier)
                          for group, items in fixed.items()}
            finally:
                protocol.render = original_render
            if name == "geometry_gate_0.7":
                for panel in panels.values():
                    panel["raw_metrics"] = panel["metrics"]
                    panel["cases"] = [gated(c) for c in panel["cases"]]
                    panel["metrics"] = metrics(
                        [c["row"] for c in panel["cases"]], panel["cases"])
            models[name] = {"fingerprint": fingerprint,
                            "weights_sha256": sha256(path / "model.safetensors"),
                            "panels": panels, "model_calls": verifier.calls,
                            "source_failures": sum(
                                c["prediction"]["failure"] is not None
                                for p in panels.values() for c in p["cases"])}
            assert verifier.calls + models[name]["source_failures"] == (
                len(GROUPS) * QUOTA + len(natural))
            assert verifier.hits == 0
            del verifier, net
    result = {"status": "frozen_new_layout_same_candidate",
              "challenge_sha256": sha256(FROZEN),
              "generated_sha256": sha256(GENERATED),
              "audit": audit(challenge), "G_interface_failures": failures,
              "models": models}
    write_json(RESULT, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate", "score"))
    stage = parser.parse_args().stage
    if stage == "freeze":
        print(audit(freeze()), sha256(FROZEN))
    elif stage == "generate":
        print(generate()["model_calls"], sha256(GENERATED))
    else:
        report = score()
        print(report["status"], report["challenge_sha256"],
              report["generated_sha256"])
        for name, model in report["models"].items():
            print(name, model["model_calls"], model["source_failures"],
                  {g: (sum(c["prediction"]["text"] == c["row"]["target_text"]
                           for c in p["cases"]), len(p["cases"]))
                   for g, p in model["panels"].items()})
