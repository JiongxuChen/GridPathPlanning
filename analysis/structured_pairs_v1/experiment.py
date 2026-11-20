"""Equal-budget structured-offset V with natural hard negatives and matched legal paths."""
from collections import Counter
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/natural_collision_v1")]

import experiment as prior
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, binding_for
from gridpath.experiments.data import MANIFEST, load, new_row, canonical_row
from gridpath.planning.protocol import actions_of
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.planning.verification import metrics, Verifier

NAME = "structured_pairs_v1"
OUT = ROOT / "results" / NAME
POOL = ROOT / "analysis/natural_collision_v1/pool.json"
OVERLAY = OUT / "overlay.json"
RUN = ROOT / "checkpoints" / NAME / "t5_base/verify"
DEV = OUT / "verification_dev.json"
QUOTAS = {1: 7, 2: 6, 3: 9, 4: 7, 5: 7, 6: 2, 7: 2}
SEED = 20261124


def step(text):
    return execute_label_step(text)


def execute_label_step(text):
    from gridpath.metrics import parse_verification_label
    parsed = parse_verification_label(text)
    if parsed is None or parsed.valid:
        raise ValueError("Expected invalid label")
    return parsed.step


def prepare():
    source = {"script": sha256(Path(__file__)),
              "protocol": sha256(HERE / "protocol.md"),
              "pool": sha256(POOL), "dataset": sha256(MANIFEST)}
    if OVERLAY.exists():
        saved = read_json(OVERLAY)
        if saved["sources"] != source:
            raise ValueError("Overlay source changed")
        audit(saved)
        return saved
    pool = read_json(POOL)
    assert pool["dataset_sha256"] == source["dataset"]
    original = load(["train_verify", "train_generate", "dev_samples",
                     "test_id_samples", "test_ood_samples"])
    base = original["train_verify"]
    train_layouts = {x["layout_id"] for x in original["train_generate"]}
    forbidden_envs = {x["environment_id"] for key in
                      ("train_verify", "dev_samples", "test_id_samples",
                       "test_ood_samples") for x in original[key]}
    picked, pairs = [], []
    for target, count in QUOTAS.items():
        candidates = [record for record in pool["records"]
                      if record["prediction"]["failure"] is None
                      and record["truth"].endswith("reason=obstacle_collision")
                      and record["unchecked_endpoint_is_goal"]
                      and execute_label_step(record["truth"]) == target
                      and record["source"]["layout_id"] in train_layouts
                      and record["source"]["environment_id"] not in forbidden_envs
                      and len(actions_of(record["prediction"]["text"]))
                      == record["source"]["shortest_distance"]]
        candidates.sort(key=lambda x: digest(
            [SEED, "natural", x["source"]["sample_id"]]))
        if len(candidates) < count:
            raise ValueError(f"Insufficient natural collision step {target}")
        for record in candidates[:count]:
            source_row = record["source"]
            grid = grid_from_record(source_row)
            legal = tuple(shortest_path(grid).actions)
            bad = tuple(actions_of(record["prediction"]["text"]))
            if bad == legal or execute(grid, legal).verification_label != "valid":
                raise ValueError("Matched legal counterfactual failed")
            pair_id = "matched_" + digest([source_row["sample_id"], bad])[:24]
            negative = prior.natural_row(record, pair_id)
            positive = new_row(grid, "train", "VERIFY", legal)
            positive = canonical_row({**positive, "paired_id": pair_id,
                                      "origin": "same_environment_shortest_counterfactual"})
            assert negative["target_text"] == record["truth"]
            assert negative["candidate_plan"] != positive["candidate_plan"]
            assert len(negative["candidate_plan"]) == len(positive["candidate_plan"])
            assert negative["layout_id"] == positive["layout_id"]
            assert negative["environment_id"] == positive["environment_id"]
            picked.append(source_row["sample_id"])
            pairs.extend((negative, positive))
    removed = []
    for target, count in QUOTAS.items():
        options = [row for row in base if row["error_reason"] == "obstacle_collision"
                   and row["unchecked_endpoint_is_goal"]
                   and execute_label_step(row["target_text"]) == target]
        options.sort(key=lambda x: digest([SEED, "remove_bad", x["sample_id"]]))
        if len(options) < count:
            raise ValueError(f"Insufficient original hard collision step {target}")
        removed.extend(x["sample_id"] for x in options[:count])
    optimal = [row for row in base if row["is_valid"]
               and len(row["candidate_plan"]) == row["shortest_distance"]]
    optimal.sort(key=lambda x: digest([SEED, "remove_valid", x["sample_id"]]))
    removed.extend(x["sample_id"] for x in optimal[:40])
    kept = [row for row in base if row["sample_id"] not in set(removed)]
    output = {"sources": source, "quotas": {str(k): v for k, v in QUOTAS.items()},
              "removed_ids": sorted(removed), "selected_pool_ids": picked,
              "rows": kept + pairs}
    audit(output)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OVERLAY, output)
    return output


def audit(data):
    assert data["sources"] == {"script": sha256(Path(__file__)),
                                "protocol": sha256(HERE / "protocol.md"),
                                "pool": sha256(POOL), "dataset": sha256(MANIFEST)}
    rows = data["rows"]
    assert len(rows) == 1000 and len({r["sample_id"] for r in rows}) == 1000
    assert Counter(r["is_valid"] for r in rows) == {True: 500, False: 500}
    assert Counter(r["error_reason"] for r in rows if not r["is_valid"]) == {
        x: 125 for x in ("obstacle_collision", "out_of_bounds",
                         "invalid_action_token", "goal_not_reached")}
    assert sum(r["error_reason"] == "obstacle_collision"
               and r["unchecked_endpoint_is_goal"] for r in rows) == 83
    assert sum(r["is_valid"] and len(r["candidate_plan"]) > r["shortest_distance"]
               for r in rows) == 200
    assert len(data["selected_pool_ids"]) == len(set(data["selected_pool_ids"])) == 40
    new = [r for r in rows if r.get("origin") in (
        "frozen_G_on_unseen_train_environment",
        "same_environment_shortest_counterfactual")]
    assert len(new) == 80
    groups = {}
    for row in new:
        assert execute(grid_from_record(row), row["candidate_plan"]).verification_label == row["target_text"]
        groups.setdefault(row["paired_id"], []).append(row)
    assert len(groups) == 40
    for pair in groups.values():
        assert len(pair) == 2 and sum(x["is_valid"] for x in pair) == 1
        assert pair[0]["environment_id"] == pair[1]["environment_id"]
        assert len(pair[0]["candidate_plan"]) == len(pair[1]["candidate_plan"])
    return {"rows": len(rows), "pairs": len(groups),
            "hard_collisions": 83, "nonoptimal_valid": 200}


def binding():
    cfg, _, dev, base = binding_for("improvement_verify_v1", "t5_base",
                                    "verify", True)
    overlay = prepare()
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {"experiment": NAME, "base_binding": base["fingerprint"],
               "config": cfg, "overlay_sha256": sha256(OVERLAY),
               "script_sha256": sha256(Path(__file__)),
               "packages": {p: importlib.metadata.version(p)
                            for p in ("torch", "transformers", "numpy")}}
    return cfg, overlay["rows"], dev, {"fingerprint": digest(payload), **payload}


def train():
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
            raise ValueError("Completed checkpoint mismatch")
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    original_binding, original_budget = training.binding_for, training.budget
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        training.binding_for, training.budget = original_binding, original_budget


def development_items():
    from gridpath.training_config import digest
    original = load(["dev_verify", "dev_generate"])
    fixed = [(r, r["candidate_plan"]) for r in original["dev_verify"]]
    natural = []
    for row in original["dev_generate"]:
        case = read_json(ROOT / "results/v2/t5_base/dev/cases" /
                         (row["sample_id"] + ".json"))
        assert case["row_digest"] == digest(row)
        assert case["initial"]["failure"] is None
        natural.append((row, case["initial"]["text"]))
    assert len(fixed) == 80 and len(natural) == 60
    return fixed, natural


def score(items, verifier):
    from gridpath.planning.protocol import actions_of
    cases = []
    for row, candidate in items:
        actions = actions_of(candidate)
        prediction, detail = verifier(grid_from_record(row), actions)
        annotated = canonical_row({**row, "task": "VERIFY",
                                   "candidate_plan": actions})
        cases.append({"row": annotated, "prediction": asdict(prediction),
                      "detail": detail})
    return {"metrics": metrics([x["row"] for x in cases], cases),
            "cases": cases}


def evaluate():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier

    cfg, _, _, identity = binding()
    done = read_json(RUN / "completed.json")
    assert done["fingerprint"] == identity["fingerprint"]
    assert done["state"]["global_step"] == 750
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    if DEV.exists():
        saved = read_json(DEV)
        assert saved["fingerprint"] == identity["fingerprint"]
        return saved
    fixed, natural = development_items()
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:t5_base:dev") as budget:
        tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
        net = StepVerifier.from_pretrained(RUN / "latest", local_files_only=True).to("cuda").eval()
        verifier = Verifier(net, tok, cfg, budget)
        panels = {"fixed_V80": score(fixed, verifier),
                  "natural_G60": score(natural, verifier)}
        assert verifier.calls == 140 and verifier.hits == 0
        result = {"fingerprint": identity["fingerprint"],
                  "overlay_sha256": sha256(OVERLAY),
                  "model_sha256": sha256(RUN / "latest/model.safetensors"),
                  "panels": panels, "model_calls": verifier.calls}
        write_json(DEV, result)
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "train", "dev"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(OVERLAY))
    elif stage == "audit":
        print(audit(read_json(OVERLAY)), sha256(OVERLAY))
    elif stage == "train":
        print(train())
    else:
        result = evaluate()
        for name, panel in result["panels"].items():
            print(name, panel["metrics"]["macro_f1"],
                  panel["metrics"]["joint_diagnosis_accuracy"],
                  panel["metrics"]["interface_failure_count"])
