"""Train-only natural-collision replacement for the frozen T5-base step verifier.

Run in a separate copy of a completed v2/improvement_verify_v1 reproduction.
The original data manifest, checkpoints and source fingerprints stay untouched.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridpath.bfs import shortest_path
from gridpath.data_schema import environment_id, grid_from_record, layout_id
from gridpath.evaluation import Predictor
from gridpath.executor import execute
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, completed_model
from gridpath.experiments.data import MANIFEST, canonical_row, load, new_row
from gridpath.inference import Prediction
from gridpath.planning.protocol import virtual_states
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.types import Grid

NAME = "natural_collision_v1"
OUT = ROOT / "results" / NAME
OVERLAY = OUT / "overlay.json"
POOL = OUT / "pool.json"
TRAIN_RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"
POOL_SIZE = 1200
REPLACEMENTS = 40


def candidate_environments(train, count=POOL_SIZE):
    """New start/goal tasks on train layouts; never reuse a frozen environment."""
    layouts = {r["layout_id"]: grid_from_record(r) for r in train}
    frozen = load(["train_generate", "train_verify", "dev_samples",
                   "test_id_samples", "test_ood_samples"])
    forbidden = {row["environment_id"] for group in frozen.values() for row in group}
    rng = random.Random(42017)
    keys = sorted(layouts)
    rng.shuffle(keys)
    rows = []
    used = set(forbidden)
    attempts = 0
    while len(rows) < count and attempts < count * 100:
        source = layouts[keys[attempts % len(keys)]]
        free = [(r, c) for r in range(source.rows) for c in range(source.cols)
                if (r, c) not in source.obstacles]
        start, goal = rng.sample(free, 2)
        grid = Grid(source.rows, source.cols, source.obstacles, start, goal)
        attempts += 1
        eid = environment_id(grid)
        path = shortest_path(grid)
        if eid in used or not path.reachable or not 2 <= path.distance <= 72:
            continue
        assert layout_id(grid) in layouts
        used.add(eid)
        rows.append(new_row(grid, "train", "GENERATE"))
    if len(rows) != count:
        raise ValueError(f"Only {len(rows)} distinct train-layout tasks available")
    return rows


def pool_predictions():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if POOL.exists():
        return read_json(POOL)
    gcfg, gpath, gb = completed_model("v2", "t5_base", "generate")
    rows = candidate_environments(load(["train_generate"])["train_generate"])
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "generate_cache"
    cache.mkdir(exist_ok=True)
    with GPUBudget(LEDGER, gcfg["gpu_hours_cap"], gcfg["session_wall_seconds"],
                   f"{NAME}:t5_base:pool") as budget:
        tok = AutoTokenizer.from_pretrained(gpath, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(gpath, local_files_only=True).to("cuda").eval()
        predictor = Predictor(model, tok, gcfg, budget, cache)
        records = []
        for i, row in enumerate(rows):
            prediction = predictor(row["input_text"])
            grid = grid_from_record(row)
            truth = execute(grid, prediction.text)
            records.append({"source": row, "prediction": asdict(prediction),
                            "truth": truth.verification_label,
                            "unchecked_endpoint_is_goal":
                                virtual_states(grid.start, prediction.text)[-1] == grid.goal})
            if (i + 1) % 100 == 0:
                print(f"Frozen G pool: {i + 1}/{len(rows)}", flush=True)
        output = {"dataset_sha256": sha256(MANIFEST), "generator": gb["fingerprint"],
                  "generator_model_sha256": read_json(gpath / "checkpoint.json")["files"]["model.safetensors"],
                  "pool_size": len(rows), "records": records}
        write_json(POOL, output)
        return output


def natural_row(record, pair_id):
    source = record["source"]
    prediction = Prediction(**record["prediction"])
    grid = grid_from_record(source)
    row = {**source, "task": "VERIFY", "candidate_plan": prediction.text.lower().split(),
           "sample_id": "natural_" + digest([source["sample_id"], prediction.text])[:24],
           "paired_id": pair_id, "origin": "frozen_G_on_unseen_train_environment",
           "generator_prediction": asdict(prediction),
           "unchecked_endpoint_is_goal": record["unchecked_endpoint_is_goal"]}
    row = canonical_row(row)
    if row["target_text"] != record["truth"]:
        raise ValueError("Frozen pool label changed")
    return row


def make_overlay(pool, replacements=REPLACEMENTS):
    if OVERLAY.exists():
        existing = read_json(OVERLAY)
        if existing["pool_sha256"] != sha256(POOL):
            raise ValueError("Frozen G pool changed after overlay creation")
        return existing
    original = load(["train_verify"])["train_verify"]
    if sha256(MANIFEST) != pool["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    train_layouts = {r["layout_id"] for r in load(["train_generate"])["train_generate"]}
    reserved = {r["environment_id"] for r in original}
    candidates = [r for r in pool["records"] if r["prediction"]["failure"] is None
                  and r["source"]["layout_id"] in train_layouts
                  and r["source"]["environment_id"] not in reserved]
    hard = [r for r in candidates if "reason=obstacle_collision" in r["truth"]
            and r["unchecked_endpoint_is_goal"]]
    valid = [r for r in candidates if r["truth"] == "valid"]
    hard.sort(key=lambda r: digest([42, "collision", r["source"]["sample_id"]]))
    if len(hard) < replacements or len(valid) < replacements:
        raise ValueError(f"Insufficient natural pairs: {len(hard)} hard collisions, {len(valid)} valid; "
                         f"need {replacements}. No silent quota reduction.")
    chosen = hard[:replacements]
    available = {r["source"]["sample_id"]: r for r in valid}
    pairs = []
    for bad in chosen:
        b = bad["source"]
        match = min(available.values(), key=lambda r: (
            r["source"]["layout_id"] != b["layout_id"],
            abs(r["source"]["shortest_distance"] - b["shortest_distance"]),
            abs(len(r["prediction"]["text"].split()) - len(bad["prediction"]["text"].split())),
            digest([42, "valid", r["source"]["sample_id"]])))
        available.pop(match["source"]["sample_id"])
        pair_id = "natural_pair_" + digest([b["sample_id"], match["source"]["sample_id"]])[:24]
        pairs.extend([natural_row(bad, pair_id), natural_row(match, pair_id)])
    remove_positive = sorted((r for r in original if r["is_valid"]),
                             key=lambda r: digest([42, "replace_positive", r["sample_id"]]))[:replacements]
    remove_collision = sorted((r for r in original if r["error_reason"] == "obstacle_collision"),
                              key=lambda r: digest([42, "replace_collision", r["sample_id"]]))[:replacements]
    removed = {r["sample_id"] for r in remove_positive + remove_collision}
    rows = [r for r in original if r["sample_id"] not in removed] + pairs
    assert len(rows) == 1000 and len({r["sample_id"] for r in rows}) == 1000
    assert Counter(r["is_valid"] for r in rows) == {True: 500, False: 500}
    assert all(r["layout_id"] in train_layouts for r in pairs)
    output = {"pool_sha256": sha256(POOL), "original_manifest_sha256": sha256(MANIFEST),
              "replacements_per_class": replacements, "removed_ids": sorted(removed),
              "natural_pair_ids": sorted({r["paired_id"] for r in pairs}),
              "available_hard_collisions": len(hard), "available_valid": len(valid),
              "same_layout_pairs": sum(a["layout_id"] == b["layout_id"] for a, b in zip(pairs[::2], pairs[1::2])),
              "rows": rows}
    write_json(OVERLAY, output)
    return output


def training_binding():
    from gridpath.experiments.config import binding_for

    base_cfg, _, dev, base = binding_for("improvement_verify_v1", "t5_base", "verify", True)
    overlay = make_overlay(read_json(POOL))
    cfg = {**base_cfg, "run_directory": str(TRAIN_RUN.relative_to(ROOT))}
    payload = {"experiment": NAME, "base_binding": base["fingerprint"],
               "config": cfg, "overlay_sha256": sha256(OVERLAY),
               "script_sha256": sha256(Path(__file__)),
               "generator": read_json(POOL)["generator"],
               "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")}}
    return cfg, overlay["rows"], dev, {"fingerprint": digest(payload), **payload}


def train():
    if not OVERLAY.exists():
        raise FileNotFoundError("Run pool first")
    if (TRAIN_RUN / "completed.json").exists():
        cfg, rows, _, binding = training_binding()
        done = read_json(TRAIN_RUN / "completed.json")
        if done["fingerprint"] != binding["fingerprint"] or done["state"]["consumed_examples"] != 3 * len(rows):
            raise ValueError("Completed experiment binding mismatch")
        return done
    old_binding, old_budget = training.binding_for, training.budget
    training.binding_for = lambda method, model, role, verify_weights=False: training_binding()
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, 20, 3600, f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        training.binding_for, training.budget = old_binding, old_budget


def evaluate():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier, metrics

    cfg, rows, _, binding = training_binding()
    done = read_json(TRAIN_RUN / "completed.json")
    if done["fingerprint"] != binding["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Experiment must complete 750 updates before dev evaluation")
    dev = load(["dev_verify", "dev_generate"])
    cases = read_json(ROOT / "results/v2/t5_base/dev/summary.json")
    source = ROOT / "results/v2/t5_base/dev/cases"
    candidates = []
    for row in dev["dev_generate"]:
        case = read_json(source / (row["sample_id"] + ".json"))
        candidates.append((row, case["initial"]["text"]))
    result_path = OUT / "summary_dev.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["binding"] != binding["fingerprint"]:
            raise ValueError("Existing dev result belongs to another model")
        return result
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:t5_base:dev") as resource:
        from gridpath.training_helpers import validate_checkpoint
        validate_checkpoint(TRAIN_RUN / "latest", binding["fingerprint"])
        tok = AutoTokenizer.from_pretrained(TRAIN_RUN / "latest", local_files_only=True)
        model = StepVerifier.from_pretrained(TRAIN_RUN / "latest", local_files_only=True).to("cuda").eval()
        verifier = Verifier(model, tok, cfg, resource, OUT / "verify_cache_dev")
        fixed, natural = [], []
        for group, items, target in (
            ("fixed", [(r, r["candidate_plan"]) for r in dev["dev_verify"]], fixed),
            ("natural", candidates, natural),
        ):
            for row, candidate in items:
                grid = grid_from_record(row)
                pred, detail = verifier(grid, candidate)
                truth = execute(grid, candidate)
                annotated = canonical_row({**row, "task": "VERIFY",
                    "candidate_plan": candidate.split(), "sample_id": row["sample_id"]})
                target.append({"row": annotated, "prediction": asdict(pred), "detail": detail,
                               "truth": truth.verification_label})
        natural_collisions = [x for x in natural if "reason=obstacle_collision" in x["truth"]]
        endpoint = [x for x in natural_collisions if virtual_states(
            grid_from_record(x["row"]).start, x["row"]["candidate_plan"])[-1]
            == grid_from_record(x["row"]).goal]
        def summarize(items):
            return metrics([x["row"] for x in items], [
                {"prediction": x["prediction"], "detail": x["detail"]} for x in items])
        output = {"binding": binding["fingerprint"], "baseline_dev_fingerprint": cases["fingerprint"],
                  "fixed_V80": summarize(fixed), "natural_G60": summarize(natural),
                  "natural_collision": {"count": len(natural_collisions),
                      "detected": sum(x["prediction"]["text"] != "valid" for x in natural_collisions),
                      "joint_correct": sum(x["prediction"]["text"] == x["truth"] for x in natural_collisions)},
                  "endpoint_goal_collision": {"count": len(endpoint),
                      "detected": sum(x["prediction"]["text"] != "valid" for x in endpoint),
                      "joint_correct": sum(x["prediction"]["text"] == x["truth"] for x in endpoint)},
                  "fixed_cases": fixed, "natural_cases": natural}
        write_json(result_path, output)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pool", "train", "evaluate"))
    args = parser.parse_args()
    if args.stage == "pool":
        pool = pool_predictions()
        overlay = make_overlay(pool)
        print({k: overlay[k] for k in ("replacements_per_class", "available_hard_collisions",
                                       "available_valid", "same_layout_pairs")})
    elif args.stage == "train":
        print(train())
    else:
        report = evaluate()
        print({k: report[k] for k in ("fixed_V80", "natural_G60", "natural_collision",
                                      "endpoint_goal_collision")})


if __name__ == "__main__":
    main()
