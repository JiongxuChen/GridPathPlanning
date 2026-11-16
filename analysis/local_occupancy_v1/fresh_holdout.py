"""One-shot fresh-layout verifier challenge, frozen before any model inference."""
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
from gridpath.evaluation import Predictor
from gridpath.executor import execute
from gridpath.experiments.config import completed_model
from gridpath.experiments.data import MANIFEST, load, new_row
from gridpath.planning import protocol
from gridpath.planning.events import apply_events, choices, goal_endpoint
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

FROZEN = train.OUT / "fresh_holdout.json"
RESULT = train.OUT / "fresh_holdout_predictions.json"


def prepare():
    if FROZEN.exists():
        data = read_json(FROZEN)
        if data["source_sha256"] != sha256(Path(__file__)) or data["dataset_sha256"] != sha256(MANIFEST):
            raise ValueError("Frozen fresh challenge has a different source")
        return data
    frozen = load(["train_generate", "train_verify", "dev_samples",
                   "test_id_samples", "test_ood_samples"])
    forbidden = {r["layout_id"] for group in frozen.values() for r in group}
    rng = random.Random(20261115)
    cells = [(r, c) for r in range(6) for c in range(6)]
    positive, negative, generated = [], [], []
    used = set(forbidden)
    attempts = 0
    while (len(positive), len(negative), len(generated)) != (40, 40, 60):
        attempts += 1
        if attempts > 100000:
            raise ValueError("Could not meet frozen fresh-layout quotas")
        blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
        free = [cell for cell in cells if cell not in blocked]
        start, goal = rng.sample(free, 2)
        grid = Grid(6, 6, blocked, start, goal)
        lid = layout_id(grid)
        if lid in used or layout_split(lid, 42) != "test":
            continue
        reference = shortest_path(grid)
        if not reference.reachable or not 2 <= reference.distance <= 12:
            continue
        if len(negative) < 40:
            hard = [event for event in choices(grid, reference.actions,
                                               ErrorReason.OBSTACLE_COLLISION, "return_block")
                    if goal_endpoint(grid, apply_events(reference.actions, [event]))]
            if hard:
                event = min(hard, key=lambda e: digest([42, lid, e.to_dict()]))
                row = new_row(grid, "fresh_holdout", "VERIFY",
                              apply_events(reference.actions, [event]), [event])
                if row["error_reason"] != "obstacle_collision" or not row["unchecked_endpoint_is_goal"]:
                    raise ValueError("Fresh hard collision changed")
                negative.append(row)
                used.add(lid)
                continue
        if len(positive) < 40:
            positive.append(new_row(grid, "fresh_holdout", "VERIFY", reference.actions))
        elif len(generated) < 60:
            generated.append(new_row(grid, "fresh_holdout", "GENERATE"))
        else:
            continue
        used.add(lid)
    data = {"source_sha256": sha256(Path(__file__)), "dataset_sha256": sha256(MANIFEST),
            "seed": 20261115, "attempts": attempts, "fixed_positive": positive,
            "fixed_hard_collision": negative, "generate": generated}
    train.OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def score():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier

    frozen = prepare()
    cfg, _, _, identity = train.binding()
    done = read_json(train.RUN / "completed.json")
    if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Local verifier training identity changed")
    validate_checkpoint(train.RUN / "latest", identity["fingerprint"])
    gcfg, gpath, gb = completed_model("v2", "t5_base", "generate")
    bcfg, bpath, bb = completed_model("improvement_verify_v1", "t5_base", "verify")
    if RESULT.exists():
        saved = read_json(RESULT)
        if (saved["challenge_sha256"] != sha256(FROZEN) or
                saved["local_fingerprint"] != identity["fingerprint"] or
                saved["baseline_fingerprint"] != bb["fingerprint"]):
            raise ValueError("Existing fresh challenge result identity mismatch")
        return saved
    fixed = [(r, r["candidate_plan"]) for r in
             frozen["fixed_positive"] + frozen["fixed_hard_collision"]]
    with GPUBudget(train.LEDGER, 20, 3600, f"{train.NAME}:t5_base:fresh_holdout") as budget:
        gt = AutoTokenizer.from_pretrained(gpath, local_files_only=True)
        gn = AutoModelForSeq2SeqLM.from_pretrained(gpath, local_files_only=True).to("cuda").eval()
        generator = Predictor(gn, gt, gcfg, budget, train.OUT / "fresh_generate_cache")
        natural = []
        for row in frozen["generate"]:
            prediction = generator(row["input_text"])
            if prediction.failure is not None:
                raise ValueError("Fresh G interface failure needs separate scoring")
            natural.append((row, prediction.text))
        g_outputs = [{"sample_id": r["sample_id"], "candidate": c,
                      "truth": execute(grid_from_record(r), c).verification_label}
                     for r, c in natural]
        write_json(train.OUT / "fresh_G_outputs.json",
                   {"challenge_sha256": sha256(FROZEN), "generator_fingerprint": gb["fingerprint"],
                    "outputs": g_outputs})
        vt = AutoTokenizer.from_pretrained(bpath, local_files_only=True)
        baseline = StepVerifier.from_pretrained(bpath, local_files_only=True).to("cuda").eval()
        base_v = Verifier(baseline, vt, bcfg, budget, train.OUT / "fresh_baseline_cache")
        base_fixed = evaluate_dev.evaluate_group(fixed, base_v)
        base_natural = evaluate_dev.evaluate_group(natural, base_v)
        base_calls, base_hits = base_v.calls, base_v.hits
        del baseline, base_v
        lt = AutoTokenizer.from_pretrained(train.RUN / "latest", local_files_only=True)
        local = StepVerifier.from_pretrained(train.RUN / "latest", local_files_only=True).to("cuda").eval()
        local_v = Verifier(local, lt, cfg, budget, train.OUT / "fresh_local_cache")
        original_render = protocol.render
        protocol.render = representation.render
        try:
            local_fixed = evaluate_dev.evaluate_group(fixed, local_v)
            local_natural = evaluate_dev.evaluate_group(natural, local_v)
        finally:
            protocol.render = original_render
        result = {"challenge_sha256": sha256(FROZEN), "local_fingerprint": identity["fingerprint"],
                  "baseline_fingerprint": bb["fingerprint"], "generator_fingerprint": gb["fingerprint"],
                  "fixed_counts": dict(Counter(r["error_reason"] or "valid" for r, _ in fixed)),
                  "generated_counts": dict(Counter(x["truth"].split("reason=")[-1] if "reason=" in x["truth"]
                                                   else "valid" for x in g_outputs)),
                  "baseline": {"fixed": base_fixed, "natural": base_natural,
                               "natural_collision": evaluate_dev.collisions(base_natural["cases"])},
                  "local": {"fixed": local_fixed, "natural": local_natural,
                            "natural_collision": evaluate_dev.collisions(local_natural["cases"])},
                  "generator_calls": generator.calls,
                  "baseline_verifier_calls": base_calls, "baseline_cache_hits": base_hits,
                  "local_verifier_calls": local_v.calls, "local_cache_hits": local_v.hits}
        write_json(RESULT, result)
        return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "evaluate"))
    a = p.parse_args()
    if a.stage == "prepare":
        frozen = prepare()
        print({"fixed_positive": len(frozen["fixed_positive"]),
               "fixed_hard_collision": len(frozen["fixed_hard_collision"]),
               "generate": len(frozen["generate"]), "attempts": frozen["attempts"],
               "sha256": sha256(FROZEN)})
    else:
        report = score()
        for name in ("baseline", "local"):
            print(name, {key: {"macro_f1": report[name][key]["metrics"]["macro_f1"],
                                "joint": report[name][key]["metrics"]["joint_diagnosis_accuracy"]}
                         for key in ("fixed", "natural")}, report[name]["natural_collision"])
