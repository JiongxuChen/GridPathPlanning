"""Development-only verification comparison for the frozen natural-collision run."""
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment as exp
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.experiments.data import canonical_row, load
from gridpath.planning.protocol import actions_of, virtual_states
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json

RESULT = exp.OUT / "verification_dev.json"


def development_items():
    dev = load(["dev_verify", "dev_generate"])
    fixed = [(r, r["candidate_plan"]) for r in dev["dev_verify"]]
    g_cases = ROOT / "results/v2/t5_base/dev/cases"
    natural = []
    for row in dev["dev_generate"]:
        case = read_json(g_cases / (row["sample_id"] + ".json"))
        if case["sample_id"] != row["sample_id"] or case["row_digest"] != digest(row):
            raise ValueError("Frozen dev G row does not match original case")
        if case["initial"]["failure"] is not None:
            raise ValueError("Dev G interface failure needs separate scoring")
        candidate = case["initial"]["text"]
        if execute(grid_from_record(row), candidate).verification_label != case["executor_feedback"]:
            raise ValueError("Frozen dev G execution changed")
        natural.append((row, candidate))
    return fixed, natural


def score_items(items, verifier):
    from gridpath.planning.verification import metrics

    scored = []
    for row, candidate in items:
        actions = actions_of(candidate)
        grid = grid_from_record(row)
        prediction, detail = verifier(grid, actions)
        annotated = canonical_row({**row, "task": "VERIFY", "candidate_plan": actions})
        scored.append({"row": annotated, "prediction": asdict(prediction), "detail": detail})
    return {"metrics": metrics([x["row"] for x in scored], scored), "cases": scored}


def collision_summary(items):
    collisions = [x for x in items if x["row"]["error_reason"] == "obstacle_collision"]
    endpoint = [x for x in collisions if virtual_states(
        grid_from_record(x["row"]).start, x["row"]["candidate_plan"])[-1]
        == grid_from_record(x["row"]).goal]

    def counts(group):
        return {"count": len(group),
                "detected": sum(x["prediction"]["failure"] is None
                                and x["prediction"]["text"].startswith("invalid |") for x in group),
                "joint_correct": sum(x["prediction"]["failure"] is None
                                     and x["prediction"]["text"] == x["row"]["target_text"] for x in group)}
    return {"all": counts(collisions), "endpoint_at_goal": counts(endpoint)}


def main():
    import torch
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier
    from gridpath.training_helpers import validate_checkpoint

    cfg, _, _, binding = exp.training_binding()
    done = read_json(exp.TRAIN_RUN / "completed.json")
    if done["fingerprint"] != binding["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Training identity or budget mismatch")
    validate_checkpoint(exp.TRAIN_RUN / "latest", binding["fingerprint"])
    fixed, natural = development_items()
    if len(fixed) != 80 or len(natural) != 60:
        raise ValueError("Development panel size changed")
    if RESULT.exists():
        saved = read_json(RESULT)
        if saved["training_fingerprint"] != binding["fingerprint"] or saved["evaluator_sha256"] != sha256(Path(__file__)):
            raise ValueError("Existing result identity mismatch")
        return saved
    with GPUBudget(exp.LEDGER, 20, 3600, f"{exp.NAME}:t5_base:dev_verification") as budget:
        tokenizer = AutoTokenizer.from_pretrained(exp.TRAIN_RUN / "latest", local_files_only=True)
        network = StepVerifier.from_pretrained(exp.TRAIN_RUN / "latest", local_files_only=True).to("cuda").eval()
        verifier = Verifier(network, tokenizer, cfg, budget, exp.OUT / "verify_cache_dev")
        controlled = score_items(fixed, verifier)
        generated = score_items(natural, verifier)
        baseline = read_json(ROOT / "results/improvement_verify_v1/t5_base/dev/summary.json")
        output = {"training_fingerprint": binding["fingerprint"],
                  "evaluator_sha256": sha256(Path(__file__)),
                  "baseline_dev_fingerprint": baseline["fingerprint"],
                  "pool_sha256": sha256(exp.POOL), "overlay_sha256": sha256(exp.OVERLAY),
                  "fixed_V80": controlled, "natural_G60": generated,
                  "natural_collision": collision_summary(generated["cases"]),
                  "gpu_calls": verifier.calls}
        write_json(RESULT, output)
        return output


if __name__ == "__main__":
    report = main()
    for key in ("fixed_V80", "natural_G60"):
        m = report[key]["metrics"]
        print(key, {name: m[name] for name in ("classification_accuracy", "macro_f1",
              "first_error_localization_accuracy", "joint_diagnosis_accuracy")})
    print("natural_collision", report["natural_collision"])
