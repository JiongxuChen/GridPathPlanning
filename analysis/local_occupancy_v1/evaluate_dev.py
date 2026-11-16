"""Development-only verifier comparison; no repair or held-out test calls."""
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import representation
import train
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.experiments.data import canonical_row, load
from gridpath.planning import protocol
from gridpath.planning.protocol import actions_of, virtual_states
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint

RESULT = train.OUT / "verification_dev.json"


def dev_items():
    data = load(["dev_verify", "dev_generate"])
    fixed = [(r, r["candidate_plan"]) for r in data["dev_verify"]]
    natural = []
    for row in data["dev_generate"]:
        path = ROOT / "results/v2/t5_base/dev/cases" / (row["sample_id"] + ".json")
        case = read_json(path)
        if case["sample_id"] != row["sample_id"] or case["row_digest"] != digest(row):
            raise ValueError("Development G case has a different source")
        if case["initial"]["failure"] is not None:
            raise ValueError("Development G interface failure needs separate scoring")
        candidate = case["initial"]["text"]
        if execute(grid_from_record(row), candidate).verification_label != case["executor_feedback"]:
            raise ValueError("Development G execution changed")
        natural.append((row, candidate))
    if len(fixed) != 80 or len(natural) != 60:
        raise ValueError("Development panel size changed")
    return fixed, natural


def evaluate_group(items, verifier):
    from gridpath.planning.verification import metrics

    scored = []
    for row, candidate in items:
        actions = actions_of(candidate)
        prediction, detail = verifier(grid_from_record(row), actions)
        annotated = canonical_row({**row, "task": "VERIFY", "candidate_plan": actions})
        scored.append({"row": annotated, "prediction": asdict(prediction), "detail": detail})
    return {"metrics": metrics([x["row"] for x in scored], scored), "cases": scored}


def collisions(items):
    group = [x for x in items if x["row"]["error_reason"] == "obstacle_collision"]
    endpoint = [x for x in group if virtual_states(
        grid_from_record(x["row"]).start, x["row"]["candidate_plan"])[-1]
        == grid_from_record(x["row"]).goal]
    def counts(rows):
        return {"count": len(rows),
                "detected": sum(x["prediction"]["failure"] is None and
                                x["prediction"]["text"].startswith("invalid |") for x in rows),
                "joint_correct": sum(x["prediction"]["failure"] is None and
                                     x["prediction"]["text"] == x["row"]["target_text"] for x in rows)}
    return {"all": counts(group), "endpoint_at_goal": counts(endpoint)}


def main():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier

    cfg, _, _, binding = train.binding()
    done = read_json(train.RUN / "completed.json")
    if done["fingerprint"] != binding["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Training is incomplete or has a different identity")
    validate_checkpoint(train.RUN / "latest", binding["fingerprint"])
    fixed, natural = dev_items()
    if RESULT.exists():
        saved = read_json(RESULT)
        if saved["training_fingerprint"] != binding["fingerprint"] or saved["evaluator_sha256"] != sha256(Path(__file__)):
            raise ValueError("Existing development result identity mismatch")
        return saved
    baseline = read_json(ROOT / "results/improvement_verify_v1/t5_base/dev/summary.json")
    original_render = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(train.LEDGER, 20, 3600, f"{train.NAME}:t5_base:dev_verification") as budget:
            tokenizer = AutoTokenizer.from_pretrained(train.RUN / "latest", local_files_only=True)
            model = StepVerifier.from_pretrained(train.RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(model, tokenizer, cfg, budget, train.OUT / "verify_cache_dev")
            controlled = evaluate_group(fixed, verifier)
            generated = evaluate_group(natural, verifier)
            result = {"training_fingerprint": binding["fingerprint"],
                      "evaluator_sha256": sha256(Path(__file__)),
                      "baseline_dev_fingerprint": baseline["fingerprint"],
                      "fixed_V80": controlled, "natural_G60": generated,
                      "natural_collision": collisions(generated["cases"]),
                      "gpu_calls": verifier.calls}
            write_json(RESULT, result)
            return result
    finally:
        protocol.render = original_render


if __name__ == "__main__":
    report = main()
    for panel in ("fixed_V80", "natural_G60"):
        m = report[panel]["metrics"]
        print(panel, {k: m[k] for k in ("classification_accuracy", "macro_f1",
              "first_error_localization_accuracy", "joint_diagnosis_accuracy")})
    print("natural_collision", report["natural_collision"])
