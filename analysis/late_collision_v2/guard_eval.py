"""Check center-only verifier on previously public project panels."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"),
                str(ROOT / "analysis/representation_attribution_v1")]

import evaluate_dev
import train as local_train
import variants
import experiment as attribution
from gridpath.experiments.data import load
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

OUT = ROOT / "results/late_collision_v2"
RESULT = OUT / "center_public_guards.json"


def main():
    cfg, _, _, identity = attribution.binding("center_only")
    checkpoint = attribution.run_dir("center_only") / "latest"
    validate_checkpoint(checkpoint, identity["fingerprint"])
    data = load(["test_id_verify", "test_ood_verify"])
    source_hashes = {
        "id": sha256(ROOT / "data/v2/shared/test_id_verify.jsonl"),
        "ood": sha256(ROOT / "data/v2/shared/test_ood_verify.jsonl"),
        "original_dev": sha256(local_train.OUT / "verification_dev.json"),
        "script": sha256(Path(__file__)),
    }
    if RESULT.exists():
        saved = read_json(RESULT)
        if saved["fingerprint"] != identity["fingerprint"] or saved["sources"] != source_hashes:
            raise ValueError("Public guard identity changed")
        return saved
    fixed_dev, natural = evaluate_dev.dev_items()
    groups = {
        "dev_V80": fixed_dev,
        "natural_G60": natural,
        "ID_V80": [(r, r["candidate_plan"]) for r in data["test_id_verify"]],
        "OOD_V80": [(r, r["candidate_plan"]) for r in data["test_ood_verify"]],
    }
    assert [len(x) for x in groups.values()] == [80, 60, 80, 80]
    original = protocol.render
    protocol.render = lambda grid, actions: variants.render(grid, actions, "center_only")
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"],
                       "late_collision_v2:center_public_guards") as budget:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            network = StepVerifier.from_pretrained(checkpoint, local_files_only=True).to("cuda").eval()
            verifier = Verifier(network, tokenizer, cfg, budget)
            results = {group: evaluate_dev.evaluate_group(items, verifier)
                       for group, items in groups.items()}
            calls = verifier.calls
    finally:
        protocol.render = original
    summary = {name: {
        "joint_correct": sum(case["prediction"]["text"] == case["row"]["target_text"]
                             for case in result["cases"]),
        "count": len(result["cases"]),
        "interface_failures": result["metrics"]["interface_failure_count"],
        "macro_f1": result["metrics"]["macro_f1"],
    } for name, result in results.items()}
    report = {"fingerprint": identity["fingerprint"], "sources": source_hashes,
              "summary": summary, "groups": results, "model_calls": calls,
              "status": "public_post_hoc_guards"}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, report)
    return report


if __name__ == "__main__":
    print(main()["summary"])
