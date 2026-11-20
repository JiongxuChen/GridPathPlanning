"""Fixed original-full baseline on the boundary-loss reserved panel."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1")]

import evaluate_dev
import representation
import train as local_train
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

OUT = ROOT / "results/late_collision_v2"
FROZEN = OUT / "boundary_holdout.json"
RESULT = OUT / "boundary_reference.json"
GROUPS = ("short_valid", "long_valid", "early_collision", "late_collision")


def gated(case):
    tags = list(case["detail"]["tags"])
    for i, tag in enumerate(tags[:-1]):
        if tag == 1 and case["detail"]["probabilities"][i][1] < 0.7:
            tags[i] = 0
    return protocol.aggregate(tags, len(case["row"]["candidate_plan"]))


def main():
    cfg, _, _, identity = local_train.binding()
    path = local_train.RUN / "latest"
    validate_checkpoint(path, identity["fingerprint"])
    if RESULT.exists():
        result = read_json(RESULT)
        assert result["challenge_sha256"] == sha256(FROZEN)
        assert result["fingerprint"] == identity["fingerprint"]
        return result
    pairs = read_json(FROZEN)["pairs"]
    original = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"],
                       "late_collision_v2:boundary_reference") as budget:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            network = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = Verifier(network, tok, cfg, budget)
            groups = {name: evaluate_dev.evaluate_group(
                [(pair[name], pair[name]["candidate_plan"]) for pair in pairs], verifier)
                for name in GROUPS}
            calls = verifier.calls
    finally:
        protocol.render = original
    summary = {name: {
        "full": sum(x["prediction"]["text"] == x["row"]["target_text"]
                    for x in group["cases"]),
        "full_gate_0.7": sum(gated(x) == x["row"]["target_text"]
                             for x in group["cases"]),
        "count": len(group["cases"]),
        "failures": group["metrics"]["interface_failure_count"],
    } for name, group in groups.items()}
    result = {"challenge_sha256": sha256(FROZEN), "fingerprint": identity["fingerprint"],
              "summary": summary, "groups": groups, "calls": calls}
    write_json(RESULT, result)
    return result


if __name__ == "__main__":
    print(main()["summary"])
