"""Post-hoc fixed-model comparisons across two already opened holdouts."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"),
                str(HERE)]

import aligned_input as aligned
import aligned_representation
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
RESULT = OUT / "cross_holdout.json"
GROUPS = ("short_valid", "long_valid", "early_collision", "late_collision")


def main():
    full_cfg, _, _, full = local_train.binding()
    boundary_cfg, _, _, boundary = aligned.boundary.binding()
    aligned_cfg, _, _, aligned_id = aligned.binding()
    models = {
        "aligned_on_boundary": (aligned_cfg, aligned.RUN / "latest",
                                aligned_id["fingerprint"], aligned_representation.render,
                                aligned.boundary.FROZEN),
        "boundary_on_aligned": (boundary_cfg, aligned.boundary.RUN / "latest",
                                boundary["fingerprint"], representation.render,
                                aligned.FROZEN),
        "full_on_aligned": (full_cfg, local_train.RUN / "latest",
                            full["fingerprint"], representation.render,
                            aligned.FROZEN),
    }
    identities = {name: {"fingerprint": identity, "panel": sha256(panel)}
                  for name, (cfg, path, identity, render, panel) in models.items()}
    for cfg, path, identity, render, panel in models.values():
        validate_checkpoint(path, identity)
    if RESULT.exists():
        report = read_json(RESULT)
        assert report["identities"] == identities
        return report
    original = protocol.render
    results = {}
    try:
        with GPUBudget(local_train.LEDGER, full_cfg["gpu_hours_cap"],
                       full_cfg["session_wall_seconds"],
                       "late_collision_v2:cross_holdout") as budget:
            for name, (cfg, path, identity, render, panel) in models.items():
                protocol.render = render
                tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
                network = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
                verifier = Verifier(network, tok, cfg, budget)
                pairs = read_json(panel)["pairs"]
                results[name] = {group: evaluate_dev.evaluate_group(
                    [(pair[group], pair[group]["candidate_plan"]) for pair in pairs], verifier)
                    for group in GROUPS}
                results[name]["calls"] = verifier.calls
                del verifier, network
    finally:
        protocol.render = original
    summary = {name: {group: {
        "correct": sum(case["prediction"]["text"] == case["row"]["target_text"]
                       for case in results[name][group]["cases"]),
        "count": len(results[name][group]["cases"]),
        "failures": results[name][group]["metrics"]["interface_failure_count"],
    } for group in GROUPS} for name in models}
    report = {"identities": identities, "summary": summary, "models": results,
              "status": "post_hoc_cross_panel_not_model_selection"}
    write_json(RESULT, report)
    return report


if __name__ == "__main__":
    print(main()["summary"])
