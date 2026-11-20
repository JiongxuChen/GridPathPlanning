"""Post-hoc paired diagnosis of full versus center-only occupancy verifiers."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"),
                str(ROOT / "analysis/representation_attribution_v1")]

import representation
import train as local_train
import variants
from gridpath.data_schema import grid_from_record
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

import experiment as attribution

OUT = ROOT / "results/late_collision_v2"
RESULT = OUT / "paired_representation.json"


def panels():
    robust = local_train.OUT / "robustness_v1_challenge.json"
    late = ROOT / "results/late_collision_v1/prepared.json"
    sealed = ROOT / "results/sealed_verifier_v1/challenge.json"
    sources = {"robust": robust, "late": late, "sealed": sealed}
    a, b, c = (read_json(path) for path in sources.values())
    groups = {
        "robust_late": [item["row"] for item in a["groups"]["late_collision_6"]],
        "robust_valid": [item["row"] for item in a["groups"]["long_valid_6"]],
        "sealed_late": [item["row"] for item in c["groups"]["late_collision"]],
        "sealed_long_valid": [item["row"] for item in c["groups"]["valid_long"]],
    }
    for split in ("dev", "test"):
        for name in ("short_valid", "long_valid", "early_collision", "late_collision"):
            groups[f"late_{split}_{name}"] = [
                pair[name] for pair in b["panels"][split]]
    return groups, {name: sha256(path) for name, path in sources.items()}


def evaluate():
    groups, sources = panels()
    full_cfg, _, _, full = local_train.binding()
    center_cfg, _, _, center = attribution.binding("center_only")
    models = {
        "full": (full_cfg, local_train.RUN / "latest", full["fingerprint"],
                 representation.render),
        "center_only": (center_cfg, attribution.run_dir("center_only") / "latest",
                        center["fingerprint"],
                        lambda grid, actions: variants.render(grid, actions, "center_only")),
    }
    fingerprints = {name: spec[2] for name, spec in models.items()}
    for cfg, path, fingerprint, render in models.values():
        validate_checkpoint(path, fingerprint)
    if RESULT.exists():
        saved = read_json(RESULT)
        if saved["sources"] != sources or saved["fingerprints"] != fingerprints:
            raise ValueError("Paired result identity changed")
        return saved

    original_render = protocol.render
    results = {}
    try:
        with GPUBudget(local_train.LEDGER, full_cfg["gpu_hours_cap"],
                       full_cfg["session_wall_seconds"],
                       "late_collision_v2:paired_representation") as budget:
            for name, (cfg, path, fingerprint, render) in models.items():
                protocol.render = render
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                network = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
                verifier = Verifier(network, tokenizer, cfg, budget)
                results[name] = {}
                for group, rows in groups.items():
                    cases = []
                    for row in rows:
                        prediction, detail = verifier(grid_from_record(row),
                                                      row["candidate_plan"])
                        cases.append({
                            "sample_id": row["sample_id"],
                            "truth": row["target_text"],
                            "prediction": prediction.__dict__,
                            "detail": detail,
                        })
                    results[name][group] = cases
                results[name]["model_calls"] = verifier.calls
                del verifier, network
    finally:
        protocol.render = original_render
    summary = {}
    for group, rows in groups.items():
        summary[group] = {}
        for name in models:
            cases = results[name][group]
            summary[group][name] = {
                "joint_correct": sum(x["prediction"]["text"] == x["truth"] for x in cases),
                "count": len(cases),
                "interface_failures": sum(x["prediction"]["failure"] is not None
                                          for x in cases),
                "changed_from_other": [
                    {"sample_id": x["sample_id"], "truth": x["truth"],
                     "full": results["full"][group][i]["prediction"]["text"],
                     "center_only": results["center_only"][group][i]["prediction"]["text"]}
                    for i, x in enumerate(cases)
                    if results["full"][group][i]["prediction"]["text"]
                    != results["center_only"][group][i]["prediction"]["text"]
                ] if name == "center_only" else [],
            }
    report = {"sources": sources, "fingerprints": fingerprints,
              "summary": summary, "models": results,
              "status": "post_hoc_diagnosis_not_new_blind_test"}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(RESULT, report)
    return report


if __name__ == "__main__":
    for group, counts in evaluate()["summary"].items():
        print(group, *(f"{name}={row['joint_correct']}/{row['count']}"
                       for name, row in counts.items()),
              "changed", len(counts["center_only"]["changed_from_other"]))
