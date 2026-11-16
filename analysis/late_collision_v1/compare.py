"""Paired development comparison against the frozen local-occupancy V."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import evaluate_dev
import experiment
import representation
import train as local_train
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer


def model_at(path, cfg, budget):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
    return Verifier(model, tokenizer, cfg, budget)


def compare(split):
    data = experiment.prepare()
    cfg, _, _, identity = experiment.binding()
    old_cfg, _, _, old_identity = local_train.binding()
    validate_checkpoint(local_train.RUN / "latest", old_identity["fingerprint"])
    validate_checkpoint(experiment.RUN / "latest", identity["fingerprint"])
    out = experiment.OUT / f"comparison_{split}.json"
    if out.exists():
        result = read_json(out)
        if (result["prepared_sha256"] != sha256(experiment.FROZEN) or
                result["new_fingerprint"] != identity["fingerprint"] or
                result["old_fingerprint"] != old_identity["fingerprint"]):
            raise ValueError("Comparison provenance mismatch")
        return result
    original_render = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], f"{experiment.NAME}:paired_{split}") as budget:
            old = model_at(local_train.RUN / "latest", old_cfg, budget)
            old_groups = {name: evaluate_dev.evaluate_group(
                [(pair[name], pair[name]["candidate_plan"]) for pair in data["panels"][split]], old)
                for name in ("short_valid", "long_valid", "early_collision", "late_collision")}
            old_calls = old.calls
            del old
            result = {"prepared_sha256": sha256(experiment.FROZEN),
                      "old_fingerprint": old_identity["fingerprint"],
                      "new_fingerprint": identity["fingerprint"],
                      "split": split, "old_groups": old_groups,
                      "old_calls": old_calls}
            if split == "dev":
                new = model_at(experiment.RUN / "latest", cfg, budget)
                fixed, natural = evaluate_dev.dev_items()
                result["new_original_dev"] = {
                    "fixed_V80": evaluate_dev.evaluate_group(fixed, new),
                    "natural_G60": evaluate_dev.evaluate_group(natural, new),
                    "gpu_calls": new.calls}
            write_json(out, result)
            return result
    finally:
        protocol.render = original_render


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("dev", "test"))
    report = compare(parser.parse_args().split)
    for name, group in report["old_groups"].items():
        count = sum(x["prediction"]["text"] == x["row"]["target_text"]
                    for x in group["cases"])
        print("old", name, count, "/", len(group["cases"]))
    if "new_original_dev" in report:
        for name, group in report["new_original_dev"].items():
            if name != "gpu_calls":
                count = sum(x["prediction"]["text"] == x["row"]["target_text"]
                            for x in group["cases"])
                print("new", name, count, "/", len(group["cases"]))
