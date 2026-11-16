"""Historical paired ID/OOD G-V-repair evaluation with the frozen local V."""
from pathlib import Path
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import representation
import train
from gridpath.experiments import evaluation
from gridpath.experiments.config import completed_model
from gridpath.experiments.data import MANIFEST, load
from gridpath.planning import protocol
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint

NAME = "local_occupancy_v1"


def run(split):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from gridpath.evaluation import Predictor
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier
    from gridpath.training_helpers import seed_everything

    if split not in ("test_id", "test_ood"):
        raise ValueError("Only historical published test splits are supported")
    challenge = read_json(train.OUT / "robustness_v1_challenge.json")
    robustness = read_json(train.OUT / "robustness_v1_predictions.json")
    if (robustness["challenge_sha256"] != sha256(train.OUT / "robustness_v1_challenge.json")
            or challenge["protocol_sha256"] != sha256(Path(__file__).with_name("robustness_v1.md"))):
        raise ValueError("Robustness result/protocol was not frozen")
    vcfg, _, _, identity = train.binding()
    if robustness["local_fingerprint"] != identity["fingerprint"]:
        raise ValueError("Candidate verifier differs from robustness evaluation")
    validate_checkpoint(train.RUN / "latest", identity["fingerprint"])
    gcfg, gpath, gb = completed_model("v2", "t5_base", "generate")
    cfg = {**gcfg, "method": "improvement_verify_v1"}
    directory = ROOT / "results" / NAME / "historical_pipeline" / split
    binding = {
        "method": NAME, "split": split,
        "generator_fingerprint": gb["fingerprint"],
        "verifier_fingerprint": identity["fingerprint"],
        "dataset_sha256": sha256(MANIFEST),
        "robustness_sha256": sha256(train.OUT / "robustness_v1_predictions.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "protocol": "historical_G60_V80_R60_four_routes_one_attempt",
    }
    binding["fingerprint"] = digest(binding)
    if (directory / "binding.json").exists():
        if read_json(directory / "binding.json") != binding:
            raise ValueError("Existing historical run belongs to another identity")
    else:
        write_json(directory / "binding.json", binding)
    if (directory / "summary.json").exists():
        return read_json(directory / "summary.json")
    data = load([split + "_" + k for k in
                 ("generate", "verify", "repair_seeds", "records", "samples")])
    seed_everything(42)
    with GPUBudget(train.LEDGER, 20, 3600, f"{NAME}:t5_base:historical_{split}") as budget:
        began = time.perf_counter()
        gt = AutoTokenizer.from_pretrained(gpath, local_files_only=True)
        vt = AutoTokenizer.from_pretrained(train.RUN / "latest", local_files_only=True)
        gn = AutoModelForSeq2SeqLM.from_pretrained(gpath, local_files_only=True).to("cuda").eval()
        vn = StepVerifier.from_pretrained(train.RUN / "latest", local_files_only=True).to("cuda").eval()
        generator = Predictor(gn, gt, cfg, budget, directory / "generate_cache")
        verifier = Verifier(vn, vt, vcfg, budget, directory / "verify_cache")
        original = protocol.render
        protocol.render = representation.render
        try:
            report = evaluation.evaluate_with_predictors(
                "t5_base", cfg, data, split, generator, verifier, directory, binding)
        finally:
            protocol.render = original
        report["method"] = NAME
        report["test_used_for_selection"] = False
        report["historical_public_test_not_blind"] = True
        report["performance"] = {
            "seconds": time.perf_counter() - began,
            "actual_generate_calls": generator.calls,
            "actual_verify_calls": verifier.calls,
            "cache_hits": generator.hits + verifier.hits,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        # Same G and executor branch must remain identical to the prior run.
        old = ROOT / "results/improvement_verify_v1/t5_base" / split / "cases"
        for path in (directory / "cases").glob("*.json"):
            current, previous = read_json(path), read_json(old / path.name)
            if current["initial"] != previous["initial"]:
                raise ValueError("Historical G candidate changed: " + path.name)
            for branch in ("executor_prompt", "executor_prefix"):
                if current["outcomes"][branch] != previous["outcomes"][branch]:
                    raise ValueError("Executor control changed: " + path.name)
        write_json(directory / "summary.json", report)
        return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("test_id", "test_ood"))
    args = parser.parse_args()
    result = run(args.split)
    print(args.split, "fixed", result["fixed_verification"]["self"],
          "natural_collision", result["real_collision_verification"],
          "performance", result["performance"])
