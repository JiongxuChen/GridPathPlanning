"""Post-hoc diagnostic: retain original text, remove only numeric offset sets."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/structured_pairs_v1"),
                str(HERE)]

import experiment as paired
from gridpath.experiments.config import LEDGER
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relational = load("relational_experiment", HERE / "experiment.py")
fresh = load("relational_multiseed_fresh", HERE / "multiseed_fresh.py")
OUT = ROOT / "results/relational_set_multiseed_v1/channel_ablation.json"


def main():
    reference_path = fresh.PREDICTIONS
    reference = read_json(reference_path)
    challenge = read_json(fresh.FROZEN)
    generated = read_json(fresh.GENERATED)
    assert reference["challenge_sha256"] == generated["challenge_sha256"] == sha256(fresh.FROZEN)
    assert reference["generated_sha256"] == sha256(fresh.GENERATED)
    if OUT.exists():
        saved = read_json(OUT)
        assert saved["reference_sha256"] == sha256(reference_path)
        return saved
    controlled = [(row, row["candidate_plan"])
                  for pair in challenge["groups"]["paired"]
                  for row in (pair["valid"], pair["collision"])]
    natural = [(row, output["prediction"]["text"])
               for row, output in zip(challenge["groups"]["natural_G"],
                                      generated["outputs"], strict=True)
               if output["prediction"]["failure"] is None]
    old_feature = relational.set_model.feature

    def text_only(tokenizer, grid, candidate, limit, labels=None):
        result = old_feature(tokenizer, grid, candidate, limit, labels)
        result["obstacle_offsets"] = [[] for _ in result["positions"]]
        return result

    models = {}
    with GPUBudget(LEDGER, 20, 3600, "relational_set_multiseed_v1:posthoc_channel") as budget:
        relational.set_model.feature = text_only
        try:
            for seed in (42, 43, 44):
                if seed == 42:
                    cfg, _, _, identity = relational.binding()
                    run = relational.RUN
                else:
                    multi = fresh.multi
                    previous_name, previous_run = multi.relational.NAME, multi.relational.RUN
                    multi.relational.NAME, multi.relational.RUN = multi.NAME, multi.run_path(seed)
                    try:
                        cfg, _, _, identity = multi.binding(seed)
                    finally:
                        multi.relational.NAME, multi.relational.RUN = previous_name, previous_run
                    run = multi.run_path(seed)
                assert read_json(run / "completed.json")["fingerprint"] == identity["fingerprint"]
                validate_checkpoint(run / "latest", identity["fingerprint"])
                path = run / "latest"
                tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
                net = relational.set_model.SetStepVerifier.from_pretrained(
                    path, local_files_only=True).to("cuda").eval()
                verifier = relational.SetVerifier(net, tok, cfg, budget)
                models[f"seed{seed}"] = {
                    "fingerprint": identity["fingerprint"],
                    "weights_sha256": sha256(path / "model.safetensors"),
                    "controlled": paired.score(controlled, verifier),
                    "natural_G": paired.score(natural, verifier),
                    "model_calls": verifier.calls}
                assert verifier.calls == len(controlled) + len(natural)
                del verifier, net
        finally:
            relational.set_model.feature = old_feature
    result = {"status": "posthoc_numeric_channel_removed_not_new_blind_test",
              "reference_sha256": sha256(reference_path),
              "challenge_sha256": sha256(fresh.FROZEN),
              "generated_sha256": sha256(fresh.GENERATED),
              "entry_sha256": sha256(Path(__file__)),
              "models": models}
    write_json(OUT, result)
    return result


if __name__ == "__main__":
    result = main()
    for name, model in result["models"].items():
        for panel in ("controlled", "natural_G"):
            print(name, panel, model[panel]["metrics"]["joint_diagnosis_count"])
