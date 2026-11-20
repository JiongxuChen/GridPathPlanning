"""Diagnostic-only use of the already-open matched-pairs holdout."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

import experiment as paired
import unpaired_ablation as ablation
from gridpath.experiments.config import LEDGER
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

OUT = ablation.OUT / "posthoc_holdout.json"


def main():
    challenge_path = paired.OUT / "holdout_challenge.json"
    generated_path = paired.OUT / "holdout_G_outputs.json"
    source_result_path = paired.OUT / "holdout_predictions.json"
    challenge = read_json(challenge_path)
    generated = read_json(generated_path)
    source = read_json(source_result_path)
    assert source["challenge_sha256"] == generated["challenge_sha256"] == sha256(challenge_path)
    assert source["generated_sha256"] == sha256(generated_path)
    cfg, _, _, identity = ablation.binding()
    assert read_json(ablation.RUN / "completed.json")["fingerprint"] == identity["fingerprint"]
    validate_checkpoint(ablation.RUN / "latest", identity["fingerprint"])
    if OUT.exists():
        saved = read_json(OUT)
        assert saved["fingerprint"] == identity["fingerprint"]
        assert saved["source_result_sha256"] == sha256(source_result_path)
        return saved
    controlled = [(row, row["candidate_plan"])
                  for pair in challenge["groups"]["paired"]
                  for row in (pair["valid"], pair["collision"])]
    natural = [(row, output["prediction"]["text"])
               for row, output in zip(challenge["groups"]["natural_G"],
                                      generated["outputs"], strict=True)
               if output["prediction"]["failure"] is None]
    with GPUBudget(LEDGER, 20, 3600, f"{ablation.NAME}:posthoc_holdout") as budget:
        tok = AutoTokenizer.from_pretrained(ablation.RUN / "latest", local_files_only=True)
        net = StepVerifier.from_pretrained(ablation.RUN / "latest", local_files_only=True).to("cuda").eval()
        verifier = Verifier(net, tok, cfg, budget)
        panels = {"controlled": paired.score(controlled, verifier),
                  "natural_G": paired.score(natural, verifier)}
        assert verifier.calls == len(controlled) + len(natural)
        result = {"status": "posthoc_opened_holdout_not_new_blind_test",
                  "fingerprint": identity["fingerprint"],
                  "weights_sha256": sha256(ablation.RUN / "latest/model.safetensors"),
                  "challenge_sha256": sha256(challenge_path),
                  "generated_sha256": sha256(generated_path),
                  "source_result_sha256": sha256(source_result_path),
                  "panels": panels, "model_calls": verifier.calls}
        write_json(OUT, result)
        return result


if __name__ == "__main__":
    report = main()
    for name, panel in report["panels"].items():
        cases = panel["cases"]
        print(name, sum(c["prediction"]["text"] == c["row"]["target_text"]
                        for c in cases), "/", len(cases),
              "F1", panel["metrics"]["macro_f1"],
              "joint", panel["metrics"]["joint_diagnosis_accuracy"])
