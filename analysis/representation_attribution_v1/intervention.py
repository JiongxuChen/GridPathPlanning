"""Exploratory, inconsistent CENTER-token interventions on frozen full V."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import experiment
import representation
import train as local_train
from gridpath.data_schema import grid_from_record
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

RESULT = experiment.OUT / "center_flip_test.json"


def run():
    data = experiment.prepare()
    baseline = read_json(experiment.OUT / "test_predictions.json")
    cfg, _, _, identity = local_train.binding()
    validate_checkpoint(local_train.RUN / "latest", identity["fingerprint"])
    if RESULT.exists():
        saved = read_json(RESULT)
        if saved["challenge_sha256"] != sha256(experiment.FROZEN):
            raise ValueError("CENTER intervention source mismatch")
        return saved
    original = protocol.render
    groups = {}
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], "representation_attribution_v1:center_flip") as budget:
            tok = AutoTokenizer.from_pretrained(local_train.RUN / "latest", local_files_only=True)
            network = StepVerifier.from_pretrained(local_train.RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(network, tok, cfg, budget)
            for group in ("hard_collision", "valid_short"):
                cases = []
                for index, row in enumerate(data["panels"]["test"][group]):
                    step = (int(row["target_text"].split("step=")[1].split()[0])
                            if group == "hard_collision" else len(row["candidate_plan"]))
                    old_value, injected = (("blocked", "free") if group == "hard_collision"
                                           else ("free", "blocked"))

                    def render_flip(grid, candidate):
                        lines = representation.render(grid, candidate).splitlines()
                        line_index = step + 1
                        needle = f"CENTER {old_value}"
                        if lines[line_index].count(needle) != 1:
                            raise ValueError("Selected CENTER token missing")
                        lines[line_index] = lines[line_index].replace(
                            needle, f"CENTER {injected}", 1)
                        return "\n".join(lines)

                    protocol.render = render_flip
                    prediction, detail = verifier(grid_from_record(row), row["candidate_plan"])
                    unchanged = baseline["models"]["full"][group]["cases"][index]["prediction"]["text"]
                    cases.append({"sample_id": row["sample_id"], "step": step,
                                  "original": unchanged, "flipped": prediction.text,
                                  "failure": prediction.failure,
                                  "changed": prediction.text != unchanged,
                                  "injected_collision_at_step": prediction.text ==
                                  f"invalid | step={step} | reason=obstacle_collision",
                                  "detail": detail})
                groups[group] = cases
            report = {"challenge_sha256": sha256(experiment.FROZEN),
                      "checkpoint_fingerprint": identity["fingerprint"],
                      "gpu_calls": verifier.calls, "groups": groups}
            write_json(RESULT, report)
            return report
    finally:
        protocol.render = original


if __name__ == "__main__":
    result = run()
    for name, rows in result["groups"].items():
        print(name, "changed", sum(x["changed"] for x in rows), "/", len(rows),
              "reports_injected_collision",
              sum(x["injected_collision_at_step"] for x in rows))
