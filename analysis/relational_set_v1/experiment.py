"""Isolated T5 plus learned raw-offset set verifier; historical code untouched."""
from contextlib import contextmanager
from pathlib import Path
import importlib.metadata
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/structured_pairs_v1"),
                str(HERE)]

import experiment as paired
import set_model
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, binding_for
from gridpath.planning import model as planning_model, protocol, verification
from gridpath.planning.verification import Verifier as BaseVerifier
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

NAME = "relational_set_v1"
OUT = ROOT / "results" / NAME
RUN = ROOT / "checkpoints" / NAME / "t5_base/verify"
DEV = OUT / "verification_dev.json"


class SetVerifier(BaseVerifier):
    def __call__(self, grid, candidate):
        original = verification.feature, verification.collate
        verification.feature, verification.collate = set_model.feature, set_model.collate
        try:
            return super().__call__(grid, candidate)
        finally:
            verification.feature, verification.collate = original


@contextmanager
def patched_training(cfg):
    original = (training.binding_for, training.budget,
                protocol.feature, planning_model.collate,
                planning_model.StepVerifier, verification.Verifier)
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    protocol.feature = set_model.feature
    planning_model.collate = set_model.collate
    planning_model.StepVerifier = set_model.SetStepVerifier
    verification.Verifier = SetVerifier
    try:
        yield
    finally:
        (training.binding_for, training.budget, protocol.feature,
         planning_model.collate, planning_model.StepVerifier,
         verification.Verifier) = original


def binding():
    cfg, _, dev, base = binding_for("improvement_verify_v1", "t5_base",
                                    "verify", True)
    rows = paired.prepare()["rows"]
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {
        "experiment": NAME, "base_binding": base["fingerprint"],
        "config": cfg, "paired_overlay_sha256": sha256(paired.OVERLAY),
        "entry_sha256": sha256(Path(__file__)),
        "set_model_sha256": sha256(HERE / "set_model.py"),
        "protocol_sha256": sha256(HERE / "protocol.md"),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("torch", "transformers", "numpy")}}
    return cfg, rows, dev, {"fingerprint": digest(payload), **payload}


def train():
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        assert done["fingerprint"] == identity["fingerprint"]
        assert done["state"]["global_step"] == 750
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    with patched_training(cfg):
        return training.run("improvement_verify_v1", "t5_base", "verify")


def evaluate():
    cfg, _, _, identity = binding()
    assert read_json(RUN / "completed.json")["fingerprint"] == identity["fingerprint"]
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    if DEV.exists():
        saved = read_json(DEV)
        assert saved["fingerprint"] == identity["fingerprint"]
        assert saved["paired_overlay_sha256"] == sha256(paired.OVERLAY)
        return saved
    fixed, natural = paired.development_items()
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:dev") as budget:
        tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
        net = set_model.SetStepVerifier.from_pretrained(
            RUN / "latest", local_files_only=True).to("cuda").eval()
        verifier = SetVerifier(net, tok, cfg, budget)
        panels = {"fixed_V80": paired.score(fixed, verifier),
                  "natural_G60": paired.score(natural, verifier)}
        assert verifier.calls == 140 and verifier.hits == 0
        result = {"fingerprint": identity["fingerprint"],
                  "paired_overlay_sha256": sha256(paired.OVERLAY),
                  "weights_sha256": sha256(RUN / "latest/model.safetensors"),
                  "panels": panels, "model_calls": verifier.calls}
        write_json(DEV, result)
        return result


def guard(report):
    fixed = report["panels"]["fixed_V80"]["cases"]
    natural = report["panels"]["natural_G60"]["cases"]
    collisions = [x for x in natural if x["row"]["error_reason"] == "obstacle_collision"]
    valid = [x for x in natural if x["row"]["target_text"] == "valid"]
    other = [x for x in fixed + natural if x["row"]["error_reason"] in (
        "out_of_bounds", "invalid_action_token", "goal_not_reached")]
    return {
        "natural_collision_joint": [sum(x["prediction"]["text"] == x["row"]["target_text"]
                                        for x in collisions), len(collisions)],
        "natural_valid": [sum(x["prediction"]["text"] == "valid" for x in valid),
                          len(valid)],
        "other_errors": [sum(x["prediction"]["text"] == x["row"]["target_text"]
                             for x in other), len(other)],
        "interface_failures": sum(x["prediction"]["failure"] is not None
                                  for x in fixed + natural)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "dev"))
    args = parser.parse_args()
    if args.stage == "train":
        print(train())
    else:
        print(guard(evaluate()))
