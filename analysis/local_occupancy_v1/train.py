"""Isolated, equal-budget T5-base V training with a 3x3 occupancy representation."""
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import representation
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, binding_for
from gridpath.planning import protocol
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256
from gridpath.training_helpers import validate_checkpoint

NAME = "local_occupancy_v1"
RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"
OUT = ROOT / "results" / NAME


def binding():
    cfg, train, dev, baseline = binding_for("improvement_verify_v1", "t5_base", "verify", True)
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {
        "experiment": NAME, "baseline_binding": baseline["fingerprint"],
        "config": cfg, "train_sample_ids": digest([r["sample_id"] for r in train]),
        "representation_sha256": sha256(Path(representation.__file__)),
        "training_entry_sha256": sha256(Path(__file__)),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
    }
    return cfg, train, dev, {"fingerprint": digest(payload), **payload}


def main():
    cfg, train, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
            raise ValueError("Completed checkpoint belongs to another experiment")
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    original_render, original_binding, original_budget = (
        protocol.render, training.binding_for, training.budget)
    protocol.render = representation.render
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        protocol.render, training.binding_for, training.budget = (
            original_render, original_binding, original_budget)


if __name__ == "__main__":
    print(main())
