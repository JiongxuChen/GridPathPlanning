"""Train the same relational-set V at seeds 43/44 without changing seed 42."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/structured_pairs_v1"),
                str(HERE)]

import experiment as paired
from gridpath.experiments import training
from gridpath.training_config import digest, sha256

spec = importlib.util.spec_from_file_location("relational_experiment", HERE / "experiment.py")
relational = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relational)

NAME = "relational_set_multiseed_v1"
OUT = ROOT / "results" / NAME
SEEDS = (43, 44)


def run_path(seed):
    return ROOT / "checkpoints" / NAME / f"seed_{seed}" / "t5_base/verify"


def binding(seed):
    assert seed in SEEDS
    cfg, rows, dev, base = relational_base_binding()
    cfg = {**cfg, "seed": seed}
    payload = {key: val for key, val in base.items() if key != "fingerprint"}
    payload.update({
        "actual_seed": seed, "config": cfg,
        "multiseed_entry_sha256": sha256(Path(__file__)),
        "multiseed_protocol_sha256": sha256(HERE / "multiseed_protocol.md"),
        "training_source_sha256": sha256(
            ROOT / "src/gridpath/experiments/training.py"),
        "helpers_source_sha256": sha256(
            ROOT / "src/gridpath/training_helpers.py")})
    return cfg, rows, dev, {"fingerprint": digest(payload), **payload}


relational_base_binding = relational.binding


def train(seed):
    assert seed in SEEDS
    old_name, old_run, old_binding = (
        relational.NAME, relational.RUN, relational.binding)
    old_seed, old_stream = training.seed_everything, training.stream_indices
    relational.NAME = NAME
    relational.RUN = run_path(seed)
    relational.binding = lambda: binding(seed)
    training.seed_everything = lambda ignored: old_seed(seed)
    training.stream_indices = lambda count, ignored, consumed, size: old_stream(
        count, seed, consumed, size)
    try:
        return relational.train()
    finally:
        relational.NAME, relational.RUN, relational.binding = (
            old_name, old_run, old_binding)
        training.seed_everything, training.stream_indices = old_seed, old_stream


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    print(train(args.seed))
