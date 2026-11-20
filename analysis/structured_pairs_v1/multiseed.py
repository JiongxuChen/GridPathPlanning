"""Isolated matched-seed training and frozen-layout verifier comparison."""
from pathlib import Path
import importlib.metadata
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

import experiment as paired
import position_ablation as unpaired
import holdout as prior_holdout
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, binding_for, completed_model
from gridpath.experiments.data import MANIFEST
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

NAME = "structured_pairs_multiseed_v1"
OUT = ROOT / "results" / NAME
SEEDS = (43, 44)
ARMS = {"paired": paired.OVERLAY, "position_unpaired": unpaired.OVERLAY}
EXCLUSIONS = OUT / "exclusions.json"
FROZEN = OUT / "challenge.json"
GENERATED = OUT / "G_outputs.json"
RESULT = OUT / "predictions.json"
PANEL_SEED = 20261127


def run_path(seed, arm):
    return ROOT / "checkpoints" / NAME / f"seed_{seed}" / arm / "t5_base/verify"


def binding(seed, arm):
    assert seed in SEEDS and arm in ARMS
    cfg, _, dev, base = binding_for("improvement_verify_v1", "t5_base",
                                    "verify", True)
    overlay = paired.prepare() if arm == "paired" else unpaired.prepare()
    cfg = {**cfg, "seed": seed,
           "run_directory": str(run_path(seed, arm).relative_to(ROOT))}
    payload = {
        "experiment": NAME, "arm": arm, "actual_seed": seed,
        "base_binding": base["fingerprint"], "config": cfg,
        "overlay_sha256": sha256(ARMS[arm]),
        "entry_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(HERE / "multiseed_protocol.md"),
        "training_source_sha256": sha256(ROOT / "src/gridpath/experiments/training.py"),
        "helpers_source_sha256": sha256(ROOT / "src/gridpath/training_helpers.py"),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("torch", "transformers", "numpy")}}
    return cfg, overlay["rows"], dev, {"fingerprint": digest(payload), **payload}


def train(seed, arm):
    cfg, _, _, identity = binding(seed, arm)
    run = run_path(seed, arm)
    if (run / "completed.json").exists():
        done = read_json(run / "completed.json")
        assert done["fingerprint"] == identity["fingerprint"]
        assert done["state"]["global_step"] == 750
        validate_checkpoint(run / "latest", identity["fingerprint"])
        return done
    old_binding, old_budget = training.binding_for, training.budget
    old_seed, old_stream = training.seed_everything, training.stream_indices
    training.binding_for = lambda method, model, role, verify_weights=False: binding(seed, arm)
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:seed_{seed}:{arm}:{stage}")
    training.seed_everything = lambda ignored: old_seed(seed)
    training.stream_indices = lambda count, ignored, consumed, size: old_stream(
        count, seed, consumed, size)
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        training.binding_for, training.budget = old_binding, old_budget
        training.seed_everything, training.stream_indices = old_seed, old_stream


def exclusion_manifest():
    previous = paired.OUT / "exclusions.json"
    old_challenge = paired.OUT / "holdout_challenge.json"
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
               for path in (previous, old_challenge)}
    prior = read_json(previous)
    challenge = read_json(old_challenge)
    old_ids = {pair["valid"]["layout_id"] for pair in challenge["groups"]["paired"]}
    old_ids.update(row["layout_id"] for row in challenge["groups"]["natural_G"])
    assert len(old_ids) == 80
    ids = sorted(set(prior["layout_ids"]) | old_ids)
    assert len(ids) == len(prior["layout_ids"]) + 80
    value = {"sources": sources, "layout_ids": ids}
    if EXCLUSIONS.exists():
        assert read_json(EXCLUSIONS) == value
    else:
        write_json(EXCLUSIONS, value)
    return value


def configure_holdout():
    exclusion_manifest()
    h = prior_holdout
    h.OUT = OUT
    h.EXCLUSIONS = EXCLUSIONS
    h.FROZEN = FROZEN
    h.GENERATED = GENERATED
    h.SEED = PANEL_SEED
    h.sources = lambda: {
        "entry": sha256(Path(__file__)),
        "protocol": sha256(HERE / "multiseed_protocol.md"),
        "exclusions": sha256(EXCLUSIONS), "dataset": sha256(MANIFEST),
        "paired_overlay": sha256(paired.OVERLAY),
        "unpaired_overlay": sha256(unpaired.OVERLAY),
        "holdout_generator": sha256(HERE / "holdout.py")}
    return h


def freeze():
    h = configure_holdout()
    challenge = h.prepare()
    return h.audit(challenge), sha256(FROZEN)


def generate():
    h = configure_holdout()
    h.prepare()
    old_name = h.exp.NAME
    h.exp.NAME = NAME
    try:
        result = h.generate()
    finally:
        h.exp.NAME = old_name
    return result["model_calls"], sha256(GENERATED)


def score():
    h = configure_holdout()
    challenge = h.prepare()
    generate()
    generated = read_json(GENERATED)
    assert generated["challenge_sha256"] == sha256(FROZEN)
    if RESULT.exists():
        saved = read_json(RESULT)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generated_sha256"] == sha256(GENERATED)
        return saved
    controlled = [(r, r["candidate_plan"]) for pair in challenge["groups"]["paired"]
                  for r in (pair["valid"], pair["collision"])]
    natural = []
    failures = []
    for row, output in zip(challenge["groups"]["natural_G"],
                           generated["outputs"], strict=True):
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            natural.append((row, output["prediction"]["text"]))
        else:
            failures.append(output["sample_id"])
    paths = []
    for seed in (42, *SEEDS):
        for arm in ARMS:
            if seed == 42:
                module = paired if arm == "paired" else unpaired
                cfg, _, _, identity = module.binding()
                run = module.RUN
            else:
                cfg, _, _, identity = binding(seed, arm)
                run = run_path(seed, arm)
            assert read_json(run / "completed.json")["fingerprint"] == identity["fingerprint"]
            validate_checkpoint(run / "latest", identity["fingerprint"])
            paths.append((f"seed_{seed}_{arm}", cfg, run / "latest", identity))
    original_cfg, original_path, original_identity = completed_model(
        "improvement_verify_v1", "t5_base", "verify")
    paths.append(("original_structured", original_cfg, original_path, original_identity))
    models = {}
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:fresh_V") as budget:
        for name, cfg, path, identity in paths:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            net = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = Verifier(net, tok, cfg, budget)
            models[name] = {
                "fingerprint": identity["fingerprint"],
                "weights_sha256": sha256(path / "model.safetensors"),
                "controlled": paired.score(controlled, verifier),
                "natural_G": paired.score(natural, verifier),
                "model_calls": verifier.calls}
            assert verifier.calls == len(controlled) + len(natural)
            assert verifier.hits == 0
            del verifier, net
    result = {"status": "fresh_targeted_layout_evaluation",
              "protocol_sha256": sha256(HERE / "multiseed_protocol.md"),
              "challenge_sha256": sha256(FROZEN),
              "generated_sha256": sha256(GENERATED),
              "G_interface_failures": failures,
              "models": models}
    write_json(RESULT, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate", "train", "score"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.stage == "freeze":
        print(freeze())
    elif args.stage == "generate":
        print(generate())
    elif args.stage == "train":
        assert args.seed in SEEDS and args.arm in ARMS
        print(train(args.seed, args.arm))
    else:
        print(score()["status"])
