"""New frozen layouts for seven same-candidate structured verifier models."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/structured_pairs_v1"),
                str(HERE)]

import experiment as paired
import holdout as original_holdout
from gridpath.experiments.config import LEDGER, completed_model
from gridpath.experiments.data import MANIFEST
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
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
multi = load("relational_multiseed", HERE / "multiseed.py")
paired_multi = load("paired_multiseed", ROOT / "analysis/structured_pairs_v1/multiseed.py")

NAME = "relational_set_multiseed_v1"
OUT = ROOT / "results" / NAME
EXCLUSIONS = OUT / "exclusions.json"
FROZEN = OUT / "challenge.json"
GENERATED = OUT / "G_outputs.json"
PREDICTIONS = OUT / "predictions.json"
SEED = 20261129


def exclusions():
    previous = ROOT / "results/relational_set_v1/exclusions.json"
    prior_challenge = ROOT / "results/relational_set_v1/challenge.json"
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
               for p in (previous, prior_challenge)}
    ids = set(read_json(previous)["layout_ids"])
    old = read_json(prior_challenge)
    added = {p["valid"]["layout_id"] for p in old["groups"]["paired"]}
    added.update(r["layout_id"] for r in old["groups"]["natural_G"])
    assert len(added) == 80 and not ids.intersection(added)
    ids.update(added)
    assert len(ids) == 2400
    manifest = {"sources": sources, "layout_ids": sorted(ids)}
    if EXCLUSIONS.exists():
        assert read_json(EXCLUSIONS) == manifest
    else:
        write_json(EXCLUSIONS, manifest)
    return manifest


def configured():
    exclusions()
    h = original_holdout
    h.OUT = OUT
    h.EXCLUSIONS = EXCLUSIONS
    h.FROZEN = FROZEN
    h.GENERATED = GENERATED
    h.SEED = SEED
    h.sources = lambda: {
        "entry": sha256(Path(__file__)),
        "protocol": sha256(HERE / "multiseed_protocol.md"),
        "exclusions": sha256(EXCLUSIONS),
        "dataset": sha256(MANIFEST),
        "paired_overlay": sha256(paired.OVERLAY),
        "set_model": sha256(HERE / "set_model.py"),
        "relation_entry": sha256(HERE / "experiment.py"),
        "multiseed_entry": sha256(HERE / "multiseed.py"),
        "holdout_generator": sha256(
            ROOT / "analysis/structured_pairs_v1/holdout.py")}
    return h


def freeze():
    h = configured()
    return h.audit(h.prepare()), sha256(FROZEN)


def generate():
    h = configured()
    h.prepare()
    previous = paired.NAME
    paired.NAME = NAME
    try:
        result = h.generate()
    finally:
        paired.NAME = previous
    return result["model_calls"], sha256(GENERATED)


def score():
    h = configured()
    challenge = h.prepare()
    generate()
    generated = read_json(GENERATED)
    if PREDICTIONS.exists():
        saved = read_json(PREDICTIONS)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generated_sha256"] == sha256(GENERATED)
        return saved
    controlled = [(row, row["candidate_plan"])
                  for pair in challenge["groups"]["paired"]
                  for row in (pair["valid"], pair["collision"])]
    natural, failures = [], []
    for row, output in zip(challenge["groups"]["natural_G"],
                           generated["outputs"], strict=True):
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            natural.append((row, output["prediction"]["text"]))
        else:
            failures.append(row["sample_id"])
    original_cfg, original_path, original_id = completed_model(
        "improvement_verify_v1", "t5_base", "verify")
    paths = [("original", original_cfg, original_path, original_id,
              StepVerifier, Verifier)]
    for seed in (42, 43, 44):
        if seed == 42:
            pcfg, _, _, pid = paired.binding()
            prun = paired.RUN
            rcfg, _, _, rid = relational.binding()
            rrun = relational.RUN
        else:
            pcfg, _, _, pid = paired_multi.binding(seed, "paired")
            prun = paired_multi.run_path(seed, "paired")
            previous_name, previous_run = multi.relational.NAME, multi.relational.RUN
            multi.relational.NAME, multi.relational.RUN = multi.NAME, multi.run_path(seed)
            try:
                rcfg, _, _, rid = multi.binding(seed)
            finally:
                multi.relational.NAME, multi.relational.RUN = previous_name, previous_run
            rrun = multi.run_path(seed)
        for run, identity in ((prun, pid), (rrun, rid)):
            assert read_json(run / "completed.json")["fingerprint"] == identity["fingerprint"]
            validate_checkpoint(run / "latest", identity["fingerprint"])
        paths.append((f"paired_seed{seed}", pcfg, prun / "latest", pid,
                      StepVerifier, Verifier))
        paths.append((f"learned_set_seed{seed}", rcfg, rrun / "latest", rid,
                      relational.set_model.SetStepVerifier, relational.SetVerifier))
    models = {}
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:fresh_V") as budget:
        for name, cfg, path, identity, model_type, verifier_type in paths:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            net = model_type.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = verifier_type(net, tok, cfg, budget)
            models[name] = {"fingerprint": identity["fingerprint"],
                            "weights_sha256": sha256(path / "model.safetensors"),
                            "controlled": paired.score(controlled, verifier),
                            "natural_G": paired.score(natural, verifier),
                            "model_calls": verifier.calls}
            assert verifier.calls == len(controlled) + len(natural)
            assert verifier.hits == 0
            del verifier, net
    result = {"status": "fresh_targeted_layout_evaluation",
              "challenge_sha256": sha256(FROZEN),
              "generated_sha256": sha256(GENERATED),
              "sources": challenge["sources"],
              "audit": h.audit(challenge),
              "G_interface_failures": failures,
              "models": models}
    write_json(PREDICTIONS, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate", "score"))
    args = parser.parse_args()
    if args.stage == "freeze":
        print(freeze())
    elif args.stage == "generate":
        print(generate())
    else:
        print(score()["status"])
