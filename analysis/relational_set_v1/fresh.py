"""Freeze unseen layouts and compare old and learned-offset V on identical paths."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/structured_pairs_v1"),
                str(HERE)]

import experiment as paired
import holdout as prior_holdout
import importlib.util
from gridpath.experiments.config import LEDGER, completed_model
from gridpath.experiments.data import MANIFEST
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

spec = importlib.util.spec_from_file_location("relational_experiment", HERE / "experiment.py")
relational = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relational)

NAME = "relational_set_v1"
OUT = ROOT / "results" / NAME
EXCLUSIONS = OUT / "exclusions.json"
FROZEN = OUT / "challenge.json"
GENERATED = OUT / "G_outputs.json"
PREDICTIONS = OUT / "predictions.json"
SEED = 20261128


def exclusions():
    previous = ROOT / "results/structured_pairs_multiseed_v1/exclusions.json"
    old_challenge = ROOT / "results/structured_pairs_multiseed_v1/challenge.json"
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
               for p in (previous, old_challenge)}
    ids = set(read_json(previous)["layout_ids"])
    old = read_json(old_challenge)
    new_ids = {p["valid"]["layout_id"] for p in old["groups"]["paired"]}
    new_ids.update(r["layout_id"] for r in old["groups"]["natural_G"])
    assert len(new_ids) == 80 and not ids.intersection(new_ids)
    ids.update(new_ids)
    assert len(ids) == 2320
    value = {"sources": sources, "layout_ids": sorted(ids)}
    if EXCLUSIONS.exists():
        assert read_json(EXCLUSIONS) == value
    else:
        write_json(EXCLUSIONS, value)
    return value


def configured():
    exclusions()
    h = prior_holdout
    h.OUT = OUT
    h.EXCLUSIONS = EXCLUSIONS
    h.FROZEN = FROZEN
    h.GENERATED = GENERATED
    h.SEED = SEED
    h.sources = lambda: {
        "entry": sha256(Path(__file__)),
        "protocol": sha256(HERE / "protocol.md"),
        "exclusions": sha256(EXCLUSIONS),
        "dataset": sha256(MANIFEST),
        "overlay": sha256(paired.OVERLAY),
        "relation_entry": sha256(HERE / "experiment.py"),
        "set_model": sha256(HERE / "set_model.py"),
        "holdout_generator": sha256(ROOT / "analysis/structured_pairs_v1/holdout.py")}
    return h


def freeze():
    h = configured()
    challenge = h.prepare()
    return h.audit(challenge), sha256(FROZEN)


def generate():
    h = configured()
    h.prepare()
    prior_name = paired.NAME
    paired.NAME = NAME
    try:
        result = h.generate()
    finally:
        paired.NAME = prior_name
    return result["model_calls"], sha256(GENERATED)


def score():
    h = configured()
    challenge = h.prepare()
    generate()
    generated = read_json(GENERATED)
    assert generated["challenge_sha256"] == sha256(FROZEN)
    if PREDICTIONS.exists():
        saved = read_json(PREDICTIONS)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generated_sha256"] == sha256(GENERATED)
        return saved
    controlled = [(r, r["candidate_plan"]) for pair in challenge["groups"]["paired"]
                  for r in (pair["valid"], pair["collision"])]
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
    paired_cfg, _, _, paired_id = paired.binding()
    relation_cfg, _, _, relation_id = relational.binding()
    assert read_json(paired.RUN / "completed.json")["fingerprint"] == paired_id["fingerprint"]
    assert read_json(relational.RUN / "completed.json")["fingerprint"] == relation_id["fingerprint"]
    validate_checkpoint(paired.RUN / "latest", paired_id["fingerprint"])
    validate_checkpoint(relational.RUN / "latest", relation_id["fingerprint"])
    models = {}
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:fresh_V") as budget:
        for name, cfg, path, identity, model_type, verifier_type in (
            ("original", original_cfg, original_path, original_id, StepVerifier, Verifier),
            ("paired_seed42", paired_cfg, paired.RUN / "latest", paired_id,
             StepVerifier, Verifier),
            ("learned_set_seed42", relation_cfg, relational.RUN / "latest",
             relation_id, relational.set_model.SetStepVerifier, relational.SetVerifier)):
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
