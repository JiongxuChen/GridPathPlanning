"""Frozen T5 encoder, train only the step head on center-only inputs."""
from pathlib import Path
import importlib.metadata
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import evaluate_dev
import experiment
import train as local_train
import variants
from gridpath.experiments import training
from gridpath.planning import protocol
from gridpath.planning.model import StepVerifier
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint

NAME = "representation_attribution_head_v1"
RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"


def binding():
    cfg, rows, dev, baseline = local_train.binding()
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {
        "experiment": NAME, "backbone": "frozen", "representation": "center_only",
        "baseline_binding": baseline["fingerprint"], "config": cfg,
        "source_sha256": sha256(Path(__file__)),
        "variants_sha256": sha256(HERE / "variants.py"),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("torch", "transformers", "numpy")},
    }
    return cfg, rows, dev, {"fingerprint": digest(payload), **payload}


def head_update(model, optimizer, scheduler, features, tokenizer, cfg):
    import torch
    from gridpath.inference import autocast_context
    from gridpath.planning.model import collate, path_loss

    model.train()
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    inputs, labels = collate(features, tokenizer.pad_token_id,
                             next(model.parameters()).device)
    with autocast_context(next(model.parameters()).device, cfg["precision"]):
        loss = path_loss(model(**inputs), labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite frozen-feature loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"],
                                          error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm),
            "learning_rate": scheduler.get_last_lr()[0]}


def train():
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
            raise ValueError("Head-only checkpoint mismatch")
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    old = protocol.render, training.binding_for, training.budget, training.structured_update
    original_factory = StepVerifier.__dict__["from_backbone"]
    bound_factory = StepVerifier.from_backbone

    def frozen_factory(cls, path):
        network = bound_factory(path)
        network.encoder.requires_grad_(False)
        return network

    protocol.render = lambda grid, candidate: variants.render(grid, candidate, "center_only")
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        local_train.LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{stage}")
    training.structured_update = head_update
    StepVerifier.from_backbone = classmethod(frozen_factory)
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        (protocol.render, training.binding_for, training.budget,
         training.structured_update) = old
        StepVerifier.from_backbone = original_factory


def score(split):
    from transformers import AutoTokenizer
    from gridpath.planning.verification import Verifier

    data = experiment.prepare()
    cfg, _, _, identity = binding()
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    output = experiment.OUT / f"head_{split}_predictions.json"
    if output.exists():
        result = read_json(output)
        if (result["challenge_sha256"] != sha256(experiment.FROZEN) or
                result["fingerprint"] != identity["fingerprint"]):
            raise ValueError("Head-only result provenance mismatch")
        return result
    original = protocol.render
    protocol.render = lambda grid, candidate: variants.render(grid, candidate, "center_only")
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], f"{NAME}:{split}") as budget:
            tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
            model = StepVerifier.from_pretrained(RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(model, tok, cfg, budget)
            groups = {group: evaluate_dev.evaluate_group(
                [(r, r["candidate_plan"]) for r in data["panels"][split][group]], verifier)
                for group in experiment.GROUPS}
            report = {"challenge_sha256": sha256(experiment.FROZEN),
                      "fingerprint": identity["fingerprint"], "split": split,
                      "gpu_calls": verifier.calls, "groups": groups}
            write_json(output, report)
            return report
    finally:
        protocol.render = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "dev", "test"))
    stage = parser.parse_args().stage
    result = train() if stage == "train" else score(stage)
    if "groups" not in result:
        print(result)
    else:
        print(stage, {g: sum(x["prediction"]["text"] == x["row"]["target_text"]
                             for x in result["groups"][g]["cases"])
                      for g in experiment.GROUPS})
