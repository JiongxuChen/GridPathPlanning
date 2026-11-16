"""Focused hard-prefix loss ablation on the same frozen V1000 replacement."""
from pathlib import Path
import importlib.metadata
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import experiment
import representation
from gridpath.experiments import training
from gridpath.planning import protocol
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256
from gridpath.training_helpers import validate_checkpoint

NAME = "late_collision_loss_v1"
RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"


def focused_loss(logits, labels):
    import torch
    import torch.nn.functional as F

    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                             ignore_index=-100, reduction="none").reshape(labels.shape)
    values = []
    for index, row in enumerate(labels):
        supervised = torch.nonzero(row != -100).flatten()
        last, prefix = supervised[-1], supervised[:-1]
        weight = torch.where(row[last] == 4, logits.new_tensor(0.4),
                             logits.new_tensor(1.6))
        decision = weight * losses[index, last]
        if len(prefix):
            prior = losses[index, prefix]
            hard = 1 + 2 * (1 - (-prior.detach()).exp()).square()
            values.append(0.5 * ((hard * prior).sum() / hard.sum() + decision))
        else:
            values.append(decision)
    return torch.stack(values).mean()


def focused_update(model, optimizer, scheduler, features, tokenizer, cfg):
    import torch
    from gridpath.inference import autocast_context
    from gridpath.planning.model import collate

    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs, labels = collate(features, tokenizer.pad_token_id,
                             next(model.parameters()).device)
    with autocast_context(next(model.parameters()).device, cfg["precision"]):
        loss = focused_loss(model(**inputs), labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite focused loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"],
                                          error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm),
            "learning_rate": scheduler.get_last_lr()[0]}


def binding():
    cfg, rows, dev, baseline = experiment.binding()
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {
        "experiment": NAME, "baseline_binding": baseline["fingerprint"],
        "config": cfg, "prepared_sha256": sha256(experiment.FROZEN),
        "representation_sha256": sha256(Path(representation.__file__)),
        "loss_source_sha256": sha256(Path(__file__)),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("torch", "transformers", "numpy")},
    }
    return cfg, rows, dev, {"fingerprint": digest(payload), **payload}


def train():
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
            raise ValueError("Focused checkpoint identity changed")
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    old = protocol.render, training.binding_for, training.budget, training.structured_update
    protocol.render = representation.render
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        experiment.local_train.LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    training.structured_update = focused_update
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        (protocol.render, training.binding_for, training.budget,
         training.structured_update) = old


def score_dev():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier
    import evaluate_dev
    from gridpath.training_config import write_json

    data = experiment.prepare()
    cfg, _, _, identity = binding()
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    target = experiment.OUT / "loss_dev_predictions.json"
    if target.exists():
        saved = read_json(target)
        if saved["training_fingerprint"] != identity["fingerprint"]:
            raise ValueError("Focused evaluation identity changed")
        return saved
    original = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(experiment.local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], f"{NAME}:t5_base:dev") as budget:
            tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
            model = StepVerifier.from_pretrained(RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(model, tok, cfg, budget)
            groups = {name: evaluate_dev.evaluate_group(
                [(pair[name], pair[name]["candidate_plan"]) for pair in data["panels"]["dev"]],
                verifier) for name in ("short_valid", "long_valid", "early_collision", "late_collision")}
            fixed, natural = evaluate_dev.dev_items()
            groups["fixed_V80"] = evaluate_dev.evaluate_group(fixed, verifier)
            groups["natural_G60"] = evaluate_dev.evaluate_group(natural, verifier)
            result = {"training_fingerprint": identity["fingerprint"],
                      "prepared_sha256": sha256(experiment.FROZEN), "groups": groups,
                      "gpu_calls": verifier.calls}
            write_json(target, result)
            return result
    finally:
        protocol.render = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "dev"))
    result = train() if parser.parse_args().stage == "train" else score_dev()
    if "groups" not in result:
        print(result)
    else:
        for name, group in result["groups"].items():
            count = sum(x["prediction"]["text"] == x["row"]["target_text"]
                        for x in group["cases"])
            print(name, count, "/", len(group["cases"]),
                  "interface_failures", group["metrics"]["interface_failure_count"])
