"""Positive-content ablation with paired training-row positions held fixed."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

import experiment as paired
import unpaired_ablation as unpaired
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, binding_for
from gridpath.experiments.data import load
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

NAME = "structured_pairs_position_ablation_v1"
OUT = ROOT / "results" / NAME
OVERLAY = OUT / "overlay.json"
RUN = ROOT / "checkpoints" / NAME / "t5_base/verify"
DEV = OUT / "verification_dev.json"
POSTHOC = OUT / "posthoc_holdout.json"


def sources():
    return {"entry": sha256(Path(__file__)),
            "protocol": sha256(HERE / "position_ablation_protocol.md"),
            "paired_overlay": sha256(paired.OVERLAY),
            "unpaired_overlay": sha256(unpaired.OVERLAY)}


def prepare():
    parent = paired.prepare()
    old = unpaired.prepare()
    identity = sources()
    if OVERLAY.exists():
        saved = read_json(OVERLAY)
        assert saved["sources"] == identity
        audit(saved)
        return saved
    replacements = sorted(
        (r for r in old["rows"] if r["sample_id"] in old["restored_valid_ids"]),
        key=lambda r: r["sample_id"])
    positions = [i for i, row in enumerate(parent["rows"])
                 if row.get("origin") == "same_environment_shortest_counterfactual"]
    assert len(positions) == len(replacements) == 40
    rows = list(parent["rows"])
    for position, row in zip(positions, replacements, strict=True):
        rows[position] = row
    result = {"sources": identity, "rows": rows, "replacement_positions": positions,
              "restored_valid_ids": [r["sample_id"] for r in replacements]}
    audit(result)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OVERLAY, result)
    return result


def audit(data):
    assert data["sources"] == sources()
    parent = paired.prepare()["rows"]
    old = unpaired.prepare()
    rows, positions = data["rows"], data["replacement_positions"]
    assert len(rows) == len(parent) == 1000
    assert positions == [i for i, row in enumerate(parent)
                         if row.get("origin") == "same_environment_shortest_counterfactual"]
    assert len(positions) == 40 and len(set(positions)) == 40
    restored = {r["sample_id"]: r for r in old["rows"]
                if r["sample_id"] in old["restored_valid_ids"]}
    assert len(restored) == 40
    assert data["restored_valid_ids"] == sorted(restored)
    for i, (current, original) in enumerate(zip(rows, parent, strict=True)):
        if i in positions:
            assert current == restored[data["restored_valid_ids"][positions.index(i)]]
            assert current["is_valid"] and original["is_valid"]
        else:
            assert current == original
    assert len({r["sample_id"] for r in rows}) == 1000
    assert {r["sample_id"] for r in rows} == {r["sample_id"] for r in old["rows"]}
    assert Counter(r["is_valid"] for r in rows) == {True: 500, False: 500}
    assert Counter(r["error_reason"] for r in rows if not r["is_valid"]) == {
        x: 125 for x in ("obstacle_collision", "out_of_bounds",
                         "invalid_action_token", "goal_not_reached")}
    return {"rows": 1000, "unchanged_positions": 960, "positive_replacements": 40}


def binding():
    cfg, _, dev, base = binding_for("improvement_verify_v1", "t5_base",
                                    "verify", True)
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {"experiment": NAME, "base_binding": base["fingerprint"],
               "config": cfg, "overlay_sha256": sha256(OVERLAY),
               "entry_sha256": sha256(Path(__file__)),
               "packages": {p: importlib.metadata.version(p)
                            for p in ("torch", "transformers", "numpy")}}
    return cfg, prepare()["rows"], dev, {"fingerprint": digest(payload), **payload}


def train():
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        assert done["fingerprint"] == identity["fingerprint"]
        assert done["state"]["global_step"] == 750
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    old_binding, old_budget = training.binding_for, training.budget
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        training.binding_for, training.budget = old_binding, old_budget


def score_panels(panels, path, status, source=None):
    cfg, _, _, identity = binding()
    assert read_json(RUN / "completed.json")["fingerprint"] == identity["fingerprint"]
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    if path.exists():
        saved = read_json(path)
        assert saved["fingerprint"] == identity["fingerprint"]
        assert saved.get("source_sha256") == source
        return saved
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:{status}") as budget:
        tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
        net = StepVerifier.from_pretrained(RUN / "latest",
                                            local_files_only=True).to("cuda").eval()
        verifier = Verifier(net, tok, cfg, budget)
        scored = {name: paired.score(items, verifier) for name, items in panels.items()}
        assert verifier.calls == sum(len(items) for items in panels.values())
        result = {"status": status, "fingerprint": identity["fingerprint"],
                  "overlay_sha256": sha256(OVERLAY),
                  "model_sha256": sha256(RUN / "latest/model.safetensors"),
                  "source_sha256": source, "panels": scored,
                  "model_calls": verifier.calls}
        write_json(path, result)
        return result


def dev():
    fixed, natural = paired.development_items()
    return score_panels({"fixed_V80": fixed, "natural_G60": natural},
                        DEV, "public_development")


def posthoc():
    challenge_path = paired.OUT / "holdout_challenge.json"
    generated_path = paired.OUT / "holdout_G_outputs.json"
    source_path = paired.OUT / "holdout_predictions.json"
    challenge, generated, source = (
        read_json(p) for p in (challenge_path, generated_path, source_path))
    assert source["challenge_sha256"] == generated["challenge_sha256"] == sha256(challenge_path)
    assert source["generated_sha256"] == sha256(generated_path)
    controlled = [(row, row["candidate_plan"])
                  for pair in challenge["groups"]["paired"]
                  for row in (pair["valid"], pair["collision"])]
    natural = [(row, output["prediction"]["text"])
               for row, output in zip(challenge["groups"]["natural_G"],
                                      generated["outputs"], strict=True)
               if output["prediction"]["failure"] is None]
    return score_panels({"controlled": controlled, "natural_G": natural},
                        POSTHOC, "posthoc_opened_holdout_not_new_blind_test",
                        sha256(source_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "train", "dev", "posthoc"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(OVERLAY))
    elif stage == "audit":
        print(audit(read_json(OVERLAY)), sha256(OVERLAY))
    elif stage == "train":
        print(train())
    else:
        report = dev() if stage == "dev" else posthoc()
        for name, panel in report["panels"].items():
            print(name, panel["metrics"]["macro_f1"],
                  panel["metrics"]["joint_diagnosis_accuracy"],
                  panel["metrics"]["interface_failure_count"])
