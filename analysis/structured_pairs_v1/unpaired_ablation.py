"""Keep stratified natural negatives; restore original positives as a focused ablation."""
from collections import Counter
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

import experiment as paired
from gridpath.experiments import training
from gridpath.experiments.config import LEDGER, binding_for
from gridpath.experiments.data import load
from gridpath.planning.model import StepVerifier
from gridpath.planning.verification import Verifier
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from transformers import AutoTokenizer

NAME = "structured_pairs_unpaired_v1"
OUT = ROOT / "results" / NAME
OVERLAY = OUT / "overlay.json"
RUN = ROOT / "checkpoints" / NAME / "t5_base/verify"
DEV = OUT / "verification_dev.json"


def prepare():
    parent = paired.prepare()
    sources = {"entry": sha256(Path(__file__)),
               "protocol": sha256(HERE / "unpaired_protocol.md"),
               "paired_overlay": sha256(paired.OVERLAY)}
    if OVERLAY.exists():
        saved = read_json(OVERLAY)
        assert saved["sources"] == sources
        audit(saved)
        return saved
    originals = {r["sample_id"]: r for r in load(["train_verify"])["train_verify"]}
    restored = [originals[key] for key in parent["removed_ids"]
                if originals[key]["is_valid"]]
    assert len(restored) == 40
    rows = [r for r in parent["rows"] if r.get("origin")
            != "same_environment_shortest_counterfactual"] + restored
    result = {"sources": sources, "rows": rows,
              "restored_valid_ids": sorted(r["sample_id"] for r in restored)}
    audit(result)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OVERLAY, result)
    return result


def audit(data):
    assert data["sources"] == {"entry": sha256(Path(__file__)),
                               "protocol": sha256(HERE / "unpaired_protocol.md"),
                               "paired_overlay": sha256(paired.OVERLAY)}
    rows = data["rows"]
    paired_rows = paired.prepare()["rows"]
    assert len(rows) == len(paired_rows) == 1000
    assert len({r["sample_id"] for r in rows}) == 1000
    assert Counter(r["is_valid"] for r in rows) == {True: 500, False: 500}
    assert Counter(r["error_reason"] for r in rows if not r["is_valid"]) == {
        x: 125 for x in ("obstacle_collision", "out_of_bounds",
                         "invalid_action_token", "goal_not_reached")}
    assert {r["sample_id"] for r in rows if not r["is_valid"]} == {
        r["sample_id"] for r in paired_rows if not r["is_valid"]}
    assert sum(r["is_valid"] and len(r["candidate_plan"]) > r["shortest_distance"]
               for r in rows) == 200
    assert len(data["restored_valid_ids"]) == 40
    assert {r["sample_id"] for r in rows if r.get("origin")
            == "frozen_G_on_unseen_train_environment"} == {
                r["sample_id"] for r in paired_rows if r.get("origin")
                == "frozen_G_on_unseen_train_environment"}
    return {"rows": len(rows), "same_negatives": 500, "restored_valid": 40}


def binding():
    cfg, _, dev, base = binding_for("improvement_verify_v1",
                                    "t5_base", "verify", True)
    rows = prepare()["rows"]
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {"experiment": NAME, "base_binding": base["fingerprint"],
               "config": cfg, "overlay_sha256": sha256(OVERLAY),
               "entry_sha256": sha256(Path(__file__)),
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
    old_binding, old_budget = training.binding_for, training.budget
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        training.binding_for, training.budget = old_binding, old_budget


def evaluate():
    cfg, _, _, identity = binding()
    assert read_json(RUN / "completed.json")["fingerprint"] == identity["fingerprint"]
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    if DEV.exists():
        result = read_json(DEV)
        assert result["fingerprint"] == identity["fingerprint"]
        return result
    fixed, natural = paired.development_items()
    with GPUBudget(LEDGER, 20, 3600, f"{NAME}:t5_base:dev") as budget:
        tok = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
        net = StepVerifier.from_pretrained(RUN / "latest", local_files_only=True).to("cuda").eval()
        verifier = Verifier(net, tok, cfg, budget)
        panels = {"fixed_V80": paired.score(fixed, verifier),
                  "natural_G60": paired.score(natural, verifier)}
        assert verifier.calls == 140 and verifier.hits == 0
        result = {"fingerprint": identity["fingerprint"],
                  "overlay_sha256": sha256(OVERLAY),
                  "model_sha256": sha256(RUN / "latest/model.safetensors"),
                  "panels": panels, "model_calls": verifier.calls}
        write_json(DEV, result)
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "train", "dev"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(OVERLAY))
    elif stage == "audit":
        print(audit(read_json(OVERLAY)), sha256(OVERLAY))
    elif stage == "train":
        print(train())
    else:
        report = evaluate()
        for name, panel in report["panels"].items():
            print(name, panel["metrics"]["macro_f1"],
                  panel["metrics"]["joint_diagnosis_accuracy"],
                  panel["metrics"]["interface_failure_count"])
