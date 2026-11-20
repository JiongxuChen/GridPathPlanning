"""Freeze new layouts and G candidates, then compare two structured-offset verifiers."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

import experiment as exp
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.evaluation import Predictor
from gridpath.executor import execute
from gridpath.experiments.config import LEDGER, completed_model
from gridpath.experiments.data import MANIFEST, new_row
from gridpath.planning.events import apply_events, choices, goal_endpoint
from gridpath.planning.protocol import actions_of, virtual_states
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

OUT = exp.OUT
EXCLUSIONS = OUT / "exclusions.json"
FROZEN = OUT / "holdout_challenge.json"
GENERATED = OUT / "holdout_G_outputs.json"
RESULT = OUT / "holdout_predictions.json"
SEED = 20261125
QUOTAS = {"paired": 20, "natural_G": 60}


def sources():
    return {"entry": sha256(Path(__file__)),
            "protocol": sha256(HERE / "holdout_protocol.md"),
            "exclusions": sha256(EXCLUSIONS), "dataset": sha256(MANIFEST),
            "overlay": sha256(exp.OVERLAY)}


def paired_candidate(grid, base):
    pool = [event for event in choices(
        grid, base, ErrorReason.OBSTACLE_COLLISION, "return_block")
        if goal_endpoint(grid, apply_events(base, [event]))]
    if not pool:
        return None
    event = min(pool, key=lambda e: digest([SEED, layout_id(grid), e.to_dict()]))
    return apply_events(base, [event]), event.to_dict()


def audit(data):
    assert data["sources"] == sources() and data["seed"] == SEED
    forbidden = set(read_json(EXCLUSIONS)["layout_ids"])
    used = set()
    assert {x: len(data["groups"][x]) for x in QUOTAS} == QUOTAS
    for pair in data["groups"]["paired"]:
        good, bad = pair["valid"], pair["collision"]
        assert good["layout_id"] == bad["layout_id"]
        grid = grid_from_record(good)
        lid = good["layout_id"]
        assert lid not in used | forbidden and layout_split(lid, 42) == "test"
        used.add(lid)
        assert grid.rows == grid.cols == 6
        assert good["is_valid"] and good["target_text"] == "valid"
        made = paired_candidate(grid, good["reference_plan"])
        assert made and list(made[0]) == bad["candidate_plan"]
        assert made[1] == pair["construction"]
        assert bad["error_reason"] == "obstacle_collision"
        assert bad["unchecked_endpoint_is_goal"]
        assert execute(grid, bad["candidate_plan"]).verification_label == bad["target_text"]
    for row in data["groups"]["natural_G"]:
        grid = grid_from_record(row)
        lid = row["layout_id"]
        assert lid not in used | forbidden and layout_split(lid, 42) == "test"
        assert grid.rows == grid.cols == 6 and row["task"] == "GENERATE"
        used.add(lid)
    assert len(used) == 80
    return {"layouts": 80, "excluded_prior_layouts": len(forbidden),
            "controlled_candidates": 40, "natural_tasks": 60}


def prepare():
    if FROZEN.exists():
        saved = read_json(FROZEN)
        audit(saved)
        return saved
    forbidden = set(read_json(EXCLUSIONS)["layout_ids"])
    used = set(forbidden)
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    groups = {name: [] for name in QUOTAS}
    attempts = Counter()
    for group, count in QUOTAS.items():
        while len(groups[group]) < count:
            attempts[group] += 1
            if attempts[group] > 100000:
                raise RuntimeError("Cannot fill " + group)
            blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
            free = [cell for cell in cells if cell not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(6, 6, blocked, start, goal)
            lid = layout_id(grid)
            if lid in used or layout_split(lid, 42) != "test":
                continue
            reference = shortest_path(grid)
            if not reference.reachable or not 2 <= reference.distance <= 12:
                continue
            if group == "paired":
                made = paired_candidate(grid, reference.actions)
                if made is None:
                    continue
                candidate, construction = made
                valid = new_row(grid, "structured_pairs_holdout", "VERIFY",
                                reference.actions)
                collision = new_row(grid, "structured_pairs_holdout", "VERIFY",
                                    candidate)
                groups[group].append({"valid": valid, "collision": collision,
                                      "construction": construction})
            else:
                groups[group].append(new_row(
                    grid, "structured_pairs_holdout", "GENERATE"))
            used.add(lid)
    data = {"seed": SEED, "sources": sources(), "attempts": dict(attempts),
            "groups": groups}
    audit(data)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(FROZEN, data)
    return data


def generate():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    challenge = prepare()
    cfg, path, binding = completed_model("v2", "t5_base", "generate")
    validate_checkpoint(path, binding["fingerprint"])
    if GENERATED.exists():
        saved = read_json(GENERATED)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generator_fingerprint"] == binding["fingerprint"]
        return saved
    with GPUBudget(LEDGER, 20, 3600, f"{exp.NAME}:holdout_G") as budget:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        net = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to("cuda").eval()
        predictor = Predictor(net, tok, cfg, budget)
        outputs = []
        for row in challenge["groups"]["natural_G"]:
            prediction = predictor(row["input_text"])
            truth = (execute(grid_from_record(row), prediction.text).verification_label
                     if prediction.failure is None else None)
            outputs.append({"sample_id": row["sample_id"],
                            "prediction": asdict(prediction), "truth": truth})
        result = {"challenge_sha256": sha256(FROZEN),
                  "generator_fingerprint": binding["fingerprint"],
                  "generator_weights_sha256": sha256(path / "model.safetensors"),
                  "model_calls": predictor.calls, "outputs": outputs}
        write_json(GENERATED, result)
        return result


def score():
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier

    challenge, generated = prepare(), generate()
    assert generated["challenge_sha256"] == sha256(FROZEN)
    base_cfg, base_path, base_binding = completed_model(
        "improvement_verify_v1", "t5_base", "verify")
    candidate_cfg, _, _, candidate_binding = exp.binding()
    assert read_json(exp.RUN / "completed.json")["fingerprint"] == candidate_binding["fingerprint"]
    validate_checkpoint(exp.RUN / "latest", candidate_binding["fingerprint"])
    if RESULT.exists():
        saved = read_json(RESULT)
        assert saved["challenge_sha256"] == sha256(FROZEN)
        assert saved["generated_sha256"] == sha256(GENERATED)
        assert saved["fingerprints"] == {"original": base_binding["fingerprint"],
                                          "matched_pairs": candidate_binding["fingerprint"]}
        return saved
    fixed = [(row, row["candidate_plan"])
             for pair in challenge["groups"]["paired"]
             for row in (pair["valid"], pair["collision"])]
    natural = []
    failures = []
    for row, output in zip(challenge["groups"]["natural_G"],
                           generated["outputs"], strict=True):
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            assert output["truth"] == execute(
                grid_from_record(row), output["prediction"]["text"]).verification_label
            natural.append((row, output["prediction"]["text"]))
        else:
            failures.append(output)
    models = {}
    with GPUBudget(LEDGER, 20, 3600, f"{exp.NAME}:holdout_V") as budget:
        for name, cfg, path, binding in (
            ("original", base_cfg, base_path, base_binding),
            ("matched_pairs", candidate_cfg, exp.RUN / "latest", candidate_binding),
        ):
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            net = StepVerifier.from_pretrained(path, local_files_only=True).to("cuda").eval()
            verifier = Verifier(net, tok, cfg, budget)
            models[name] = {"controlled": exp.score(fixed, verifier),
                            "natural_G": exp.score(natural, verifier),
                            "model_calls": verifier.calls,
                            "weights_sha256": sha256(path / "model.safetensors")}
            assert verifier.calls == len(fixed) + len(natural) and verifier.hits == 0
            del verifier, net
    result = {"challenge_sha256": sha256(FROZEN),
              "generated_sha256": sha256(GENERATED),
              "sources": sources(), "audit": audit(challenge),
              "fingerprints": {"original": base_binding["fingerprint"],
                               "matched_pairs": candidate_binding["fingerprint"]},
              "G_interface_failures": failures, "models": models}
    write_json(RESULT, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "generate", "score"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif stage == "audit":
        print(audit(read_json(FROZEN)), sha256(FROZEN))
    elif stage == "generate":
        print(generate()["model_calls"], sha256(GENERATED))
    else:
        result = score()
        print("challenge", result["challenge_sha256"], "G", result["generated_sha256"])
        for name, model in result["models"].items():
            for panel in ("controlled", "natural_G"):
                cases = model[panel]["cases"]
                print(name, panel, sum(c["prediction"]["text"] == c["row"]["target_text"]
                                       for c in cases), "/", len(cases),
                      "F1", model[panel]["metrics"]["macro_f1"],
                      "V_failures", model[panel]["metrics"]["interface_failure_count"])
