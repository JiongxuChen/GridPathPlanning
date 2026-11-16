"""Equal-budget late-collision data ablation and frozen paired evaluation."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1")]

import representation
import train as local_train
from gridpath.bfs import shortest_path
from gridpath.data_schema import grid_from_record, layout_id, layout_split
from gridpath.executor import execute
from gridpath.experiments import training
from gridpath.experiments.data import MANIFEST, load, new_row
from gridpath.planning import protocol
from gridpath.planning.events import Event, apply_events, choices, safe_loop
from gridpath.resources import GPUBudget
from gridpath.training_config import digest, read_json, sha256, write_json
from gridpath.training_helpers import validate_checkpoint
from gridpath.types import ErrorReason, Grid

NAME = "late_collision_v1"
RUN = ROOT / "checkpoints" / NAME / "t5_base" / "verify"
OUT = ROOT / "results" / NAME
FROZEN = OUT / "prepared.json"
SEED = 20261117


def prefix_for(grid, base):
    _, loop = safe_loop(grid, base, layout_id(grid), at=0)
    prefix = tuple(loop.tokens) * 5
    assert execute(grid, prefix + tuple(base)).success
    return prefix


def collision_for(grid, base, prefix):
    events = sorted(choices(grid, base, ErrorReason.OBSTACLE_COLLISION, "return_block"),
                    key=lambda e: digest([SEED, layout_id(grid), e.to_dict()]))
    for event in events:
        late = prefix + apply_events(base, [event])
        early = apply_events(base, [event]) + prefix
        truth = execute(grid, late)
        if (11 <= truth.first_error_step <= 17 and
                execute(grid, early).first_error_step <= 7 and
                len(late) <= 24):
            return event, late, early
    return None


def annotated(grid, split, candidate, base=None, event=None):
    row = new_row(grid, split, "VERIFY", candidate)
    if event is not None:
        row["base_plan"] = list(base)
        row["error_events"] = [event.to_dict()]
        row["atomic_error_count"] = 1
        row["primitive_token_edits"] = event.primitive_token_edits
        row["single_counterfactual_candidates"] = [list(candidate)]
        assert tuple(candidate) == apply_events(base, [event])
    return row


def candidates(grid, base, split):
    prefix = prefix_for(grid, base)
    found = collision_for(grid, base, prefix)
    if found is None:
        return None
    event, late, early = found
    late_event = Event(event.kind, len(prefix) + event.index, event.tokens)
    long_base = prefix + tuple(base)
    early_base = tuple(base) + prefix
    assert execute(grid, long_base).success
    return {
        "short_valid": annotated(grid, split, base),
        "long_valid": annotated(grid, split, long_base),
        "early_collision": annotated(grid, split, early, early_base, event),
        "late_collision": annotated(grid, split, late, long_base, late_event),
    }


def prepare():
    sources = {"script": sha256(Path(__file__)), "protocol": sha256(HERE / "protocol.md"),
               "dataset": sha256(MANIFEST), "representation": sha256(Path(representation.__file__))}
    if FROZEN.exists():
        data = read_json(FROZEN)
        if data["sources"] != sources:
            raise ValueError("Frozen preparation source changed")
        audit(data)
        return data
    original = load(["train_generate", "train_verify", "dev_samples",
                     "test_id_samples", "test_ood_samples"])
    v = original["train_verify"]
    valid = sorted((r for r in v if r["is_valid"] and r["positive_construction"]),
                   key=lambda r: digest([SEED, r["sample_id"]]))
    hard_collision = sorted((r for r in v if r["error_reason"] == "obstacle_collision"
                             and r["unchecked_endpoint_is_goal"]),
                            key=lambda r: digest([SEED, r["sample_id"]]))
    replacements = []
    for row in valid:
        grid = grid_from_record(row)
        try:
            pair = candidates(grid, row["reference_plan"], "train")
        except ValueError:
            continue
        if pair is not None:
            replacements.append((row, pair))
        if len(replacements) == 20:
            break
    if len(replacements) != 20 or len(hard_collision) < 20:
        raise RuntimeError("Cannot construct the exact replacement quota")
    removed = {r["sample_id"] for r, _ in replacements}
    removed.update(r["sample_id"] for r in hard_collision[:20])
    rows = [r for r in v if r["sample_id"] not in removed]
    for _, pair in replacements:
        rows.extend([pair["long_valid"], pair["late_collision"]])
    assert len(rows) == 1000

    old = read_json(local_train.OUT / "fresh_holdout.json")
    robust = read_json(local_train.OUT / "robustness_v1_challenge.json")
    forbidden = {r["layout_id"] for group in original.values() for r in group}
    forbidden.update(r["layout_id"] for name in
                     ("fixed_positive", "fixed_hard_collision", "generate") for r in old[name])
    forbidden.update(x["row"]["layout_id"] for items in robust["groups"].values() for x in items)
    panels = {"dev": [], "test": []}
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(6) for c in range(6)]
    attempts = Counter()
    for split in panels:
        while len(panels[split]) < 20:
            attempts[split] += 1
            if attempts[split] > 100000:
                raise RuntimeError("Cannot fill isolated layout panel")
            blocked = frozenset(rng.sample(cells, rng.randint(2, 5)))
            free = [p for p in cells if p not in blocked]
            start, goal = rng.sample(free, 2)
            grid = Grid(6, 6, blocked, start, goal)
            lid = layout_id(grid)
            if lid in forbidden or layout_split(lid, 42) != split:
                continue
            ref = shortest_path(grid)
            if not ref.reachable or not 2 <= ref.distance <= 7:
                continue
            try:
                pair = candidates(grid, ref.actions, "late_" + split)
            except ValueError:
                continue
            if pair is None:
                continue
            panels[split].append(pair)
            forbidden.add(lid)
    data = {"seed": SEED, "sources": sources, "removed_sample_ids": sorted(removed),
            "training_rows": rows, "panels": panels, "attempts": dict(attempts)}
    audit(data)
    write_json(FROZEN, data)
    return data


def audit(data):
    rows = data["training_rows"]
    assert len(rows) == len({r["sample_id"] for r in rows}) == 1000
    assert Counter(r["is_valid"] for r in rows) == {True: 500, False: 500}
    assert Counter(r["error_reason"] for r in rows if not r["is_valid"]) == {
        e.value: 125 for e in ErrorReason}
    assert sum(r["is_valid"] and len(r["candidate_plan"]) > r["shortest_distance"]
               for r in rows) == 200
    assert sum(r["error_reason"] == "obstacle_collision" and
               r["unchecked_endpoint_is_goal"] for r in rows) == 83
    assert len(data["removed_sample_ids"]) == 40
    g_layouts = {r["layout_id"] for r in load(["train_generate"])["train_generate"]}
    assert all(r["layout_id"] in g_layouts for r in rows)
    for row in rows:
        grid = grid_from_record(row)
        assert execute(grid, row["candidate_plan"]).verification_label == row["target_text"]
        if row["error_events"] and row["split"] == "train":
            from gridpath.planning.events import event_from_dict
            events = [event_from_dict(e) for e in row["error_events"]]
            assert tuple(row["candidate_plan"]) == apply_events(row["base_plan"], events)
    seen = set()
    for split, pairs in data["panels"].items():
        assert len(pairs) == 20
        for pair in pairs:
            lids = {r["layout_id"] for r in pair.values()}
            assert len(lids) == 1
            lid = lids.pop()
            assert lid not in seen and layout_split(lid, 42) == split
            seen.add(lid)
            assert all(execute(grid_from_record(r), r["candidate_plan"]).verification_label
                       == r["target_text"] for r in pair.values())
            assert pair["short_valid"]["is_valid"] and pair["long_valid"]["is_valid"]
            late = pair["late_collision"]
            early = pair["early_collision"]
            assert late["error_reason"] == early["error_reason"] == "obstacle_collision"
            assert 11 <= int(late["target_text"].split("step=")[1].split()[0]) <= 17
            assert int(early["target_text"].split("step=")[1].split()[0]) <= 7
            assert late["unchecked_endpoint_is_goal"]
    return {"training": len(rows), "replaced": len(data["removed_sample_ids"]),
            "panels": {k: len(v) for k, v in data["panels"].items()}}


def binding():
    data = prepare()
    cfg, _, dev, baseline = local_train.binding()
    cfg = {**cfg, "run_directory": str(RUN.relative_to(ROOT))}
    payload = {
        "experiment": NAME, "baseline_binding": baseline["fingerprint"],
        "config": cfg, "training_rows": sha256(FROZEN),
        "representation_sha256": sha256(Path(representation.__file__)),
        "training_entry_sha256": sha256(Path(__file__)),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
    }
    return cfg, data["training_rows"], dev, {"fingerprint": digest(payload), **payload}


def train():
    cfg, _, _, identity = binding()
    if (RUN / "completed.json").exists():
        done = read_json(RUN / "completed.json")
        if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
            raise ValueError("Completed checkpoint identity changed")
        validate_checkpoint(RUN / "latest", identity["fingerprint"])
        return done
    old = protocol.render, training.binding_for, training.budget
    protocol.render = representation.render
    training.binding_for = lambda method, model, role, verify_weights=False: binding()
    training.budget = lambda method, model, stage: GPUBudget(
        local_train.LEDGER, cfg["gpu_hours_cap"], cfg["session_wall_seconds"],
        f"{NAME}:{model}:{stage}")
    try:
        return training.run("improvement_verify_v1", "t5_base", "verify")
    finally:
        protocol.render, training.binding_for, training.budget = old


def score(split):
    from transformers import AutoTokenizer
    from gridpath.planning.model import StepVerifier
    from gridpath.planning.verification import Verifier
    import evaluate_dev

    data = prepare()
    cfg, _, _, identity = binding()
    done = read_json(RUN / "completed.json")
    if done["fingerprint"] != identity["fingerprint"] or done["state"]["global_step"] != 750:
        raise ValueError("Training incomplete or identity mismatch")
    validate_checkpoint(RUN / "latest", identity["fingerprint"])
    target = OUT / f"{split}_predictions.json"
    if target.exists():
        saved = read_json(target)
        if saved["training_fingerprint"] != identity["fingerprint"]:
            raise ValueError("Existing evaluation identity mismatch")
        return saved
    original = protocol.render
    protocol.render = representation.render
    try:
        with GPUBudget(local_train.LEDGER, cfg["gpu_hours_cap"],
                       cfg["session_wall_seconds"], f"{NAME}:t5_base:{split}") as budget:
            tokenizer = AutoTokenizer.from_pretrained(RUN / "latest", local_files_only=True)
            model = StepVerifier.from_pretrained(RUN / "latest", local_files_only=True).to("cuda").eval()
            verifier = Verifier(model, tokenizer, cfg, budget)
            groups = {name: evaluate_dev.evaluate_group(
                [(pair[name], pair[name]["candidate_plan"]) for pair in data["panels"][split]],
                verifier) for name in ("short_valid", "long_valid", "early_collision", "late_collision")}
            report = {"training_fingerprint": identity["fingerprint"],
                      "prepared_sha256": sha256(FROZEN), "split": split,
                      "gpu_calls": verifier.calls, "groups": groups}
            write_json(target, report)
            return report
    finally:
        protocol.render = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "dev", "test"))
    stage = parser.parse_args().stage
    if stage == "prepare":
        print(audit(prepare()), sha256(FROZEN))
    elif stage == "train":
        print(train())
    else:
        result = score(stage)
        for name, group in result["groups"].items():
            m = group["metrics"]
            count = sum(x["prediction"]["text"] == x["row"]["target_text"]
                        for x in group["cases"])
            print(stage, name, count, "/", len(group["cases"]),
                  "interface_failures", m["interface_failure_count"])
