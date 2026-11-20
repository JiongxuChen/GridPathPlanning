"""CPU checks for the isolated natural-collision training overlay."""
from collections import Counter

from analysis.natural_collision_v1 import experiment as exp
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.experiments.data import new_row
from gridpath.planning.protocol import virtual_states
from gridpath.training_config import sha256, write_json
from gridpath.types import Grid


def test_candidate_environments_use_unseen_train_tasks(monkeypatch):
    grid = Grid(6, 6, frozenset({(0, 1)}), (0, 0), (1, 1))
    train = [new_row(grid, "train", "GENERATE")]
    monkeypatch.setattr(exp, "load", lambda names: {
        "train_generate": train, "train_verify": [], "dev_samples": [],
        "test_id_samples": [], "test_ood_samples": []})
    new = exp.candidate_environments(train, count=8)
    assert len({r["environment_id"] for r in new}) == 8
    assert all(r["layout_id"] == train[0]["layout_id"] for r in new)
    assert train[0]["environment_id"] not in {r["environment_id"] for r in new}


def test_overlay_replaces_both_classes_without_changing_budget(monkeypatch, tmp_path):
    grid_bad = Grid(6, 6, frozenset({(0, 1)}), (0, 0), (1, 1))
    grid_good = Grid(6, 6, grid_bad.obstacles, (0, 0), (2, 1))
    bad = new_row(grid_bad, "train", "GENERATE")
    good = new_row(grid_good, "train", "GENERATE")
    original = [
        {"sample_id": f"positive_{i}", "environment_id": f"old_positive_{i}",
         "is_valid": True, "error_reason": None}
        for i in range(500)
    ] + [
        {"sample_id": f"negative_{i}", "environment_id": f"old_negative_{i}", "is_valid": False,
         "error_reason": "obstacle_collision" if i < 125 else "out_of_bounds"}
        for i in range(500)
    ]
    monkeypatch.setattr(exp, "load", lambda names: {
        "train_verify": original, "train_generate": [bad]} if names == ["train_verify"]
        else {"train_generate": [bad]})
    monkeypatch.setattr(exp, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(exp, "POOL", tmp_path / "pool.json")
    monkeypatch.setattr(exp, "OVERLAY", tmp_path / "overlay.json")
    exp.MANIFEST.write_text("{}")

    def record(source, candidate):
        grid = grid_from_record(source)
        truth = execute(grid, candidate)
        return {"source": source, "prediction": {"text": candidate,
                "generated_tokens": 3, "failure": None},
                "truth": truth.verification_label,
                "unchecked_endpoint_is_goal": virtual_states(grid.start, candidate)[-1] == grid.goal}

    pool = {"dataset_sha256": sha256(exp.MANIFEST),
            "records": [record(bad, "right down"), record(good, "down down right")]}
    write_json(exp.POOL, pool)
    overlay = exp.make_overlay(pool, replacements=1)
    natural = [r for r in overlay["rows"] if r.get("origin")]
    assert len(overlay["rows"]) == 1000
    assert Counter(r["is_valid"] for r in overlay["rows"]) == {True: 500, False: 500}
    assert Counter(r["error_reason"] for r in natural) == {
        "obstacle_collision": 1, None: 1}
    assert natural[0]["paired_id"] == natural[1]["paired_id"]
    assert overlay["same_layout_pairs"] == 1
    assert exp.make_overlay(pool, replacements=1) == overlay
