"""CPU-only provenance and prediction audit for the new verifier challenge."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridpath.data_schema import grid_from_record, layout_split
from gridpath.executor import execute
from gridpath.planning.protocol import aggregate

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def file_hash(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def check_cases(cases):
    for item in cases:
        row, prediction, detail = item["row"], item["prediction"], item["detail"]
        assert execute(grid_from_record(row), row["candidate_plan"]).verification_label == row["target_text"]
        assert prediction["failure"] is None and detail["executor_used"] is False
        assert prediction["text"] == aggregate(detail["tags"], len(row["candidate_plan"]))


def main():
    challenge = read("fresh_holdout.json")
    results = read("fresh_holdout_predictions.json")
    generated = read("fresh_G_outputs.json")
    development = read("verification_dev.json")
    assert challenge["source_sha256"] == file_hash("fresh_holdout.py")
    assert results["challenge_sha256"] == generated["challenge_sha256"] == file_hash("fresh_holdout.json")
    assert development["evaluator_sha256"] == file_hash("evaluate_dev.py")
    assert results["local_fingerprint"] == development["training_fingerprint"]
    assert (results["generator_calls"], results["baseline_verifier_calls"],
            results["local_verifier_calls"]) == (60, 140, 140)
    assert results["baseline_cache_hits"] == results["local_cache_hits"] == 0
    rows = challenge["fixed_positive"] + challenge["fixed_hard_collision"] + challenge["generate"]
    old_layouts = set()
    for name in ("train_generate", "train_verify", "dev_samples",
                 "test_id_samples", "test_ood_samples"):
        for line in (ROOT / "data/v2/shared" / (name + ".jsonl")).read_text().splitlines():
            old_layouts.add(json.loads(line)["layout_id"])
    assert len(rows) == 140
    assert len({r["sample_id"] for r in rows}) == len({r["layout_id"] for r in rows}) == 140
    assert not {r["layout_id"] for r in rows} & old_layouts
    assert all(layout_split(r["layout_id"], 42) == "test" for r in rows)
    assert all(r["is_valid"] for r in challenge["fixed_positive"])
    assert all(r["error_reason"] == "obstacle_collision" and r["unchecked_endpoint_is_goal"]
               for r in challenge["fixed_hard_collision"])
    by_id = {r["sample_id"]: r for r in challenge["generate"]}
    assert len(generated["outputs"]) == 60
    for item in generated["outputs"]:
        row = by_id[item["sample_id"]]
        assert execute(grid_from_record(row), item["candidate"]).verification_label == item["truth"]
    for name in ("baseline", "local"):
        for panel in ("fixed", "natural"):
            check_cases(results[name][panel]["cases"])
    for name in ("fixed_V80", "natural_G60"):
        check_cases(development[name]["cases"])
    return {"status": "passed", "fresh_layouts": 140, "fresh_G": 60,
            "verifier_calls_each": 140, "cached_verifier_calls": 0}


if __name__ == "__main__":
    print(main())
