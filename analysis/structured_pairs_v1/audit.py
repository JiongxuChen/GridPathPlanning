"""Read-only audit independent of the two runner modules' import names."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridpath.data_schema import grid_from_record, layout_split
from gridpath.executor import execute
from gridpath.experiments.data import canonical_row
from gridpath.planning.events import apply_events, choices, goal_endpoint
from gridpath.planning.protocol import actions_of, aggregate
from gridpath.planning.verification import metrics
from gridpath.training_config import digest, read_json, sha256
from gridpath.types import ErrorReason

OUT = ROOT / "results/structured_pairs_v1"
SEED = 20261125


def main():
    excluded = read_json(OUT / "exclusions.json")
    challenge = read_json(OUT / "holdout_challenge.json")
    generated = read_json(OUT / "holdout_G_outputs.json")
    result = read_json(OUT / "holdout_predictions.json")
    expected_sources = {
        "entry": sha256(HERE / "holdout.py"),
        "protocol": sha256(HERE / "holdout_protocol.md"),
        "exclusions": sha256(OUT / "exclusions.json"),
        "dataset": sha256(ROOT / "manifests/dataset_v2.json"),
        "overlay": sha256(OUT / "overlay.json"),
    }
    assert challenge["sources"] == result["sources"] == expected_sources
    assert challenge["seed"] == SEED
    assert result["challenge_sha256"] == generated["challenge_sha256"] == sha256(
        OUT / "holdout_challenge.json")
    assert result["generated_sha256"] == sha256(OUT / "holdout_G_outputs.json")
    assert len(challenge["groups"]["paired"]) == 20
    assert len(challenge["groups"]["natural_G"]) == len(generated["outputs"]) == 60
    assert generated["model_calls"] == 60
    forbidden = set(excluded["layout_ids"])
    assert len(forbidden) == result["audit"]["excluded_prior_layouts"]
    used = set()
    controlled = []
    for pair in challenge["groups"]["paired"]:
        good, bad = pair["valid"], pair["collision"]
        grid = grid_from_record(good)
        lid = good["layout_id"]
        assert lid not in forbidden | used and layout_split(lid, 42) == "test"
        used.add(lid)
        assert lid == bad["layout_id"] and good["environment_id"] == bad["environment_id"]
        pool = [e for e in choices(grid, good["reference_plan"],
                                   ErrorReason.OBSTACLE_COLLISION, "return_block")
                if goal_endpoint(grid, apply_events(good["reference_plan"], [e]))]
        chosen = min(pool, key=lambda e: digest([SEED, lid, e.to_dict()]))
        assert pair["construction"] == chosen.to_dict()
        assert bad["candidate_plan"] == list(apply_events(good["reference_plan"], [chosen]))
        assert good["is_valid"] and bad["error_reason"] == "obstacle_collision"
        for row in (good, bad):
            assert row["target_text"] == execute(
                grid_from_record(row), row["candidate_plan"]).verification_label
            controlled.append((row, row["candidate_plan"]))
    natural = []
    failures = []
    for row, output in zip(challenge["groups"]["natural_G"],
                           generated["outputs"], strict=True):
        lid = row["layout_id"]
        assert lid not in forbidden | used and layout_split(lid, 42) == "test"
        used.add(lid)
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            assert output["truth"] == execute(
                grid_from_record(row), output["prediction"]["text"]).verification_label
            natural.append((row, output["prediction"]["text"]))
        else:
            assert output["truth"] is None
            failures.append(output)
    assert len(used) == 80 and result["G_interface_failures"] == failures
    assert result["audit"] == {"layouts": 80, "excluded_prior_layouts": len(forbidden),
                               "controlled_candidates": 40, "natural_tasks": 60}
    for name in ("original", "matched_pairs"):
        model = result["models"][name]
        calls = 0
        for panel, expected in (("controlled", controlled), ("natural_G", natural)):
            cases = model[panel]["cases"]
            assert len(cases) == len(expected)
            for case, (row, candidate) in zip(cases, expected, strict=True):
                annotated = canonical_row({**row, "task": "VERIFY",
                                           "candidate_plan": actions_of(candidate)})
                assert case["row"] == annotated
                assert case["row"]["target_text"] == execute(
                    grid_from_record(row), candidate).verification_label
                prediction, detail = case["prediction"], case["detail"]
                assert prediction["failure"] is None and detail["executor_used"] is False
                assert prediction["text"] == aggregate(
                    detail["tags"], len(case["row"]["candidate_plan"]))
                assert len(detail["probabilities"]) == len(detail["tags"])
                for probabilities in detail["probabilities"]:
                    assert len(probabilities) == 6 and abs(sum(probabilities) - 1) < 1e-5
            assert model[panel]["metrics"] == metrics(
                [c["row"] for c in cases], cases)
            calls += len(cases)
        assert model["model_calls"] == calls
    assert result["models"]["original"]["weights_sha256"] == sha256(
        ROOT / "checkpoints/improvement_verify_v1/t5_base/verify/latest/model.safetensors")
    assert result["models"]["matched_pairs"]["weights_sha256"] == sha256(
        ROOT / "checkpoints/structured_pairs_v1/t5_base/verify/latest/model.safetensors")
    return {"status": "passed", "layouts": len(used), "G_calls": generated["model_calls"],
            "V_calls_each": result["models"]["original"]["model_calls"],
            "G_failures": len(failures), "realized_natural_collisions":
                sum("reason=obstacle_collision" in x["truth"]
                    for x in generated["outputs"] if x["truth"])}


if __name__ == "__main__":
    print(main())
