"""Read-only audit of new layouts, labels, predictions, and checkpoint identities."""
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

OUT = ROOT / "results/relational_set_v1"
PRIOR = ROOT / "results/structured_pairs_multiseed_v1"
SEED = 20261128


def main():
    exclusions, challenge, generated, results = (
        read_json(OUT / name) for name in (
            "exclusions.json", "challenge.json", "G_outputs.json", "predictions.json"))
    previous = PRIOR / "exclusions.json"
    old_challenge = PRIOR / "challenge.json"
    assert exclusions["sources"] == {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
        for p in (previous, old_challenge)}
    prior_ids = set(read_json(previous)["layout_ids"])
    old = read_json(old_challenge)
    prior_ids.update(p["valid"]["layout_id"] for p in old["groups"]["paired"])
    prior_ids.update(r["layout_id"] for r in old["groups"]["natural_G"])
    assert len(prior_ids) == 2320 and set(exclusions["layout_ids"]) == prior_ids
    expected_sources = {
        "entry": sha256(HERE / "fresh.py"),
        "protocol": sha256(HERE / "protocol.md"),
        "exclusions": sha256(OUT / "exclusions.json"),
        "dataset": sha256(ROOT / "manifests/dataset_v2.json"),
        "overlay": sha256(ROOT / "results/structured_pairs_v1/overlay.json"),
        "relation_entry": sha256(HERE / "experiment.py"),
        "set_model": sha256(HERE / "set_model.py"),
        "holdout_generator": sha256(ROOT / "analysis/structured_pairs_v1/holdout.py")}
    assert challenge["sources"] == results["sources"] == expected_sources
    assert challenge["seed"] == SEED
    assert generated["challenge_sha256"] == results["challenge_sha256"] == sha256(
        OUT / "challenge.json")
    assert results["generated_sha256"] == sha256(OUT / "G_outputs.json")
    assert len(challenge["groups"]["paired"]) == 20
    assert len(challenge["groups"]["natural_G"]) == len(generated["outputs"]) == 60
    assert generated["model_calls"] == 60

    used, controlled = set(), []
    for pair in challenge["groups"]["paired"]:
        good, bad = pair["valid"], pair["collision"]
        grid = grid_from_record(good)
        lid = good["layout_id"]
        assert lid == bad["layout_id"] and lid not in prior_ids | used
        assert layout_split(lid, 42) == "test"
        used.add(lid)
        pool = [e for e in choices(
            grid, good["reference_plan"], ErrorReason.OBSTACLE_COLLISION,
            "return_block") if goal_endpoint(
                grid, apply_events(good["reference_plan"], [e]))]
        chosen = min(pool, key=lambda e: digest([SEED, lid, e.to_dict()]))
        assert pair["construction"] == chosen.to_dict()
        assert bad["candidate_plan"] == list(apply_events(
            good["reference_plan"], [chosen]))
        for row in (good, bad):
            assert execute(grid_from_record(row), row["candidate_plan"]).verification_label == row["target_text"]
            controlled.append((row, row["candidate_plan"]))
    natural, failures = [], []
    for row, output in zip(challenge["groups"]["natural_G"],
                           generated["outputs"], strict=True):
        lid = row["layout_id"]
        assert lid not in prior_ids | used and layout_split(lid, 42) == "test"
        used.add(lid)
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            assert output["truth"] == execute(
                grid_from_record(row), output["prediction"]["text"]).verification_label
            natural.append((row, output["prediction"]["text"]))
        else:
            assert output["truth"] is None
            failures.append(row["sample_id"])
    assert len(used) == 80 and failures == results["G_interface_failures"]
    assert set(results["models"]) == {"original", "paired_seed42", "learned_set_seed42"}

    for name, model in results["models"].items():
        run = {
            "original": ROOT / "checkpoints/improvement_verify_v1/t5_base/verify",
            "paired_seed42": ROOT / "checkpoints/structured_pairs_v1/t5_base/verify",
            "learned_set_seed42": ROOT / "checkpoints/relational_set_v1/t5_base/verify",
        }[name]
        assert model["fingerprint"] == read_json(run / "completed.json")["fingerprint"]
        assert model["weights_sha256"] == sha256(run / "latest/model.safetensors")
        assert model["model_calls"] == len(controlled) + len(natural)
        for part, expected in (("controlled", controlled), ("natural_G", natural)):
            cases = model[part]["cases"]
            assert len(cases) == len(expected)
            for case, (row, candidate) in zip(cases, expected, strict=True):
                annotated = canonical_row({**row, "task": "VERIFY",
                                           "candidate_plan": actions_of(candidate)})
                assert case["row"] == annotated
                pred, detail = case["prediction"], case["detail"]
                assert pred["failure"] is None and detail["executor_used"] is False
                assert pred["text"] == aggregate(
                    detail["tags"], len(annotated["candidate_plan"]))
                assert len(detail["tags"]) == len(detail["probabilities"])
                for probs in detail["probabilities"]:
                    assert len(probs) == 6 and abs(sum(probs) - 1) < 1e-5
            assert model[part]["metrics"] == metrics(
                [case["row"] for case in cases], cases)
    return {"status": "passed", "layouts": 80, "G_calls": 60,
            "V_calls": 300, "G_failures": len(failures)}


if __name__ == "__main__":
    print(main())
