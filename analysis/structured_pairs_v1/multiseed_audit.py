"""Independent read-only audit of the frozen multiseed panel and predictions."""
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

OUT = ROOT / "results/structured_pairs_multiseed_v1"
PRIOR = ROOT / "results/structured_pairs_v1"
SEED = 20261127


def main():
    exclusions = read_json(OUT / "exclusions.json")
    challenge = read_json(OUT / "challenge.json")
    generated = read_json(OUT / "G_outputs.json")
    result = read_json(OUT / "predictions.json")
    assert exclusions["sources"] == {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
        for p in (PRIOR / "exclusions.json", PRIOR / "holdout_challenge.json")}
    prior_ids = set(read_json(PRIOR / "exclusions.json")["layout_ids"])
    prior = read_json(PRIOR / "holdout_challenge.json")
    prior_ids.update(x["valid"]["layout_id"] for x in prior["groups"]["paired"])
    prior_ids.update(x["layout_id"] for x in prior["groups"]["natural_G"])
    assert set(exclusions["layout_ids"]) == prior_ids
    assert len(prior_ids) == 2240
    sources = {
        "entry": sha256(HERE / "multiseed.py"),
        "protocol": sha256(HERE / "multiseed_protocol.md"),
        "exclusions": sha256(OUT / "exclusions.json"),
        "dataset": sha256(ROOT / "manifests/dataset_v2.json"),
        "paired_overlay": sha256(PRIOR / "overlay.json"),
        "unpaired_overlay": sha256(
            ROOT / "results/structured_pairs_position_ablation_v1/overlay.json"),
        "holdout_generator": sha256(HERE / "holdout.py")}
    assert challenge["sources"] == sources and challenge["seed"] == SEED
    assert len(challenge["groups"]["paired"]) == 20
    assert len(challenge["groups"]["natural_G"]) == 60
    assert generated["challenge_sha256"] == result["challenge_sha256"] == sha256(
        OUT / "challenge.json")
    assert result["generated_sha256"] == sha256(OUT / "G_outputs.json")
    assert result["protocol_sha256"] == sources["protocol"]
    assert generated["model_calls"] == 60
    assert len(generated["outputs"]) == 60

    seen, controlled = set(), []
    for pair in challenge["groups"]["paired"]:
        good, bad = pair["valid"], pair["collision"]
        lid = good["layout_id"]
        assert lid == bad["layout_id"] and lid not in prior_ids | seen
        assert layout_split(lid, 42) == "test"
        seen.add(lid)
        grid = grid_from_record(good)
        assert grid.rows == grid.cols == 6
        pool = [event for event in choices(
            grid, good["reference_plan"], ErrorReason.OBSTACLE_COLLISION,
            "return_block") if goal_endpoint(
                grid, apply_events(good["reference_plan"], [event]))]
        chosen = min(pool, key=lambda event: digest([SEED, lid, event.to_dict()]))
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
        assert lid not in prior_ids | seen and layout_split(lid, 42) == "test"
        seen.add(lid)
        assert row["sample_id"] == output["sample_id"]
        if output["prediction"]["failure"] is None:
            assert output["truth"] == execute(
                grid_from_record(row), output["prediction"]["text"]).verification_label
            natural.append((row, output["prediction"]["text"]))
        else:
            assert output["truth"] is None
            failures.append(row["sample_id"])
    assert len(seen) == 80 and failures == result["G_interface_failures"]

    expected_names = {f"seed_{seed}_{arm}" for seed in (42, 43, 44)
                      for arm in ("paired", "position_unpaired")}
    expected_names.add("original_structured")
    assert set(result["models"]) == expected_names
    for name, model in result["models"].items():
        if name == "original_structured":
            run = ROOT / "checkpoints/improvement_verify_v1/t5_base/verify"
        elif name == "seed_42_paired":
            run = ROOT / "checkpoints/structured_pairs_v1/t5_base/verify"
        elif name == "seed_42_position_unpaired":
            run = ROOT / "checkpoints/structured_pairs_position_ablation_v1/t5_base/verify"
        else:
            _, seed, arm = name.split("_", 2)
            run = ROOT / "checkpoints/structured_pairs_multiseed_v1" / f"seed_{seed}" / arm / "t5_base/verify"
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
                prediction, detail = case["prediction"], case["detail"]
                assert prediction["failure"] is None
                assert detail["executor_used"] is False
                assert prediction["text"] == aggregate(
                    detail["tags"], len(annotated["candidate_plan"]))
                assert len(detail["tags"]) == len(detail["probabilities"])
                for probs in detail["probabilities"]:
                    assert len(probs) == 6 and abs(sum(probs) - 1) < 1e-5
            assert model[part]["metrics"] == metrics(
                [case["row"] for case in cases], cases)
    return {"status": "passed", "layouts": len(seen), "G_calls": 60,
            "V_calls": sum(x["model_calls"] for x in result["models"].values()),
            "models": len(result["models"]), "G_failures": len(failures)}


if __name__ == "__main__":
    print(main())
