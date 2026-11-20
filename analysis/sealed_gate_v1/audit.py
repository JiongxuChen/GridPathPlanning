"""Read-only audit of frozen candidates and saved raw/gated predictions."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/sealed_verifier_v1"))
sys.path.insert(0, str(ROOT / "analysis/local_occupancy_v1"))

prior_spec = importlib.util.spec_from_file_location(
    "evaluate", ROOT / "analysis/sealed_verifier_v1/evaluate.py")
prior_module = importlib.util.module_from_spec(prior_spec)
sys.modules["evaluate"] = prior_module
prior_spec.loader.exec_module(prior_module)
gate_spec = importlib.util.spec_from_file_location("gate_evaluate", HERE / "evaluate.py")
evaluate = importlib.util.module_from_spec(gate_spec)
sys.modules["gate_evaluate"] = evaluate
gate_spec.loader.exec_module(evaluate)
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.experiments.data import canonical_row
from gridpath.planning.protocol import actions_of, aggregate
from gridpath.planning.verification import metrics
from gridpath.training_config import read_json, sha256


def main():
    challenge = read_json(evaluate.FROZEN)
    generated = read_json(evaluate.GENERATED)
    result = read_json(evaluate.RESULT)
    provenance = evaluate.audit(challenge)
    assert result["audit"] == provenance and result["sources"] == evaluate.sources()
    assert result["challenge_sha256"] == sha256(evaluate.FROZEN)
    assert generated["challenge_sha256"] == sha256(evaluate.FROZEN)
    assert result["generated_sha256"] == sha256(evaluate.GENERATED)
    assert generated["model_calls"] == len(generated["outputs"]) == evaluate.QUOTA
    assert result["cache_hits"] == 0
    all_cases = []
    for group in evaluate.GROUPS:
        panel = result["panels"][group]
        originals = challenge["groups"][group]
        if group == "natural_G":
            pairs = list(zip(originals, generated["outputs"], strict=True))
            expected = [(x["row"], y["prediction"]["text"]) for x, y in pairs
                        if y["prediction"]["failure"] is None]
            failures = [y for _, y in pairs if y["prediction"]["failure"] is not None]
            assert panel["generator_failures"] == failures
            for item, output in pairs:
                assert item["row"]["sample_id"] == output["sample_id"]
                if output["prediction"]["failure"] is None:
                    assert output["truth"] == execute(
                        grid_from_record(item["row"]), output["prediction"]["text"]
                    ).verification_label
                else:
                    assert output["truth"] is None
        else:
            expected = [(x["row"], x["row"]["candidate_plan"]) for x in originals]
            assert not panel["generator_failures"]
        assert len(panel["cases"]) + len(panel["generator_failures"]) == evaluate.QUOTA
        for (row, candidate), case in zip(expected, panel["cases"], strict=True):
            annotated = canonical_row({**row, "task": "VERIFY",
                                       "candidate_plan": actions_of(candidate)})
            assert case["row"] == annotated
            assert case["row"]["target_text"] == execute(
                grid_from_record(row), candidate).verification_label
            pred, detail = case["prediction"], case["detail"]
            if pred["failure"] is None:
                assert pred["text"] == aggregate(detail["tags"], len(
                    case["row"]["candidate_plan"]))
                assert detail["executor_used"] is False
                assert len(detail["tags"]) == len(detail["probabilities"])
                for probs in detail["probabilities"]:
                    assert len(probs) == 6 and abs(sum(probs) - 1) < 1e-5
            else:
                assert pred["text"] == "" and not detail["tags"]
            all_cases.append(case)
        cases = panel["cases"]
        assert panel["raw"] == metrics([c["row"] for c in cases], cases)
        gated = [evaluate.gated_case(c) for c in cases]
        assert panel["gate"] == metrics([c["row"] for c in gated], gated)
    assert result["raw"] == metrics([c["row"] for c in all_cases], all_cases)
    gated = [evaluate.gated_case(c) for c in all_cases]
    assert result["gate"] == metrics([c["row"] for c in gated], gated)
    assert result["model_calls"] == sum(c["prediction"]["failure"] is None
                                        for c in all_cases)
    return {"status": "passed", "layouts": provenance["layouts"],
            "G_calls": generated["model_calls"], "V_calls": result["model_calls"],
            "V_interface_failures": result["raw"]["interface_failure_count"],
            "raw_exact": sum(c["prediction"]["text"] == c["row"]["target_text"]
                             for c in all_cases),
            "gate_exact": sum(c["prediction"]["text"] == c["row"]["target_text"]
                              for c in gated)}


if __name__ == "__main__":
    print(main())
