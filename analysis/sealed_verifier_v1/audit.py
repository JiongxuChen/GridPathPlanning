"""Read-only CPU audit of the frozen challenge and saved neural predictions."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import evaluate
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.planning.protocol import aggregate
from gridpath.planning.verification import metrics
from gridpath.training_config import read_json, sha256


def main():
    challenge = read_json(evaluate.FROZEN)
    result = read_json(evaluate.RESULT)
    provenance = evaluate.audit(challenge)
    assert result["audit"] == provenance
    assert result["challenge_sha256"] == sha256(evaluate.FROZEN)
    assert result["sources"] == evaluate.source_hashes()
    assert result["model_calls"] == 160 and result["cache_hits"] == 0
    cases = []
    for group in evaluate.GROUPS:
        panel = result["group_results"][group]
        assert len(panel["cases"]) == panel["metrics"]["count"] == evaluate.QUOTA
        for source, case in zip(challenge["groups"][group], panel["cases"], strict=True):
            row, prediction, detail = case["row"], case["prediction"], case["detail"]
            assert source["row"] == row
            assert execute(grid_from_record(row), row["candidate_plan"]).verification_label == row["target_text"]
            assert detail.get("executor_used") is False
            if prediction["failure"] is None:
                assert prediction["text"] == aggregate(detail["tags"], len(row["candidate_plan"]))
                assert len(detail["probabilities"]) == len(detail["tags"])
            cases.append(case)
        assert panel["metrics"] == metrics([x["row"] for x in panel["cases"]],
                                           panel["cases"])
    assert result["overall"] == metrics([x["row"] for x in cases], cases)
    return {"status": "passed", "layouts": provenance["layouts"],
            "model_calls": result["model_calls"], "cases": len(cases),
            "joint_correct": sum(x["prediction"]["text"] == x["row"]["target_text"]
                                 for x in cases)}


if __name__ == "__main__":
    print(main())
