"""Independent read-only audit and compact boundary summary."""
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]

import evaluate as exp
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.experiments.data import canonical_row
from gridpath.planning.protocol import TAGS, actions_of, aggregate
from gridpath.planning.verification import metrics
from gridpath.training_config import read_json, sha256, write_json


def summarize(cases):
    buckets = {}
    for key, selected in {
        "all": cases,
        "valid": [c for c in cases if c["row"]["is_valid"]],
        "collision": [c for c in cases if c["row"]["error_reason"]
                      == "obstacle_collision"],
        "other_invalid": [c for c in cases if not c["row"]["is_valid"]
                          and c["row"]["error_reason"] != "obstacle_collision"],
        "low_density": [c for c in cases if len(c["row"]["obstacles"]) <= 5],
        "high_density": [c for c in cases if len(c["row"]["obstacles"]) >= 6],
    }.items():
        buckets[key] = {
            "correct": sum(c["prediction"]["failure"] is None
                           and c["prediction"]["text"] == c["row"]["target_text"]
                           for c in selected),
            "denominator": len(selected),
            "interface_failures": sum(c["prediction"]["failure"] is not None
                                      for c in selected),
            "false_positive": sum(c["row"]["is_valid"]
                                  and c["prediction"]["failure"] is None
                                  and c["prediction"]["text"] != "valid"
                                  for c in selected),
        }
    return buckets


def main():
    challenge = read_json(exp.FROZEN)
    generated = read_json(exp.GENERATED)
    result = read_json(exp.RESULT)
    assert exp.audit(challenge)["layouts"] == 120
    assert result["challenge_sha256"] == generated["challenge_sha256"] == sha256(exp.FROZEN)
    assert result["generated_sha256"] == sha256(exp.GENERATED)
    assert generated["model_calls"] == len(generated["outputs"]) == exp.NATURAL
    assert not result["G_interface_failures"]
    assert len(result["models"]) == 5
    expected = {}
    for group in (*exp.GROUPS, "natural_G"):
        if group != "natural_G":
            expected[group] = [(item["row"], item["row"]["candidate_plan"])
                               for item in challenge["groups"][group]]
        else:
            expected[group] = []
            for item, output in zip(challenge["groups"][group],
                                    generated["outputs"], strict=True):
                row = item["row"]
                assert row["sample_id"] == output["sample_id"]
                if output["prediction"]["failure"] is None:
                    assert output["truth"] == execute(
                        grid_from_record(row), output["prediction"]["text"]
                    ).verification_label
                    expected[group].append((row, output["prediction"]["text"]))
    summary = {"status": "passed",
               "challenge_sha256": sha256(exp.FROZEN),
               "generated_sha256": sha256(exp.GENERATED),
               "predictions_sha256": sha256(exp.RESULT),
               "G_failures": len(result["G_interface_failures"]),
               "natural_truth": dict(Counter(
                   output["truth"] if output["truth"] == "valid"
                   else output["truth"].split("reason=")[-1]
                   for output in generated["outputs"]
                   if output["truth"] is not None)),
               "models": {}}
    aligned = None
    for name, model in result["models"].items():
        if name == "geometry_gate_0.7":
            checkpoint = exp.GEOMETRY
        elif name == "original":
            _, checkpoint, _ = exp.completed_model(
                "improvement_verify_v1", "t5_base", "verify")
        else:
            seed = int(name.removeprefix("set_seed"))
            checkpoint = (exp.relational.RUN if seed == 42
                          else exp.multi.run_path(seed)) / "latest"
        assert model["weights_sha256"] == sha256(checkpoint / "model.safetensors")
        all_cases = []
        signature = []
        panel_summary = {}
        for group, candidates in expected.items():
            panel = model["panels"][group]
            cases = panel["cases"]
            assert len(cases) == len(candidates)
            for (row, candidate), case in zip(candidates, cases, strict=True):
                annotated = canonical_row({**row, "task": "VERIFY",
                                           "candidate_plan": actions_of(candidate)})
                assert annotated == case["row"]
                assert annotated["target_text"] == execute(
                    grid_from_record(row), candidate).verification_label
                signature.append((annotated["sample_id"], annotated["target_text"]))
                pred, detail = case["prediction"], case["detail"]
                if pred["failure"] is None:
                    assert detail["executor_used"] is False
                    assert len(detail["tags"]) == len(detail["probabilities"])
                    assert len(detail["tags"]) == len(annotated["candidate_plan"]) + 1
                    for tag, probs in zip(detail["tags"],
                                          detail["probabilities"], strict=True):
                        assert 0 <= tag < len(TAGS)
                        assert len(probs) == len(TAGS)
                        assert abs(sum(probs) - 1) < 1e-4
                    tags = list(detail["tags"])
                    if name == "geometry_gate_0.7":
                        for i, tag in enumerate(tags[:-1]):
                            if tag == 1 and detail["probabilities"][i][1] < 0.7:
                                tags[i] = 0
                    assert pred["text"] == aggregate(tags, len(
                        annotated["candidate_plan"]))
                else:
                    assert pred["text"] == "" and not detail["tags"]
                all_cases.append(case)
            assert panel["metrics"] == metrics(
                [c["row"] for c in cases], cases)
            panel_summary[group] = summarize(cases)["all"]
        if aligned is None:
            aligned = signature
        else:
            assert aligned == signature
        failures = sum(c["prediction"]["failure"] is not None for c in all_cases)
        assert failures == model["source_failures"]
        assert model["model_calls"] + failures == len(all_cases) == 120
        summary["models"][name] = {
            "weights_sha256": model["weights_sha256"],
            "model_calls": model["model_calls"],
            "source_failures": failures,
            "panels": panel_summary,
            "breakdown": summarize(all_cases),
            "failure_reasons": dict(Counter(
                c["detail"]["failure"] for c in all_cases
                if c["prediction"]["failure"] is not None)),
        }
    write_json(exp.OUT / "audit_summary.json", summary)
    return summary


if __name__ == "__main__":
    report = main()
    print(report["status"], report["challenge_sha256"],
          report["predictions_sha256"], report["natural_truth"])
    for name, item in report["models"].items():
        print(name, "all", item["breakdown"]["all"],
              "collision", item["breakdown"]["collision"],
              "valid", item["breakdown"]["valid"],
              "failure_reasons", item["failure_reasons"])
