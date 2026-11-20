"""Compare the frozen development candidates with original structured V cases."""
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridpath.training_config import read_json, write_json

NEW = ROOT / "results/structured_pairs_v1/verification_dev.json"
BASE = ROOT / "results/improvement_verify_v1/t5_base/dev"
OUT = ROOT / "results/structured_pairs_v1/comparison_dev.json"


def compare_panel(name):
    cases = read_json(NEW)["panels"][name]["cases"]
    directory = BASE / ("verification_cases" if name == "fixed_V80" else "cases")
    counts = Counter()
    changed = []
    for case in cases:
        row = case["row"]
        old = read_json(directory / (row["sample_id"] + ".json"))
        assert old["sample_id"] == row["sample_id"]
        assert old["executor_feedback"] == row["target_text"]
        if name == "natural_G60":
            assert old["initial"]["text"].lower().split() == row["candidate_plan"]
        before = old["self_verification"]["effective_label"]
        after = case["prediction"]["text"] if case["prediction"]["failure"] is None else ""
        truth = row["target_text"]
        reason = row["error_reason"] or "valid"
        counts[(reason, "total")] += 1
        counts[(reason, "before_joint")] += before == truth
        counts[(reason, "after_joint")] += after == truth
        if reason == "obstacle_collision":
            counts[(reason, "before_detected")] += before.startswith("invalid |")
            counts[(reason, "after_detected")] += after.startswith("invalid |")
        if before != after:
            changed.append({"sample_id": row["sample_id"], "truth": truth,
                            "before": before, "after": after})
    table = {reason: {metric: counts[(reason, metric)] for metric in
                      ("total", "before_joint", "after_joint", "before_detected",
                       "after_detected")} for reason in
             ("valid", "obstacle_collision", "out_of_bounds",
              "invalid_action_token", "goal_not_reached")}
    return {"counts": table, "changed": changed,
            "interface_failures": sum(c["prediction"]["failure"] is not None
                                      for c in cases)}


def main():
    fixed = compare_panel("fixed_V80")
    natural = compare_panel("natural_G60")
    base_collision = natural["counts"]["obstacle_collision"]
    fixed_counts = fixed["counts"]
    guard = {
        "natural_collision_joint_improves":
            base_collision["after_joint"] > base_collision["before_joint"],
        "no_valid_false_positives": all(panel["counts"]["valid"]["after_joint"]
                                        == panel["counts"]["valid"]["total"]
                                        for panel in (fixed, natural)),
        "other_reasons_no_regression": all(
            panel["counts"][reason]["after_joint"] >= panel["counts"][reason]["before_joint"]
            for panel in (fixed, natural)
            for reason in ("out_of_bounds", "invalid_action_token", "goal_not_reached")),
        "fixed_classification_no_regression":
            read_json(NEW)["panels"]["fixed_V80"]["metrics"]["classification_accuracy"] >= .95,
        "no_interface_failures": fixed["interface_failures"] == natural["interface_failures"] == 0,
    }
    output = {"fixed_V80": fixed, "natural_G60": natural,
              "guard": guard, "eligible_for_fresh_holdout": all(guard.values())}
    if OUT.exists() and read_json(OUT) != output:
        raise ValueError("Development comparison changed")
    write_json(OUT, output)
    return output


if __name__ == "__main__":
    result = main()
    for panel in ("fixed_V80", "natural_G60"):
        print(panel, result[panel]["counts"])
    print("guard", result["guard"], "eligible", result["eligible_for_fresh_holdout"])
