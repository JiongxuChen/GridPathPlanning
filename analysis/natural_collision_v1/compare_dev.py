"""Pair development predictions against the unchanged structured verifier."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = json.loads((ROOT / "analysis/natural_collision_v1/verification_dev.json").read_text())
BASE = ROOT / "results/improvement_verify_v1/t5_base/dev"


def compare(panel, directory):
    transition = Counter()
    changed = []
    valid_false_positives = [0, 0]
    collision = [0, 0, 0, 0]
    for item in NEW[panel]["cases"]:
        row = item["row"]
        source = json.loads((directory / (row["sample_id"] + ".json")).read_text())
        assert source["sample_id"] == row["sample_id"]
        if panel == "natural_G60":
            assert source["initial"]["text"].lower().split() == row["candidate_plan"]
        assert source["executor_feedback"] == row["target_text"]
        before = source["self_verification"]["effective_label"]
        after = item["prediction"]["text"] if item["prediction"]["failure"] is None else ""
        truth = row["target_text"]
        transition[(before, after)] += 1
        if before != after:
            changed.append({"sample_id": row["sample_id"], "truth": truth,
                            "before": before, "after": after})
        if truth == "valid":
            valid_false_positives[0] += before != "valid"
            valid_false_positives[1] += after != "valid"
        if row["error_reason"] == "obstacle_collision":
            collision[0] += before != "valid"
            collision[1] += after != "valid"
            collision[2] += before == truth
            collision[3] += after == truth
    return {"total": len(NEW[panel]["cases"]), "changed_count": len(changed),
            "valid_false_positives_before_after": valid_false_positives,
            "collision_detected_before_after": collision[:2],
            "collision_joint_before_after": collision[2:],
            "changed": changed}


if __name__ == "__main__":
    result = {"fixed_V80": compare("fixed_V80", BASE / "verification_cases"),
              "natural_G60": compare("natural_G60", BASE / "cases")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
