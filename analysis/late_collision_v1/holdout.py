"""One-time paired test for the selected original V and calibrated gate."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis/local_occupancy_v1"), str(HERE)]

import calibration
import compare
import experiment
from gridpath.training_config import read_json, sha256, write_json

RESULT = experiment.OUT / "holdout_result.json"


def main():
    decision = read_json(calibration.DECISION)
    if decision != calibration.decide():
        raise ValueError("Calibration choice changed")
    threshold = decision["selected"]
    if threshold is None:
        raise ValueError("No gate was selected; do not use reserved test")
    if RESULT.exists():
        report = read_json(RESULT)
        if report["decision_sha256"] != sha256(calibration.DECISION):
            raise ValueError("Reserved test result choice mismatch")
        return report
    paired = compare.compare("test")
    groups = paired["old_groups"]
    baseline = calibration.counts(groups, None)
    gated = calibration.counts(groups, threshold)
    changed = {}
    for name in calibration.GROUPS:
        changed[name] = [
            {"sample_id": case["row"]["sample_id"],
             "truth": case["row"]["target_text"],
             "argmax": case["prediction"]["text"],
             "gate": calibration.gated_text(case, threshold)}
            for case in groups[name]["cases"]
            if calibration.gated_text(case, threshold) != case["prediction"]["text"]
        ]
    report = {"decision_sha256": sha256(calibration.DECISION),
              "test_panel_sha256": sha256(experiment.FROZEN),
              "paired_predictions_sha256": sha256(experiment.OUT / "comparison_test.json"),
              "threshold": threshold, "baseline": baseline, "gated": gated,
              "changed": changed}
    write_json(RESULT, report)
    return report


if __name__ == "__main__":
    print(main())
