"""Read-only audit of late-collision and source-limit predictions."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_count(detail):
    if "input_tokens" in detail:
        return detail["input_tokens"]
    match = re.search(r"(\d+) token", detail.get("failure", ""))
    return int(match.group(1)) if match else None


def positions(row):
    result = [tuple(row["start"])]
    for action in row["candidate_plan"]:
        dr, dc = DELTA.get(action, (0, 0))
        r, c = result[-1]
        result.append((r + dr, c + dc))
    return result


def main():
    challenge = read(HERE / "robustness_v1_challenge.json")
    predictions = read(HERE / "robustness_v1_predictions.json")
    late = []
    for index, item in enumerate(predictions["local"]["late_collision_6"]["cases"]):
        row = item["row"]
        truth = int(row["target_text"].split("step=")[1].split()[0])
        predicted = int(item["prediction"]["text"].split("step=")[1].split()[0])
        pos = positions(row)
        probabilities = item["detail"]["probabilities"]
        blocked = {tuple(p) for p in row["obstacles"]}
        def step(n):
            at = pos[n]
            return {
                "action": row["candidate_plan"][n - 1],
                "proposed": at,
                "center_blocked": at in blocked,
                "continue_probability": round(probabilities[n - 1][0], 6),
                "collision_probability": round(probabilities[n - 1][1], 6),
            }
        late.append({
            "index": index, "sample_id": row["sample_id"],
            "truth_step": truth, "predicted_step": predicted,
            "event": challenge["groups"]["late_collision_6"][index]["construction"]["events"][0],
            "preceding": step(truth - 1), "actual": step(truth),
            "input_tokens": token_count(item["detail"]),
        })
    wrong = [x for x in late if x["truth_step"] != x["predicted_step"]]
    assert len(late) == 20 and len(wrong) == 5
    assert all(x["predicted_step"] == x["truth_step"] - 1 for x in wrong)
    boundary = []
    for item in predictions["local"]["token_boundary"]["cases"]:
        row, detail = item["row"], item["detail"]
        boundary.append({
            "sample_id": row["sample_id"],
            "actions": len(row["candidate_plan"]),
            "input_tokens": token_count(detail),
            "interface_failure": item["prediction"]["failure"],
            "correct": item["prediction"]["text"] == row["target_text"],
        })
    historical = {}
    for split in ("test_id", "test_ood"):
        directory = HERE / "historical_pipeline" / split
        cases = [read(p) for p in (directory / "cases").glob("*.json")]
        fixed = [read(p) for p in (directory / "verification_cases").glob("*.json")]
        all_cases = fixed + cases
        lengths = [len(c["self_verification"]["detail"]["tags"]) - 1
                   for c in all_cases]
        tokens = [token_count(c["self_verification"]["detail"]) for c in all_cases]
        assert len(all_cases) == 200 and all(x is not None for x in tokens)
        historical[split] = {
            "candidates": len(all_cases), "max_actions": max(lengths),
            "max_input_tokens": max(tokens), "interface_failures": sum(
                c["self_verification"]["prediction"]["failure"] is not None for c in all_cases),
        }
    return {
        "late_collision": {
            "count": len(late), "correct": len(late) - len(wrong),
            "wrong": wrong,
            "all": late,
        },
        "token_boundary": boundary, "historical_6x6": historical,
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=True, indent=2))
