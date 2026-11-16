"""Tokenizer-only 6x6 source-limit capacity for the frozen loop-path layouts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import representation
import train
from gridpath.data_schema import grid_from_record
from gridpath.executor import execute
from gridpath.training_config import read_json, sha256, write_json

CHALLENGE = train.OUT / "robustness_v1_challenge.json"
OUTPUT = train.OUT / "length_coverage_tokenizer_only.json"


def main():
    from transformers import AutoTokenizer

    challenge = read_json(CHALLENGE)
    tokenizer = AutoTokenizer.from_pretrained(train.RUN / "latest", local_files_only=True)
    cfg, _, _, identity = train.binding()
    limit = cfg["verify_source_tokens"]
    cap = cfg["max_plan_actions"]
    cases = []
    for group in ("long_valid_6", "late_collision_6", "token_boundary"):
        for item in challenge["groups"][group]:
            row = item["row"]
            grid = grid_from_record(row)
            base = tuple(row["reference_plan"])
            loop = tuple(row["candidate_plan"][:2])
            assert len(loop) == 2 and execute(grid, loop + base).success
            lengths = []
            for repeats in range((cap - len(base)) // 2 + 1):
                candidate = loop * repeats + base
                tokens = len(tokenizer(
                    representation.render(grid, candidate), truncation=False)["input_ids"])
                lengths.append({"actions": len(candidate), "input_tokens": tokens,
                                "accepted": tokens <= limit})
            accepted = [x for x in lengths if x["accepted"]]
            refused = [x for x in lengths if not x["accepted"]]
            assert accepted and (not refused or accepted[-1]["actions"] < refused[0]["actions"])
            cases.append({
                "sample_id": row["sample_id"], "group": group,
                "base_actions": len(base), "max_accepted": accepted[-1],
                "first_refused": refused[0] if refused else None,
            })
    result = {
        "source": "tokenizer_only_no_model_calls",
        "challenge_sha256": sha256(CHALLENGE),
        "verifier_fingerprint": identity["fingerprint"],
        "source_limit": limit, "action_limit": cap, "cases": cases,
    }
    write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    result = main()
    capacities = [x["max_accepted"]["actions"] for x in result["cases"]]
    print(json.dumps({
        "count": len(capacities), "min": min(capacities), "max": max(capacities),
        "capacities": capacities, "source_limit": result["source_limit"],
        "action_limit": result["action_limit"],
    }))
