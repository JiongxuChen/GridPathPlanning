"""Build a traceable union of all earlier layout IDs; standard library only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/structured_pairs_v1/exclusions.json"
JSON_PATHS = (
    "analysis/local_occupancy_v1/fresh_holdout.json",
    "analysis/local_occupancy_v1/robustness_v1_challenge.json",
    "analysis/natural_collision_v1/pool.json",
    "results/late_collision_v1/prepared.json",
    "results/late_collision_v1/calibration_frozen.json",
    "results/representation_attribution_v1/challenge.json",
    "results/sealed_verifier_v1/challenge.json",
    "results/sealed_gate_v1/challenge.json",
)
DATA_PATHS = tuple(f"data/v2/shared/{name}.jsonl" for name in
                   ("train_generate", "train_verify", "dev_samples",
                    "test_id_samples", "test_ood_samples"))


def visit(value, target):
    if isinstance(value, dict):
        if "layout_id" in value:
            target.add(value["layout_id"])
        for child in value.values():
            visit(child, target)
    elif isinstance(value, list):
        for child in value:
            visit(child, target)


def main():
    paths = list(JSON_PATHS)
    paths.extend(str(p.relative_to(ROOT)).replace("\\", "/")
                 for p in sorted((ROOT / "results/late_collision_v2").glob("*.json")))
    paths.extend(DATA_PATHS)
    ids = set()
    hashes = {}
    for name in paths:
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name.endswith(".jsonl"):
            for line in path.open(encoding="utf-8"):
                visit(json.loads(line), ids)
        else:
            visit(json.loads(path.read_text(encoding="utf-8")), ids)
    output = {"sources": hashes, "layout_ids": sorted(ids)}
    if OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) != output:
        raise ValueError("Previously frozen exclusion manifest differs")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("excluded", len(ids), "sources", len(paths),
          "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
