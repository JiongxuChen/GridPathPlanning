# Verifier alternatives and local checkpoint archive

As of 2026-11-20, the sealed default for the geometric-preprocessing line
remains the full 3x3 local-occupancy V with its fixed 0.7 collision gate.
The learned raw-offset set V is recorded as an **alternative verifier**, not
a replacement. Seed 42 is the prespecified representative; seeds 43 and 44
are robustness replications, not test-selected winners. The new same-candidate
6x6 panel gives geometry 115/120 and each set seed 111/120. All four score
35/35 on collisions; the four-point gap is due to additional source-token
rejections by the set representation. See [the boundary report](../relational_boundary_v1/report.md).

## What is local

The following complete checkpoint directories are now in the ignored
`checkpoints/` folder. Each contains `completed.json` and a `latest/`
directory with `checkpoint.json`, `model.safetensors`, tokenizer assets,
config, and `training_state.pt`. The generator additionally has its
generation config. Every file was checked against the SHA256 map in its
checkpoint metadata; completed/checkpoint metadata and weight hashes are
anchored separately by [the download script](../../scripts/download_sealed_verifiers.ps1).

| Model | Local relative checkpoint directory | Weight SHA256 |
| --- | --- | --- |
| Set V seed 42 | `checkpoints/relational_set_v1/t5_base/verify` | `3eeefd73ebf8e40f831dc32e646b1769a37d5801b4003e69567d995fef203be8` |
| Set V seed 43 | `checkpoints/relational_set_multiseed_v1/seed_43/t5_base/verify` | `acb11d93f3e7a5651f04e121f91bd434a0521cd5c1e9e323129addfdd6e615b6` |
| Set V seed 44 | `checkpoints/relational_set_multiseed_v1/seed_44/t5_base/verify` | `fe8e720d3eb42252062710c015a7d92f823d748b5b60cdb1051b1773a5a17375` |
| Sealed geometry V | `checkpoints/local_occupancy_v1/t5_base/verify` | `23d96f5d072e76cab17c147539d2f6586ad6b65a56529f5f2b56023a47067b09` |
| Original structured V | `checkpoints/improvement_verify_v1/t5_base/verify` | `41aae273afb09ca96ca074a87bcfec43eac571a316103bea260b6fd62e990b3a` |
| Frozen v2 G | `checkpoints/v2/t5_base/generate` | `2e3ffab1370bb0f912944e0a9038b5e852be560be570cfa79711ba90d902ed9e` |

The geometry weight is **unchanged** from the earlier seal, not a new
training run. No local copy was present at the beginning of this archival
pass, so it was downloaded once; subsequent invocations verify and skip it.
The 0.7 gate is evaluation logic, not a second weight file.

The set-V training overlay, protocols, architecture and entry scripts, three
sets of frozen challenges, G outputs, per-case probabilities, audits, and GPU
ledger snapshots are local under `analysis/` and `results/`. The boundary
challenge, G output and predictions have SHA256 values in its report.
Local source SHA256 for the seed-42 entry, set head, protocol and paired
training overlay matches the checkpoint binding. The six checkpoints
validate their own completion fingerprints and training steps.
All 37 analysis/root Markdown documents were scanned for local links:
zero missing. The saved boundary experiment's independent audit passed;
remote regression was 154 passed.

This is a complete **local artifact set for this same-candidate V comparison**,
subject to installing the pinned Python/PyTorch/Transformers environment for
actual inference. It is not a claim that every historical project checkpoint
is local: paired/unpaired, late-collision retrain, representation-ablation,
repair models and other model-family checkpoints remain outside this backup.
Saved JSON can be reviewed without those weights; re-running their model
forward passes would require separate downloads or retraining.
Ignored checkpoint files are not included in Git or the public release; keep
an independent backup if the local drive may be lost.

## Repeatable download and verification

In PowerShell, from `D:\GridPathPlanning`:

```powershell
& .\scripts\download_sealed_verifiers.ps1 -Port 10048
```

This checks/downloads the three set V checkpoints. To also archive the
sealed geometry V and both direct comparison dependencies:

```powershell
& .\scripts\download_sealed_verifiers.ps1 -Port 10048 -IncludeGeometry -IncludeComparators
```

Use `-SshHost dd-a6000` only if the SSH alias differs, and change `-Port`
when the rental server port changes. Existing files with valid hashes are
skipped; a mismatched file stops the script rather than being overwritten.
Re-running the full command with all files already present performs a
local hash verification without transferring them again.
