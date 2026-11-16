# Original local-occupancy V: post-exploration sealed evaluation

This is a new, one-shot 6x6 evaluation designed after earlier development,
robustness, late-collision and attribution results were seen. It is not the
project's original blind ID/OOD test. Freeze the exact challenge and its
SHA256 before loading the model; do not tune weights, representation, logits
or thresholds against this panel. Evaluate only the original full 3x3
local-occupancy T5-base checkpoint (750 updates), not the center-only
ablation, replacement-data models or the exploratory 0.7 gate.

Use seed 20261120 and 20 distinct layouts per group in the seed-42 test
bucket. Exclude all original train/dev/test layouts and every previously
constructed local-occupancy, late-collision and attribution challenge.
The eight groups are shortest valid, legal-loop valid, long valid (five
legal two-action loops), hard endpoint-at-goal obstacle collision, late
endpoint-at-goal obstacle collision (first error at step >= 11),
out-of-bounds, invalid token and goal-not-reached. The four error classes
are represented. The long and late groups test a specific constructed
path family, not arbitrary natural generation. All 160 rows count in their
original denominators, including source-limit and other interface failures.

Audit the executor's offline label and provenance before inference.
The executor must never enter neural V inference. Record full step
probabilities, tags, failure states, original truth, model and source
fingerprints, group metrics and aggregate metrics. Run once on the frozen
challenge; do not select a new checkpoint from the results. Report
classification, Macro-F1, first-error and reason joint diagnosis, valid
false positives, interface failures, late-collision position errors, and
per-group denominators. Keep the original 2048-token limit unchanged.
