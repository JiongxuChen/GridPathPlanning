# Fresh-layout structured-offset V paired holdout

Development guardrails in `protocol.md` passed before this protocol was
written. The held-out panel remains a post-development, targeted evaluation,
not the project's original blind ID/OOD test.

Use seed 20261125; 20 distinct 6x6 seed-42 test-bucket layouts, each with
one shortest valid and one endpoint-at-goal `return_block` collision
candidate. Use another 60 distinct layouts for frozen v2 T5-base G
generation, with no selection based on candidate validity. Exclude all
original data layouts and all previously constructed local-occupancy,
natural-pool, late-collision, attribution, earlier sealed and gated sealed
layouts. Freeze an explicit exclusion manifest with source hashes, then
freeze the 80 environment rows and controlled candidates before G or V
inference. Run G once and freeze its raw 60 outputs with model fingerprint,
failure states and SHA256 before V. G interface failures retain their
pipeline denominator and are not submitted to V.

Evaluate the unchanged original structured-offset T5-base verifier and the
new matched-pairs structured-offset verifier on identical candidates,
without executor assistance at inference. Record raw six-class
probabilities, full labels, all interface failures and model/checkpoint
hashes. Report controlled and natural denominators separately, collision
detection and first-error/reason joint diagnosis, valid false positives,
classification and Macro-F1. No threshold search or model selection on
these held-out results. If the new model regresses, retain it as a failed
candidate and preserve the complete result.
