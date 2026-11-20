# Full local-occupancy V with fixed 0.7 collision gate

This is a post-development, one-shot 6x6 evaluation. It is not the original
project blind ID/OOD test. The model is the original full 3x3
`local_occupancy_v1` T5-base latest checkpoint at 750 updates. No retraining,
checkpoint selection, representation change or threshold search is permitted.
For each action whose original argmax tag is `obstacle_collision`, change that
tag to `continue` iff its collision softmax probability is strictly below
0.7; leave every other action and END tag unchanged, then aggregate the
earliest predicted error. This threshold was selected on the earlier
independent development calibration, before this challenge.

Before loading V, freeze seed 20261123, the exact challenge and its SHA256.
Use 20 distinct seed-42 test-bucket layouts per group, excluding all project
train/dev/test and previous local-occupancy, natural-pool, late-collision,
attribution and sealed-evaluation layouts. Controlled groups: shortest valid,
long valid, hard collision, late collision (first error >= 11), out of bounds,
invalid token, goal not reached, two independent errors (different base-plan
indices; each independently invalid; the combined candidate differs from
both singles), and long legal path at the existing source-token boundary.
Also freeze 20 new GENERATE layouts. Generate their candidates once with
the frozen v2 T5-base G; save raw output, interface failure, checkpoint
identity and SHA256 before any V inference. Do not select or filter G outputs
by their labels. G failures remain in the natural panel denominator as
pipeline interface failures; they never become V calls.

The executor supplies offline labels only, not V predictions. Evaluate the
same frozen controlled and generated candidates with raw argmax and the gate
without another V forward pass; store full step probabilities, tags, source
length, failures, model identities, GPU ledger and group denominators. Count
all V source-limit failures in their original denominators. Report
classification, Macro-F1, first-error and reason joint diagnosis, valid
false positives, collision position deltas and limits of coverage. The
generated panel is natural candidate distribution, not a collision quota;
report its realized collision count. Never use results to tune the gate.
