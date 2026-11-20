# Frozen same-candidate verifier boundary comparison

This is a new 6x6, seed-42 test-bucket layout evaluation, not the original
project ID/OOD blind test. Freeze the challenge before any verifier inference.
Exclude the first 2,400 layouts in the relational-set exclusion manifest,
the second relational-set challenge (80 layouts), and the sealed-gate challenge
(already contained in the first 2,400; union 2,480 layouts). Use seed 20261130,
ten distinct layouts each for shortest valid,
loop-extended valid, hard collision with unchecked goal endpoint, collision
after a legal loop (first error at least step 11), out of bounds, invalid
action token, goal not reached, and long legal source-token-boundary plans.
Sample obstacle counts 2-5 in the first half and 6-9 in the second half
of every controlled stratum. Freeze 40 more natural GENERATE layouts without
filtering on the eventual generator label.

Call the frozen v2 T5-base G once per natural layout, record raw output and
interface failures before scoring any V. The executor computes offline truth;
it must not be used to supply a V input label or fix any prediction. Score
the same controlled and G-generated candidates with the original structured
V, the full 3x3 local-occupancy V with its sealed 0.7 gate, and relational-set
V seeds 42, 43, 44. The gate only changes action collision argmax tags to
continue when the collision softmax probability is strictly below 0.7.
Never tune it or select a seed on this panel.

Keep candidate IDs, trace labels, per-step probabilities, weights SHA256,
source lengths and failure codes. Include G failures and V source-limit
failures in original panel denominators and report them separately from
incorrect predictions. Compare exact full labels, collision step-and-reason,
valid-path false positives, other error types, length and obstacle-density
strata. Existing opened panels remain historical diagnostics, not blind tests.
