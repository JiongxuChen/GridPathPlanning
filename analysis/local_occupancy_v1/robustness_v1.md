# Local-occupancy V: frozen robustness protocol

This protocol precedes robustness_v1 model inference. Keep the trained
checkpoint, representation, decoding and 2048-token source limit unchanged.
The prior development and fresh-layout challenge are exploratory evidence,
not untouched tests. All new layouts must be absent from train, dev, published
ID/OOD and the previous fresh challenge, and assigned to the `test` layout
bucket by seed 42. Construction seed: 20261116.

Six strata, 20 distinct layouts each:

1. `long_valid_6`: valid 6x6 path with five legal two-action loops prepended;
   at least 12 actions. This is a length shift within the original grid size.
2. `late_collision_6`: the same legal prefix followed by a single blocked
   return-block event on the reference path; first error at step 11 or later,
   while unchecked displacement still ends at goal.
3. `double_6`: one blocked return-block and one invalid-token insertion at
   distinct indices of the original reference path. Each event alone is
   invalid, their combination differs from both singles and the base, and
   the executor decides the first error. Alternate event order. The second
   error is generally not observed after execution stops: do not score it
   as a second diagnosed error.
4. `sparse_8`: 8x8, 2-6 obstacles, ten shortest valid paths and ten
   return-block collisions.
5. `dense_8`: 8x8, 12-18 obstacles, ten shortest valid paths and ten
   return-block collisions.
6. `size_10`: 10x10, 3-12 obstacles, ten shortest valid paths and ten
   return-block collisions.

Additionally, 12 separate 6x6 valid paths target lengths 16, 24, 32, 48,
64 and 72 (two layouts each). They probe the *unchanged* source-token
limit and action cap; model refusal is an interface failure, not a valid
prediction. No increase to token limit is permitted in this evaluation.
These challenges are targeted post-development OOD extensions, not the
original ID/OOD panel or a pre-project blind test. They do not certify
arbitrary grids, rare action strings, or probabilistic calibration.

Freeze the JSON challenge and its SHA256 before loading a model. Verify
executor truth, layout disjointness, exact quotas, single-event
counterfactuals, and endpoint properties on CPU. Evaluate both the original
structured V and local-occupancy V on the same candidates. Report by
stratum: denominator, valid false positives, invalid detection, first-error
localization, error type and joint diagnosis, plus interface failures and
input-token lengths. Never drop refused examples from denominators. An
incomplete or over-limit stratum is a finding, not a reason to silently
regenerate it. The previous 6x6 result remains a separate comparison.

Decision rule: retain local occupancy as the selected candidate only if
the 6x6 long/late/double strata show no material regression against the
original structured V, including false positives, and the shifted-size
strata have usable coverage. Any regression is reported before the
published ID/OOD historical comparison. No retuning on these results.
