# Decision-only focused ablation

This is an additional, exploratory T5-base experiment. In a separate copy of
`GridPathPlanning-run`, replace only `src/gridpath/planning/model.py` with
`model.py` here. The regular structured verifier uses equal contributions from
the legal-prefix mean and the class-weighted decisive position on paths with
both; this variant removes the prefix term while retaining the original
decisive-position coefficient. For a path without a prefix, both losses are
identical. Architecture, input representation, data, optimizer, seed, stream,
training duration and inference protocol remain unchanged.

Do not replace the file in the original reproduction: the source fingerprint
would invalidate its completed verifier. Preserve the original GPU budget
history in the isolated copy, and move copied improvement checkpoints,
results, development selection and test lock out of their active paths before
running the same `improvement_verify_v1` entry point. The v2 generator remains
valid because its fingerprint does not include `planning/model.py`.

Train and evaluate **dev only** in the isolated copy. The existing test sets
have already been inspected, so this exploratory ablation makes no new
held-out-test claim. A single seed gives descriptive, not statistical,
component attribution. GPU wall time is included in the cumulative budget.
