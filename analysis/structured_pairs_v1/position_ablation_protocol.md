# Position-controlled positive ablation

This is a diagnostic follow-up to the already opened structured-pairs
holdout, not a new blind test. Keep the paired overlay's 1,000 row positions
and all 500 negative records exactly unchanged. Replace only the 40
same-environment shortest valid counterfactual rows, *at their existing
indices*, with the same 40 original optimal valid records restored in the
earlier unpaired ablation. Map the replacements by sorted original sample ID
to paired-positive index for deterministic ordering. Check each unchanged
slot by full row equality, the 40 replacement slots, all class counts and
negative identities before freezing the overlay.

Keep the original structured-offset representation, pretrained T5-base,
seed 42, batch 4 and 750 updates. Freeze source/overlay hashes and use a
new run directory. Score public development V80/G60, then optionally the
already-open holdout as a post-hoc comparison on identical candidates.
No threshold tuning, model selection on the opened holdout, changes to
the repairer, or claims of causal certainty from one training seed.
