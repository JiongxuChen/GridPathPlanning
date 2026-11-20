# Ablation: stratified natural collisions without same-environment positives

The paired candidate already passed development and its fresh holdout has
been opened. This ablation is an exploratory development comparison, not a
new blind test. Keep the same 40 selected natural collisions, same 40
removed original hard collisions, same original 40 optimal valid paths
selected for removal, same training budget and original structured-offset
representation. Remove only the 40 newly constructed same-environment
shortest valid counterfactuals from the paired overlay, and reinstate the
40 original optimal valid rows. Thus both variants have identical
negative examples, class quotas, update count, training seed and backbone;
only the positive examples differ. Freeze and audit this unpaired overlay
before training. Train from pretrained T5-base for 750 updates and
evaluate original development V80 and natural G60. For diagnostics only,
the already-open paired holdout may be run afterward; never describe
that comparison as fresh/blind for this ablation.
