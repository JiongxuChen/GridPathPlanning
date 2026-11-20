# Step-marker occupancy alignment ablation

Train a new T5-base verifier from the original backbone and V1000 for
750 updates, seed 42, original six-label head and original path loss.
Keep the complete nine local occupancy values, coordinates, global
header, end offset and labels. Move the existing CENTER value to
immediately after each step marker; list the other eight neighbors in
their established order. This changes proximity, not available geometry.

Freeze a new 40-layout test-bucket, four-candidate-per-layout challenge
before training. Exclude all project and previous research layouts,
including the boundary-loss holdout. Use the already seen fresh dev
panel, fixed V80, natural G60, and public ID/OOD V80 as development
guards. Advance to the holdout only if all dev guards remain at least as
good as original full V and late collisions strictly improve over its
35/40. This ablation cannot be preferred over the boundary-loss model
solely for tying its 40/40 on small controlled panels; compare breadth,
input length, and robustness without retuning on test.
