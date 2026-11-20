# Paired multiseed verification, structured-offset V

Before scoring any verifier on the new panel, freeze an explicit union of
the prior 2,160 excluded layouts and the 80 layouts of the first
structured-pairs holdout. Use a new deterministic generator seed 20261127
and the same 20 controlled valid/collision pairs plus 60 one-shot v2 G
tasks as the previous holdout. Freeze all rows and G outputs, including
failures, before any V inference. Do not select candidates by their truth.

Run two additional matched seeds, 43 and 44, for both the original paired
overlay and the position-controlled unpaired overlay. Within each seed,
use identical T5-base initialization, 1,000 row positions, 500 negative
examples, stream shuffle, optimizer, 750 updates and fixed-final
checkpoint. The only differing input is the 40 positive records at
identical indices. The prior seed-42 models are not retrained. Record the
actual seed in the checkpoint identity. Do not modify the project's
original training module or original checkpoint identities.

After all four new runs finish, score the six seed/arm models on exactly
the same new controlled and frozen natural G candidates; include the
original structured V as contextual reference. Primary metric is the
first-error-step plus error-type joint accuracy on collision examples,
reported as exact numerators/denominators by panel and by seed. Report
valid-path retention, classification, other errors, interface failures,
paired differences and all seeds, not just a favorable aggregate.
The previously opened holdout and public development results remain
diagnostics, not model-selection data. The new panel is a targeted fresh
layout evaluation, not the original project ID/OOD blind test. Do not
change G, repairer, thresholds or geometry-gated sealed model.
