# Learnable obstacle-offset set head, isolated verifier experiment

Hold the paired structured V1000 overlay, T5-base backbone, exact original
structured text (including all obstacle-minus-proposed offsets), six
step labels, loss, seed 42, batch 4, 750 updates, and fixed-final
checkpoint constant. Add a second, numerical view of the same per-action
raw (dr, dc) pairs. It must include every obstacle exactly once, not sort
by distance or provide blocked/valid flags. A shared learned pair encoder
and query attention aggregate the permutation-invariant set at each step;
combine with the T5 marker hidden state in a trainable head. END has an
empty offset set. No executor or label lookup during inference.

Freeze the code, overlay SHA256, backbone identity, checkpoint directory
and development guardrails before training. Score the already-public fixed
V80 and natural G60. Continue to a separately frozen new layout panel
only if natural collision joint diagnosis improves over original
structured V's 3/17, all 42 natural valid candidates stay valid, other
three error types do not regress, and interface failures are zero.
The matched-pairs seed-42 V is a same-data comparator, not a selection
target. The newly frozen panel must exclude all earlier train, dev,
ID/OOD and exploratory layouts, freeze one-shot G outputs before V, and
compare the original, matched-pairs and set-head models on identical
candidates. Report classification and first-error/type joint separately.
No threshold search, repairer change, geometry-rule preprocessing,
retraining of historical models, or claim that a single seed is stable.
