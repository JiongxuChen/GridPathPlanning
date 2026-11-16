# Late-collision distribution experiment

This is a focused successor to `local_occupancy_v1`, not a replacement for
the frozen project dataset. Keep the T5-base backbone, local 3x3 occupancy
representation, step head, loss, seed 42, batch 4, 750 updates, and third-pass
checkpoint unchanged. Replace exactly 20 nonoptimal valid and 20 hard
obstacle-collision records in V1000. The new 20 valid/collision pairs share
training layouts and a legal ten-action prefix; collision is a single
return-block event after that prefix. Preserve V1000 class counts, the
500/500 split, 200 nonoptimal valid paths, and the collision hard-negative
quota. G3000 and all original frozen files remain untouched.

Before training, freeze two disjoint new-layout panels, each with 20 layouts
and four candidates per layout: shortest valid, long valid, long candidate
with an early collision, and long candidate with a collision at step 11-17.
Development layouts belong to the seed-42 `dev` bucket, held-out layouts to
`test`; neither overlaps any existing project or robustness layout. Both
panels are constructed without model predictions. The development panel
selects whether to retain this candidate; score the held-out panel once
after that decision. The published late-collision 20 are only a historical
reference, never a threshold or model-selection set.

Primary endpoints: late-collision joint first-error/type diagnosis, long
valid false-positive count, early-collision joint diagnosis, and interface
failures. Also report original fixed V80 and natural G60 development
results. Do not infer that longer length, turning, or nearby obstacles are
causal from the old five errors alone. If distribution repair fails, run a
separate loss-weight ablation. Calibrate any probability gate only on
another independent development split and keep it distinct from pure neural
argmax verification.
