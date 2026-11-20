# Structured-offset V: matched natural collision pairs, experiment 1

Keep the original `improvement_verify_v1` T5-base step-tagger,
unchecked-displacement/obstacle-offset text representation, six-way head,
prefix/decision losses, pretrained initialization, seed 42, batch 4,
750 updates/3000 exposures, and fixed-final checkpoint. Do not use the
local-occupancy representation, `CENTER blocked` or the executor at inference.
Do not modify G, original data, models or historical results.

Use the previously frozen 1200-candidate train-layout G pool only for
training. Replace exactly 40 hard `obstacle_collision` V1000 examples and
40 optimal valid examples with 40 pairs. Each negative is a natural G
candidate: no generator failure, endpoint at goal by unchecked displacement,
first error collision, action count equal to the environment's shortest
distance. Its positive is the BFS legal shortest path on the *identical*
grid, start and goal, thus same length and endpoint. Each pair must have
different candidate tokens, distinct sample IDs and exact offline labels.
Use fixed first-error-step quotas 1:7, 2:6, 3:9, 4:7, 5:7, 6:2, 7:2;
deterministically rank by seed 20261124 and source sample ID. Remove
original hard collisions using the same per-step quotas and 40 original
optimal valid examples, deterministically ranked. Verify 1000 records,
500/500, each invalid reason 125, hard collision 83, valid nonoptimal 200,
unique sample IDs and train-layout/environment isolation.

Freeze the overlay and hashes before training. Evaluate only existing
development V80 and natural G60 first, paired against the unchanged
original structured V. Primary targets: natural G collision detection and
first-error/reason joint accuracy; guardrails: no valid false positives,
no regression in other three error classes, fixed V80 classification,
and no interface failures. Original development is public and is a
development panel, not a new blind test. Do not use historical ID/OOD
for model choice. If guardrails fail or natural collision joint accuracy
does not improve, stop this candidate and report it; otherwise freeze
new-layout candidates before any candidate-model evaluation.
