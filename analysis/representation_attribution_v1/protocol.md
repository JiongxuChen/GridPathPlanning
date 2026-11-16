# Local-occupancy information attribution, preregistered protocol

Study the existing 6x6 verifier, not the postponed late-collision repair.
Keep frozen V1000, T5-base backbone, seed 42, batch 4, 750 updates,
loss, six-label head, and earliest-error aggregation identical.

Three occupancy representations:

- `full`: the existing local-occupancy checkpoint, all nine cells.
- `center_only`: retain CENTER free/blocked/outside; replace eight neighbor
  values with `redacted` while retaining all slots and the global header.
- `no_center`: retain eight neighbors; replace CENTER with `redacted`.

The latter two are trained from the same pretrained backbone, not evaluated
as out-of-distribution masks on the `full` checkpoint. All variants retain
unchecked coordinates, actions, global obstacles, END and goal offset.
Thus the comparison estimates the *marginal contribution of the explicit
center occupancy channel under this training recipe*, not an additive
percentage of neural versus preprocessing contribution.

Before training, freeze new, layout-disjoint dev and test panels in seed-42
layout buckets, excluding project data and all previously used challenges.
The dev panel has 10 shortest valid, 10 nonoptimal valid, 20 endpoint-at-goal
return-block collisions, 10 return-block out-of-bounds, 10 invalid token,
and 10 goal-not-reached examples. The test panel doubles these quotas
(140 total). Each example uses a distinct layout. All labels are generated
offline by the executor and independently audited; no executor is called
at V inference. Report classification, first-error/type joint diagnosis,
valid false positives, source failures and per-category matched differences.
Evaluate both new variants and the frozen `full` model on identical panels;
the original structured-offset V is a contextual reference, not a clean
fourth occupancy factor.

An additional deterministic reader of the rendered CENTER status, actions,
and END goal offset establishes whether the preprocessing supplies
sufficient information. It is an executor-like *symbolic probe*, not a
neural V competitor or project-compliant self-verification route.
Do not infer "X% of the gain is due to preprocessing" from accuracy ratios.
If the center-only model matches full, that supports sufficiency of the
explicit center channel under these conditions; if no-center succeeds,
it still has global obstacles and coordinates from which to learn.
