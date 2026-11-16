# Collision-confidence gate protocol

This is a post hoc exploratory calibration of the original frozen
`local_occupancy_v1` checkpoint, not another neural training run. Freeze
40 additional seed-42 development-bucket layouts before inference. Exclude
all train, project dev/test, prior fresh/robustness challenges, and both
`late_collision_v1` panels. Use four candidates per layout: short valid,
long valid, early collision, and late collision. Do not use the published
five errors or the reserved test panel to choose a threshold.

Only a predicted `obstacle_collision` action tag may be converted to
`continue` when its collision softmax probability is below a global
threshold. Leave all other tags and the earliest-error aggregator unchanged.
Candidate thresholds, in ascending order, are 0.5, 0.7, 0.9, 0.95, 0.99.
Choose the smallest threshold that strictly improves late-collision joint
diagnosis versus argmax, without reducing early-collision joint diagnosis
or either valid-group accuracy on this calibration panel. Otherwise reject
the gate. Check the original V80/G60 development predictions for regressions
as an additional guard. This protocol does not claim calibrated
probabilities, and any selected gate must be reported as a separate
inference variant.
