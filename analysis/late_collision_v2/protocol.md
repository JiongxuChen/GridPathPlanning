# Late-collision representation confirmation

The prior full-versus-center-only comparison on published panels is
post-hoc diagnosis. Freeze a fresh, layout-disjoint 6x6 development panel
and a separate test panel **before** inference. Each panel has 40 layouts
with four paired candidates per layout: short valid, long valid, early
collision, and late collision. Both checkpoints are fixed: the original
750-update full 3x3 V and the independently trained 750-update center-only
V. The previously selected 0.7 collision gate applies only to full V;
do not refit it here.

Exclude project train/dev/ID/OOD, local robustness, previous late-collision
and calibration, representation attribution, sealed verifier, and natural
collision pool layouts. Audit all ground-truth labels with the executor
offline. Inference itself must not invoke the executor. Persist sample
identities, checkpoint fingerprints, model probabilities, failures, and
both data hashes. Evaluate dev before test, but do not change the three
specified routes or test panel after dev inspection.

Primary comparison: late-collision first-error/type joint correct, with
early collision and both valid groups as regression guards. Report each
group's full denominator and interface failures. Test is a new
confirmation of the previously explored representation hypothesis, not
the project's original blind ID/OOD test. The controlled four-candidate
construction is not a general natural-G claim. If center-only dominates
on the fresh test without regression, retain full V as archived baseline
and recommend center-only as a candidate; defer new data/loss training.
Otherwise investigate the failures before attempting further training.
