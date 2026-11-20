# Balanced adjacent-collision training ablation

Keep the complete 3x3 representation, original T5-base backbone, seed 42,
750 updates, original six-tag head and path loss. From the original V1000,
replace the same 20 nonoptimal valid and 20 hard-collision rows as the
previous late-only experiment. Add 20 paired long valid rows and 20
collision rows on those training layouts: 10 early, 10 late. Preserve
V1000, 500/500 valid-invalid, all four error quotas, 200 nonoptimal valid,
and the original hard-collision quota. No evaluation layout enters train.

Freeze a new test-bucket 40-layout, four-candidate-per-layout panel before
training. Exclude all previously used layouts, including both fresh
representation panels. The already viewed fresh dev, fixed V80, natural
G60, and ID/OOD V80 are development guards. Advance the trained model to
the new holdout only if its dev late collision joint count strictly
improves over original full V and early collision, valid, natural G60,
and public ID/OOD V80 do not regress. Otherwise record the failed run,
do not touch the holdout, and retain the prior model.

The holdout is not the project's original blind test; construction is
informed by earlier results. Report all denominators, interface failures,
and every changed sample. No model selection against holdout.
