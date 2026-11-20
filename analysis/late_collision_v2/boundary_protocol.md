# Adjacent collision boundary loss ablation

Start again from the original V1000 and pretrained T5-base. Keep the full
3x3 input, architecture, optimizer, seed 42, batch 4, 750 updates, and
original path loss. Add only a fixed 0.1 times softplus(1 + collision_logit
at the legal previous step - collision_logit at the true collision step)
for training paths where a supervised `continue` immediately precedes an
`obstacle_collision`. No global hard-prefix reweighting. This is a
single preselected loss setting, not a coefficient sweep.

Before training, freeze 40 new test-bucket layouts with the same four
candidate types as the earlier paired panels. Exclude every old panel,
both fresh representation panels and the balanced-training holdout.
Use the already viewed fresh dev panel plus original V80/G60 and
published ID/OOD V80 as guards. Qualify for the new holdout only if the
late collision joint count strictly improves over original full V and
early collision, both valid groups, V80/G60, and ID/OOD V80 do not
regress. Otherwise do not access holdout predictions. Keep all previous
checkpoints unchanged and report failures and denominators.
