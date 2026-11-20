# Learned raw-offset set V: two additional seeds

Keep the exact frozen paired V1000 overlay and set_model.py architecture,
text, labels, loss, optimizer and 750 updates. Train seeds 43 and 44
from pretrained T5-base in independent run directories. Bind each seed
to both initialization and epoch shuffle and include it in the checkpoint
fingerprint. Do not modify/retrain seed 42 or historical paired baselines.

Treat the first relational_set_v1 holdout as opened. For a new targeted
evaluation, exclude its 80 layouts in addition to all 2,320 prior
exclusions, freeze 20 controlled pairs and 60 one-shot G tasks using
seed 20261129, and freeze G raw outputs before any candidate V scoring.
Evaluate original structured V, the three matched-pairs text V seeds and
the three learned-set V seeds on identical paths. Report collision joint
step/type with exact denominators, valid retention, other error types,
interface failures, and paired by-seed deltas. No favorable-seed
selection, threshold tuning, changes to G/repairer, or replacing the
sealed geometric model from this targeted panel.
