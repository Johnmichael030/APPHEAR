# Baseline-preserving bounded adaptation

Predeclared after measured TRAIN/DEV regression diagnosis in expand06, before
this experiment fits any data. This is a research experiment, with no production
promotion or changes to Android, assets, thresholds or frozen tests.

Use all455 verified prior TRAIN utterances plus the next complete verified
NCSLGR timed preparation batch. Freeze that manifest before smoke or fitting;
do not wait for the independently acquired ASLLRP release. If that additional
release validates later, it requires another versioned manifest and experiment.
Keep the exact original9DEV identities, literal labels, features, greedy CTC
decoder and WER-then-loss selection. The reserved17TEST remains untouched.

Keep the same48-channel causal TCN (RF61) and all378 features. Warm-start from
the frozen expand03 reference checkpoint, keep its TRAIN100 mean/std, copy
output rows by exact token identity, append new TRAIN labels with zero weight
and bias-10, and reset AdamW at learning rate0.0001. No label normalization,
beam search, blank penalty or threshold tuning is introduced. Before fitting,
verify old-logit parity and identical greedy predictions on fixed9DEV.

Run one-epoch four-TRAIN/two-DEV smoke, then a maximum20-epoch/180-second
full-data run, patience5, batch4/accumulate2, FP32, clip5, seed20261020.
Select by the unchanged9DEV WER then CTC loss, allowing the initialized
epoch0 checkpoint to win. A selected epoch0 means adaptation did not improve
selection; actual fitted epochs and consumed utterances must still be reported.

This tests whether preserving the baseline representation and normalization
avoids the measured cold-start/deletion regression. It does not establish a
unique regression cause or guarantee improvement. TRAIN recognition is poor in
all previous models, so a simple overfitting explanation is unsupported.

Historical signer identities remain incomplete. This is a collection-disjoint
diagnostic; it cannot establish signer-independent recognition or deployment
readiness. No extra participant consent or ground truth is invented.
