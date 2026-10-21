# Frozen data readiness and split isolation

| Experiment or evaluation | Status | Data and limitations |
|---|---|---|
| A existing-class fine-tuning | PASS executed | 1,500 frozen literal-pool TRAIN; 80 compatible DEV; all1,580 feature hashes/schema/targets/chronology/CTC pass |
| B local new classes | BLOCKED | Missing executed permission and qualified label records; old schema3.0; isolated prompts; no independent new-class validation |
| B RIT expansion | BLOCKED | Existing crop/time-origin mismatch; no verified conversion/split/independent validation |
| C retention/imbalance variant | BLOCKED by frozen gate | A gain1.0309pp<2pp; B has no admissible data |
| Known-signer recording-family transfer | PARTIAL | DEV80/485tokens, two known signers, five masters, two families; development exposed |
| Unseen-signer recognition | BLOCKED | No established independent-signer evaluation here |
| No-sign false activations | BLOCKED | No consented labeled no-sign minutes; zero DEV insertions is not a false-activation-per-minute result |
| Physical sign boundaries / repeated signs | PARTIAL/BLOCKED | Publisher interval diagnostics only; no adjacent repeats in DEV; no qualified whole-sign boundary review |
| Phone camera-to-text performance | BLOCKED this mission | No device modifications authorized; host model-only synthetic timings are separate |

The exact new copies of old TRAIN/DEV manifests and vocabulary are frozen in manifests/ and vocabulary.json. Their SHA256 values were written to A/TRAINING_CONFIGURATION.json before fitting:

```
TRAIN1500 19d4fd93c8c71264484013bc52943e1a8660ebbd9292b273b0399a41ab42a902
DEV80     72685c50277bec0bf4873d870369d10be14cbbb0310c7642ff888a0eaf2bbc8e
vocabulary ca0d9576b434bd032deb5d572768e4daf553faeb3e094b37d770c3a514da8b9f
```

The lists, not each row's inherited publisher/global `split` string, define the frozen experimental role. TRAIN has 12,423 target occurrences, 2,287 literal labels, three publisher signer IDs, 34 masters and six conservative recording families. DEV has 176 distinct reference labels, all known to TRAIN. It includes six sequences /42tokens in participant2's family and74 /443 in participant3's family. The 27 stronger-support subgroup remains descriptive and does not replace DEV80.

Guards verify no shared sample IDs, source master, raw SHA, decoded pixel SHA or conservative family between TRAIN and DEV. Families use publisher signer plus recording date, grouping all undated masters conservatively. These guards do not isolate causal signer effects: the same signers appear on both sides. Existing source-split policy and all prior six ASLLRP overlaps, one frame-count mismatch and two RIT overlap exclusions remain unchanged. All 3,121 reported eligible utterances and source identities remain preserved; only the authorized1,500 are fitted here. Counts of externally held examples are metadata reuse, not access to reserved TEST.

The reserved17 TEST set was not enumerated, opened, hashed, fitted, selected on or evaluated by this mission. Existing frozen held-out results were not changed. Production, Android, vocabulary packaging, renderer, APK, GLB and thresholds are snapshotted and unchanged. preservation_verify.log confirms221protected files and1,580feature files. The final read-only verifier checks these exact paths without walking any TEST location.
