# AppHear separate ASL experiments — PARTIAL

**Actual training executed on the RTX3050: three separate isolated-sign runs (5epochs each) and one new continuous-gloss run (11epochs). No production model, threshold, flag, Android behavior, app data or avatar asset changed.**

The matched isolated FP32 baseline scores65.13% top1 /64.12% macro-F1. Primary warm-start scores67.10% /66.57%; repeat66.54% /65.78%. These are modest repeated validation gains on3,264 feature rows from5 held-out signer IDs, validation evidence, with the later frozen final-test results reported below; no live-app claim. The initial scratch model scored19.45% /15.49% and was not selected. Current production RecognitionV2 accuracy remains NOT MEASURED on these incompatible legacy258 features.

Continuous baseline WER reproduces88% (6S+16D+0I /25 refs,9hyp tokens), while the native-cadence candidate scores92%. It stays unselected. Recording collection separation does not prove signer independence. Negative/boundary truth remains unavailable; no false-activation/rejection improvement is claimed. CTC blank/UNK is not calibrated rejection.

How2Sign newly acquired:1,778,074,855-byte official validation clip archive and311,131-byte original English table. ZIP CRC/path checks passed;1,739 archive clips pair to1,741 annotation rows with2missing videos. Eight clips (5,046,739bytes,811frames,720p24FPS) were fully decoded/timing checked. Existing re-aligned English TRAIN/VAL annotations were verified/reused. Eight-clip pose features (273observations) were preserved. Authentic gloss labels were NOT AVAILABLE in the inspected public release, so no How2Sign gloss training occurred. The official Bash script had a stale404 identifier; the current website link worked after a public-confirmation detector correction. No access controls were bypassed. No TEST or full RGB training dataset was acquired, and no translation model was trained on official validation.

Primary new ONNX700,036bytes: max probability error2.09e-6, zero class changes across3264validation rows. CPU ORT p50/p95 0.463/0.784ms, model-only. Landmark extraction remains roughly47–68ms depending on source/cadence; no Android FPS claim. Actual worker memory and VRAM are recorded, with primary warm-start peak working set 2,221,531,136bytes and allocated VRAM 29,080,576bytes. No simultaneous GPU training jobs were launched.

Focused suite:65passed, including9new model/contract/checkpoint checks. Public confirmation tests pass. Full protected-copy suite: 810passed, 130failed, 54errors, 10skipped across 1004JUnit entries. Missing MediaPipe, missing historical files, Windows snapshot paths/API assumptions and some unclassified failures prevent a green regression gate. The prior missing phase7 Keras artifact remains absent; no substitute was fabricated. All2,932 original files retain hashes and timestamps; all401 completed prior artifacts plus their manifest remain preserved. New sources and external training input hashes are recorded. Git metadata is absent.



Final native-test evaluation was added after training/checkpoint/threshold decisions were frozen. On 2620 rows from 11 test signer IDs, disjoint from TRAIN/VAL: frozen FP32 top1 61.22% / macro-F1 60.22%; primary 62.56% / 61.69%; repeat 62.18% / 61.45%. Primary gains 1.34 top1 points (35 additional correct rows), repeat 0.95 points (25 additional correct rows). No post-test training or selection occurred. This is a legacy258 isolated-sign benchmark; current live-app/negative/boundary behavior remains unmeasured. See final_test_report.md and final_test_freeze.json.

Acceptance statuses:

| Gate | Status |
|---|---|
| GPU environment and training execution | PASS |
| How2Sign inventory and download verification | PASS_BOUNDED_PILOT; two annotation/video gaps documented |
| How2Sign gloss annotation availability | NOT AVAILABLE_IN_INSPECTED_PUBLIC_RELEASE |
| Dataset eligibility and permissions | PARTIAL_PRIVATE_RESEARCH_ONLY |
| Isolated-sign experiment | PARTIAL_REPEATED_VALIDATION_GAIN; new fingerspelling blocked |
| Continuous-sign experiment | PARTIAL_TRAINED_NO_GAIN |
| Accuracy improvement against matched legacy validation baseline | PASS |
| Accuracy improvement against current production live path | NOT MEASURED |
| False-activation improvement | NOT MEASURED |
| Long-sequence and boundary behavior | PARTIAL_ENGINEERING_CHECKS; natural boundary evaluation BLOCKED |
| Speed and memory | PARTIAL_LAPTOP_MEASURED_NO_ANDROID |
| Export parity | PASS_PRIMARY_NEW_EXPORT |
| Regression suite | FAIL_ENVIRONMENT_FIXTURE_AND_UNCLASSIFIED_SNAPSHOT_FAILURES |
| Production-asset preservation | PASS |
| Production integration | NOT INTEGRATED |


Exact paths:

- New run: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02`
- Tooling: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/tools/asl_research_v2`
- Primary checkpoint: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/experiment_A/warmstart/checkpoints/epoch_004.pt`
- Repeat checkpoint: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/experiment_A/warmstart_repeat/checkpoints/epoch_003.pt`
- Continuous checkpoint: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/experiment_B/checkpoints/epoch_005.pt`
- Training/resource/evaluation logs: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/logs`, with per-candidate training_log.csv and validation_best.json.
- Safe continuation commands: `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02/reproduction_commands.ps1`; executed command history and fresh-run cautions: reproduction_notes.md.

Remaining blockers: current350-feature production benchmark/transfer; real non-sign/incomplete-sign labels and rejection operating curves; reliable timed gloss boundaries and NCSLGR signer identities; new temporal fingerspelling data eligibility; full raw-feature lineage; current-path independent final evaluation and more signer clusters; full regression dependency/fixture restoration; deployment/distribution rights and representative Android/device/delegate tests. There is no outstanding approval to integrate or install an APK in this mission.

**Safest next action:** preserve these candidates unselected; obtain compatible current-path signer-disjoint streaming and consented negative/boundary data. The isolated gain also persisted on the subsequently frozen native test; a compatible current-path streaming benchmark is still required before any adapter/integration. Further larger-model or cadence sweeps on25 gloss references are not justified. Every required deliverable exists in this new run, with blocked work stated explicitly.

Artifact integrity: artifact_manifest.final.json is authoritative after final evaluation. artifact_manifest.json and logs/pre_final_test_report_snapshot retain pre-test evidence. Final-test results were never used to choose a different seed/checkpoint.
