# AppHear continuation — PARTIAL; new training BLOCKED by evaluation readiness

Continued the frozen `20261020T004837_separate02` run. No training, repeated download, checkpoint selection, threshold change, held-out inference, dependency installation, Android operation or production change occurred. This new subdirectory contains additive evidence; earlier reports and manifests remain untouched.

## Source and archive: PASS

On 2026-10-20 Asia/Shanghai, opened the [official How2Sign download page](https://how2sign.github.io/#download) and clicked **Green Screen RGB clips* (frontal view), Validation (1.7G)**. Its actual destination is [val_rgb_front_clips.zip](https://drive.google.com/file/d/1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO/view?usp=sharing). Verified file ID: `1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO`. It matches the saved acquisition plan and receipt; no guessed ID was used.

Reused `../data/raw/how2sign/val_rgb_front_clips.zip`, **1,778,074,855 bytes**, below the unchanged **2,147,483,648-byte cap**. Fresh SHA-256: `6cb48170131fdb7fc64d40381a3c4be41112ba4454444170f5a31af2227b6d36`, matching the receipt. This is a locally observed hash, not a publisher checksum. Fresh checks passed all ZIP CRCs, 1,740 member paths, link/special-file rejection, duplicate names and the 3 GiB expanded-size bound. There are 1,739 video members. No extraction occurred in this continuation. The prior complete decoding of eight clips is reused; full video decoding of the archive is not claimed.

Preserved failed URL: `https://drive.google.com/uc?export=download&id=1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1`. The retained attempt records exact error **`HTTP_404`** at `2026-10-19T16:53:14.5183865Z`; see `../logs/download_how2sign_val_frontal_pilot_archive_attempt1.json`. This stale official-script destination differs from the verified current website link. The failure was not erased by the later successful acquisition.

The official frontal 2D-keypoint Validation link also resolves to [val_2D_keypoints.tar.gz](https://drive.google.com/file/d/1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w/view?usp=sharing). Its payload was not acquired or tested because RGB succeeded. Link resolution alone proves neither payload availability nor schema compatibility. Coordinate representation, confidence fields, cadence, ordering and labels in that archive remain **UNVERIFIED**; it is not training eligible. No fallback download is necessary now. Future remote availability is not guaranteed by a saved successful archive.

## Labels, features and data readiness

The acquired How2Sign original validation table has 1,741 rows and columns `VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME, START, END, SENTENCE`. These are English translations and sentence timing. Two annotation names lack videos. No authentic ASL gloss sequence or individual-sign timing was verified in the inspected release. Official validation remains validation; English was not converted into gloss or used to train a translation model.

The existing eight-clip pose probe has 273 observations, label type `ENGLISH_TRANSLATION_NOT_GLOSS`, and schema `apphear.continuous_asl.how2sign_pose_probe.v1`. It uses causal 378-wide segmentation features derived from Candidate01 raw observations, sampled at minimum 100 ms spacing (about 8 Hz for these 24 FPS clips). It is neither a 350-wide recognition-v2 tensor nor full-rate streaming evaluation data. No tensor padding, renaming or English-label substitution makes it compatible.

| Experiment/data | Labels actually available | Signer-independent split | Genuinely trainable now? | Gate / missing requirement |
|---|---|---|---|---|
| Legacy isolated 236-class ASL Citizen package | Isolated gloss class; schema3.0 `[96,258]`, masks, lengths, signer IDs | Verified 35 TRAIN / 5 VAL / 11 frozen TEST signers, no overlap | **Yes, already completed**; do not retrain | **PASS** for the completed isolated benchmark; current streaming transfer remains unverified |
| NCSLGR research subset, 37 clips | Authentic ordered ASL gloss sequences; no per-sign boundaries in the eligible index | 30/7 split by source collection; signer IDs unknown | **Yes for the completed diagnostic CTC fit only** | **PARTIAL**; need verified signer identities/splits, larger references and boundary/no-sign truth before a new generalization claim |
| How2Sign gloss CTC | RGB plus English sentence translations/timing | Official VAL only; inspected table lacks signer identity | **No** | **BLOCKED**: authentic gloss targets, training partition and verified signer protocol absent |
| How2Sign English translation | English sentence targets, explicitly not gloss | Acquired RGB is official VAL only | **No with current payload/pipeline** | **BLOCKED**: permitted training partition, translation baseline/model and independent evaluation protocol not validated |
| Current WORD streaming recognition-v2 / segmentation | Contracts exist: recognition `apphear.perception_vnext.recognition.v2`, width350; segmentation width378 | No verified fresh streaming signer split available here | **No new supported experiment** | **BLOCKED**: synchronized compatible observations, expert gloss/timing, rights and independent signer roles |
| Consented no-sign / rejection evaluation | Historical audit documents; no verified usable fresh consented timed corpus in checked locations | Historical P1 aliases overlap; E6 single-signer evidence was development exposed | **No** | **BLOCKED**: raw sources, consent/scope, reviewed non-sign intervals, exposure minutes, signer roles |
| Supplementary local Recorder clips | 4,035 manifest rows and all 4,035 referenced videos exist; clip-level ASL labels and signer IDs | No new signer-disjoint streaming assignment verified | Only historical isolated use is documented; **not a verified new streaming experiment** | **PARTIAL**: permission/consent scope, exposure lineage, current feature extraction and natural sequence/boundary labels |
| Phone transition/compute fixtures | Nonexpert design-cohort boundary labels, recorded states/actions, engineering summaries | Not independent qualified-signer evaluation | **No accuracy training/evaluation corpus** | **PARTIAL** engineering evidence; original synchronized session bundles absent here |
| Official frontal 2D-keypoint fallback | Archive title/link only; payload schema/labels unknown | Not inspected | **No** | **BLOCKED** pending actual schema/labels/permissions checks; acquisition unnecessary while RGB is valid |

Evidence: `evidence/data_inventory.json`, the existing dataset audits and licensing reports. NCSLGR authentic gloss supervision is retained; the absence of How2Sign glosses does not invalidate it. ASL Citizen and NCSLGR remain bounded private research sources under the previously reviewed terms; distribution/deployment clearance is not inferred.

Metadata-only inspection of `D:/derived/derived/apphear3/features_streaming_conversational_fps/train.npz` confirms schema3.0 `[13236,96,258]`; `features_streaming_recorded_local/recorded.npz` is schema3.0 `[1076,96,258]`. Directory names containing “streaming” do not establish continuous sequence supervision. Canonical signs named `NEGATIVE`, `PERMISSION` or `LICENSE` are ordinary glosses, not negative examples or consent records.

The checked current and duplicate transfer copies lack the referenced P1 reviewed-interval CSV, signer-governance audit, S2-R3 negative raw directory, original phone observation sessions and E6/N0/P1 raw directories. Exact paths and existence checks are retained. Historical documentation describes 197 reused P1 windows, disputed zero-visible-hand labels, and 560 E6 short negative/OOV clips from one signer; later exposure records prevent treating these as pristine new evaluation data. Hand absence, hands-visible non-sign behavior, and out-of-vocabulary signing require distinct labels. No data were fabricated or relabeled from filenames, CTC blanks or model confidence. Unrelated recordings were not treated as consented data. This is a scoped inventory, not a claim that no such data exists on any disk.

## Frozen experiment status: preserved

Three isolated runs of five epochs and one continuous run of eleven epochs remain complete. On the existing frozen 2,620-row/11-signer isolated test, unchanged FP32 baseline is **61.22% Top-1 / 60.22% macro-F1**, primary **62.56% / 61.69%**, repeat **62.18% / 61.45%**. These legacy258 research candidates are not selected for production and do not establish live-app improvement. Existing export parity/latency results were reused without rerunning inference.

The NCSLGR native-cadence candidate remains unselected: diagnostic WER **92%**, worse than **88%** from the verified earlier pilot. Neither is current-app cross-signer WER. No selection or optimization used frozen test results. GPU is still an RTX3050 Laptop, 4,096 MiB; no training process was found and the GPU snapshot was idle. Git reports `fatal: not a git repository`; branch/commit identity cannot be asserted from this copy. CIM process inspection was unavailable; `Get-Process` supplied the fallback inventory.

## Proposed measurable evaluation gates — not yet passed

These are **proposed research acceptance criteria**, to freeze before collecting/scoring the independent qualification set. They do not change existing model thresholds or establish production approval. The existing repository requirement of sustained completed processing **at least 15 FPS** remains fixed. See `acceptance_protocol.md` for denominators, matching, clocks, confidence intervals and collection size limits.

| Metric | Proposed criterion | Current status |
|---|---|---|
| Continuous end-to-end gloss WER | At least 5 absolute percentage points better than the unchanged compatible baseline on the same held-out sessions; paired signer-bootstrap 95% CI for candidate-minus-baseline WER entirely below zero | **BLOCKED**: no qualified independent timed streaming reference corpus |
| False activations | At frozen operating point, upper one-sided 95% Poisson exposure bound at most 0.1 false commits/min of reviewed negative time; report signer/category counts and model assumptions | **BLOCKED**: no eligible reviewed exposure denominator |
| Missed signs | No matched committed event for at most 10% of scorable reference sign events, with one-sided 95% upper bound at most 10%; separately report wrong-class substitutions, correct-but-dropped and duplicates | **BLOCKED**: missing qualified event/boundary truth |
| Boundary behavior | One-to-one event F1 at IoU≥0.5 at least0.90; p95 absolute onset and offset error each≤200ms; duplicate commits≤1% | **BLOCKED**: insufficient independent boundary annotations |
| End-to-end latency | p95 reference-sign-end→normal-app committed delivery≤1,000ms, plus sustained completed processing≥15FPS over a5-minute session on the declared target device | **BLOCKED**: no eligible normal-app synchronized capture; model-only laptop times cannot pass |

## Reused test evidence and unresolved failures

Focused suite remains **65 passed**. Existing broader protected-copy XML contains **810 passed, 130 failed, 54 errors, 10 skipped**. No full suite or known failing test was rerun. The new per-entry traceback classification covers all184 failed/error entries: 25 missing MediaPipe imports; 133 missing-file tracebacks; 2 NumPy `trapezoid` API errors; 1 explicit research-sandbox subprocess denial; 1 temporary-path `FileExistsError`; 22 remaining assertion/validation failures. Twelve of the missing-file tracebacks point into the generated temporary-fixture directory; the other121 require individual path/fixture investigation. A missing temporary path is not automatically a missing historical dataset or an established Windows-only defect.

The classification is evidence triage, not proof that all failures predate the experiments. The NumPy API incompatibility is explicit in the saved tracebacks. No package was upgraded to conceal it. Missing governed fixtures should be restored by matching manifests in an isolated test copy; remaining assertions still need targeted diagnosis. **Broad regression gate: PARTIAL / NOT GREEN.** Details and original XML hash are in `evidence/regression_classification.json` and `evidence/regression_triage.json`.

## Preservation, changes and next action

Fresh additive preservation verification passed **3,338 protected records**, all **2,932 original timestamps**, and **336 frozen artifact hashes**, with zero mismatches. This includes prior completed artifacts and original production models/assets. The old final manifest SHA-256 remains `1081281f0a6d28426d4804d202300381549d4374d25a7bb204050671f8a2d8b4`. No held-out feature payload was reopened for inference. The checks read saved artifacts; no completed verifier was invoked in a mode that overwrites prior results.

All new/changed deliverables are confined to this continuation directory. `artifact_hashes.json` enumerates their SHA-256 values and excludes itself to avoid recursive hashing. `reproduction_commands.ps1` verifies this new evidence without training, network access or repeated full old checks; its optional archive audit requires a fresh output directory. Original `../reproduction_commands.ps1` was read but not executed; its Acquire/Resume phases are unnecessary for this continuation.

**Highest-value next action:** collect the small explicitly consented, qualified-signer streaming evaluation pilot in `collection_plan.md`, beginning with one signer/session to validate consent, schema and annotation joins, then a second disjoint signer. Do not recommend or launch another training run until a compatible baseline replay, label review and independent split/rights checks pass. The pilot establishes data readiness, not deployment qualification; a larger signer-disjoint collection is required before applying the qualification gates.

Overall **PARTIAL**: source/archive and preservation **PASS**; completed training evidence preserved; new streaming/rejection/How2Sign-gloss training **BLOCKED**. Exact safe commands: `reproduction_commands.ps1 -Phase Verify`, `-Phase Matrix`, and `-Phase CollectionPlan`. There is no justified next training command.
