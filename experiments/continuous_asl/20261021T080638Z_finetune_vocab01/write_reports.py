"""Assemble reports from completed immutable measurement files. No inference or fitting."""
from pathlib import Path
from collections import Counter
import csv,json,platform,statistics
import fine_tune as ft

R=ft.RUN; a=ft.read(R/'A/RESULTS.json');stats=ft.read(R/'POST_EVALUATION.json');u=ft.read(R/'PAIRED_UNCERTAINTY.json');inv=ft.read(R/'DATA_INVENTORY.json');timing=ft.read(R/'TEMPORAL_DIAGNOSTICS.json')
base=ft.read(R/'baseline_greedy_DEV80.json');candidate=ft.read(R/'A/selected_DEV80.json');sb=ft.read(R/'baseline_streaming_DEV80.json');sc=ft.read(R/'A/selected_streaming_DEV80.json');tr=ft.read(R/'A/selected_TRAIN1500.json');cfg=ft.read(R/'A/TRAINING_CONFIGURATION.json');hist=ft.read(R/'A/metrics_history.json');v=ft.read(R/'vocabulary.json')['tokens']
def report(name,text):
 p=R/name
 with p.open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')
pp=lambda x:f'{100*x:.4f}%'

report('MISSION_REPORT.md',f'''
# Continuous ASL fine tuning and vocabulary expansion

Date: 2026-10-21. Overall **PARTIAL**. The original model was successfully fine-tuned, but the candidate does not pass the recognition or promotion gates. No new vocabulary class was admitted or learned. Production and the reserved TEST set remain unchanged.

| Protocol on identical development records | Frozen epoch 12 | Selected A epoch 2 |
|---|---:|---:|
| Full-sequence greedy WER | {pp(base['WER'])} | {pp(candidate['WER'])} |
| Substitutions / deletions / insertions | 136 / 328 / 0 | 132 / 327 / 0 |
| Exact sequences | 0/80 | 0/80 |
| Mean sequence CTC loss | {base['CTC_loss']:.6f} | {candidate['CTC_loss']:.6f} |
| Argmax blank fraction | {pp(base['blank_fraction'])} | {pp(candidate['blank_fraction'])} |
| Current streaming replay WER | {pp(sb['WER'])} | {pp(sc['WER'])} |
| Streaming substitutions / deletions / insertions | 176 / 288 / 0 | 185 / 279 / 0 |
| Full TRAIN greedy WER | 88.4005% (reused) | {pp(tr['WER'])} |

The full-sequence candidate gains five correct aligned tokens, from 21 to 26 of 485. Nine utterances improve, four worsen, and 67 have the same edit count. The paired utterance-bootstrap descriptive interval for candidate-minus-baseline WER is [{100*u['descriptive_percentile95_interval'][0]:.3f}, {100*u['descriptive_percentile95_interval'][1]:.3f}] percentage points. It includes zero and ignores correlations among utterances in only two recording families. This is development-exposed, known-signer evidence, not an independent test or unseen-signer claim. The unchanged aggregate streaming score conceals changed predictions and a substitution/deletion tradeoff.

## Actual execution

A restored epoch-12 model weights and AdamW moments, changed LR from 0.001 to 0.0001, and kept architecture, normalization, vocabulary, frozen 1,500 TRAIN / 80 DEV split, features and greedy metric fixed. Batch 4, accumulation 2, FP32, seed 20261020. A disposable one-update smoke was saved and then discarded by reloading the source exactly. The main fit completed 10 epochs / 1,880 optimizer updates and stopped under its predeclared lack-of-gain rule, within the 12-epoch / 180-second bound.

Measured training plus epoch-evaluation wall time: {a['wall_training_and_epoch_eval_seconds']:.3f} s. Sum of fit-loop time: {a['fit_seconds']:.3f} s. Epoch fit times: {min(r['fit_seconds'] for r in hist):.3f}–{max(r['fit_seconds'] for r in hist):.3f} s. Selected-checkpoint full TRAIN and replay scoring occurred afterward and is excluded from those training times. Peak PyTorch allocated VRAM: {a['peak_allocated_bytes']/2**30:.3f} GiB; reserved: {a['peak_reserved_bytes']/2**30:.3f} GiB. Sampled device-reported peak: {stats['sampled_nvidia_smi_device_peak_MiB']:.0f} MiB. WDDM sampling and allocator measurements are separate quantities. See A/METRICS.csv for measured frame throughput and A/gpu_memory.csv for samples.

Selected research checkpoint: `A/checkpoints/epoch_002.pt`, SHA256 `{a['selected_checkpoint']['sha256']}`. It is source epoch 12 plus two fine-tuning epochs, not the final epoch-10 checkpoint. Checkpoints include optimizer, RNG and configuration identity. The A driver supports last-completed-epoch recovery and refuses to repeat a completed run. Resume-monitor output would restart its GPU CSV; no resume was required here. A new comparison must use a new directory.

## Recognition diagnosis

**PASS computational contract:** all 1,580 permitted feature files retain their hashes, embedded labels, schema, strictly increasing timestamps and CTC-feasible lengths. Blank 0 / UNK 1, all 2,289 class IDs and frozen mean/std remain correct. No current unknown-label mapping or CTC infeasibility explains these results.

**PARTIAL recognition:** the model emits 158 hypothesis tokens for 485 references; 93.18% of raw argmax frames are blank. It emits 49 distinct labels, with IX-1p accounting for 51/158 tokens. This is severe deletion and limited diversity, not complete empty-output or single-token collapse. Continued fitting lowers TRAIN loss while DEV CTC loss increases. TRAIN WER remains 85.70%, so this full task is also poorly fitted; the explanation cannot be pure overfitting alone.

Sparse supervision matters: 1,182/2,287 labels have one distinct TRAIN utterance and 1,808 have fewer than five. However, frequent-label recall is also only 25/251 for support >=20, while all other buckets together achieve 1/234. Rare labels are not the sole failure. The clean literal pool is kept separate from NCSLGR main glosses; adding names or combining feature-compatible pools would not repair this.

Prior TRAIN-only observability evidence found 880/12,423 published occurrences with no retained sample, plus 1,440 with no observed hand inside the retained interval. This task's DEV publisher intervals have {timing['zero_sampled_observations']}/485 with zero samples and {timing['one_sampled_observation']}/485 with one. The publisher spans are not qualified physical whole-sign boundaries. Surviving body/context information, unresolved visual annotation quality, source distribution and model capacity/optimization can all contribute; no single causal root cause has been established.

Cached-logit beam tests use no English prior. Baseline beam8 reaches 94.6392% WER; A beam8 reaches 94.8454%, worse than A greedy. Thus beam search is not a consistent remedy. Baseline logits were saved before A; the beam probe itself ran after A, rather than before fitting as requested. This ordering limitation is recorded explicitly. No production decoder or confidence threshold was changed.

## Vocabulary and detection

463 distinct proposed new labels are inventoried, including local HELLO and 462 RIT literals; **zero have passed all data gates**. This is a proposal, not 463 learned words. HELLO has 60 potential isolated recordings from one signer/session, incompatible schema 3.0 and no established consent/review/independent validation. The user supplied the dataset root as the records location; the renewed recursive search found no completed consent/review record. All 480 reviewer_notes fields are empty. The thesis describes future consent and additional written permission, rather than supplying executed records. Participant media were not decoded or processed.

The append-only head, old logits/IDs, checkpoint serialization, new-row CTC gradient and expanded AdamW step pass synthetic engineering tests. No synthetic labels or model were distributed as learned signs. B remains blocked and its future real-data training path is not end-to-end validated. C was not run: A's 1.0309 pp improvement fails the predeclared 2 pp promising screen and streaming WER is unchanged.

No improvement in physical boundary timing, adjacent repeats, genuine no-sign false activations or phone camera latency can be claimed. DEV contains no adjacent repeated-label sequences, no consented no-sign cohort, and no independent boundary review. Host CPU synthetic forward timings are in HOST_INFERENCE_LATENCY.csv and are not phone/end-to-end measurements. No app build/install or asset update occurred. Tensor payload is {stats['model_tensor_bytes']:,} bytes, selected resumable checkpoint {stats['selected_checkpoint_bytes']:,} bytes; APK asset impact is zero.

## Decision and next action

**Do not promote or continue the same-data fit.** Preserve the existing app model. Obtain the actual consent/permission records, qualified annotation provenance and an independent new-class recording session before admitting local HELLO. For continuous expansion, resolve the RIT publisher crop/time origin without guessing or trimming frames. Review the already selected TRAIN observability events with a qualified ASL reviewer, then design a single justified feature/cadence or label-support experiment. More unique classes alone is not the current remedy.

Run `Verify-Completed.ps1` or the direct Python verifier from commands_and_results.txt for read-only integrity checks. `train_expansion_B.py` without --train performs admission checks only and currently exits 2. It must remain blocked until real records satisfy every gate. No reserved TEST evaluation is authorized by this result.
''')

report('VOCABULARY_EXPANSION_REPORT.md',f'''
# Vocabulary expansion evidence and blockers

**PARTIAL engineering; BLOCKED real new-class training.** The model still contains 2,287 literal labels plus blank 0 / UNK 1. New classes added: **0**. Verified admitted examples per new class: **none**. There are 510 proposal rows and 463 distinct proposed new labels; potential metadata occurrences must not be counted as eligible training examples.

| Source | Actual evidence | Admission decision |
|---|---|---|
| Local recorder | 480 JSON records and matching MP4s; 8 prompts x60; one signer, one session per prompt; HELLO is the sole new literal among these prompts | BLOCKED consent/permission records, qualified gloss/target review, genuine 378-feature extraction and independent validation |
| Top-level local videos | 232 named MP4s, 14,394,510 bytes; dictionary/demo-style isolated examples | UNKNOWN rights and authoritative targets; filenames are not labels |
| RIT SignStream3 | 278 media records; 276 after two existing overlap exclusions; 462 proposed new literals | BLOCKED 60fps/crop-origin contract: +2 native frames vs inclusive publisher span; no invented offset, trim or zero padding |
| ASLLRP3 | Existing 2,119 eligible examples already assigned 1,500 internal TRAIN /206 internal held /413 external DEV | No reassignment of held families to supply new classes |
| NCSLGR2 | 1,002 eligible under a different main-gloss annotation contract | Separate pool; no union on the basis of 378-dimensional features |
| How2Sign | Existing English translations | Not authentic gloss sequence supervision; no repeat acquisition |
| DawnSignPress | Official directory is visible, but publisher dataset document excludes its download | No media acquired; visibility is not authorization |

User follow-up supplied `D:\\ASL VIDEOS DATASETS` as the record location. LOCAL_PERMISSION_SCAN.json lists the renewed read-only search across 1,223 files: no permission/consent/review-named records; 480 empty reviewer_notes fields; no consent/license/verified-label metadata keys found. The thesis's ethical section describes future approval and additional written consent; it is not an executed participant release. This does not prove that records cannot exist elsewhere. No participant pixel data, recordings or old schema arrays were used in fitting or perception.

Plan ranking uses actual recorder prompts, literal published occurrence counts, available sessions, missing support and implementation cost. RIT labels such as personal-name tokens and unusual morphology are retained literally; their spelling is not an English translation or a claim of broad user usefulness. All require target-contract review. Forty existing under-supported labels are included for collection/replay planning; no synthetic examples were created.

Implementation: vocabulary_extension.py preserves all old IDs and backbone/output rows, appends sorted approved literals, initializes only added rows with seeded Kaiming-uniform weights and median existing-label bias, and extends Adam moments with zeros only for the appended rows. Old raw logits remain exact at initialization, but extra logits necessarily alter the softmax denominator. Tests include roundtrip serialization, repeat/blank conventions via the canonical decoder, finite new-row CTC gradients and an actual synthetic AdamW update. Synthetic test classes are not actual trained vocabulary additions and no synthetic checkpoint was saved.

B admission requires reviewed permission/target/temporal evidence paths and hashes, disjoint recording families, duplicate/exclusion guards, frozen reviewed TRAIN/VAL records, literal label conversion, exact schema/hash/embedded labels, strictly increasing timestamps and CTC feasibility. The minimum 3 TRAIN /1 independent VAL utterances per new class is a technical smoke threshold, not evidence of generalization. New-data replay uses genuine old TRAIN examples, not held-out records. The future B driver is not proven on real expanded data; it currently fails safely before model fitting.

Authoritative source checks reused in this mission: [ASLLRP terms](https://www.bu.edu/asllrp/dai-terms.html), [SignBank terms](https://www.bu.edu/asllrp/signbank-terms.pdf), [publisher dataset description](https://www.bu.edu/asllrp/about-datasets.pdf), [annotation report](https://www.bu.edu/asllrp/rpt24/asllrpr24.pdf). Research permission, redistribution and production-use rights are distinct. No new dataset payload was downloaded.
''')

report('DATA_READINESS_AND_SPLITS.md',f'''
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
TRAIN1500 {cfg['manifest_sha256']['TRAIN1500']}
DEV80     {cfg['manifest_sha256']['DEV80']}
vocabulary {cfg['vocabulary_sha256']}
```

The lists, not each row's inherited publisher/global `split` string, define the frozen experimental role. TRAIN has 12,423 target occurrences, 2,287 literal labels, three publisher signer IDs, 34 masters and six conservative recording families. DEV has 176 distinct reference labels, all known to TRAIN. It includes six sequences /42tokens in participant2's family and74 /443 in participant3's family. The 27 stronger-support subgroup remains descriptive and does not replace DEV80.

Guards verify no shared sample IDs, source master, raw SHA, decoded pixel SHA or conservative family between TRAIN and DEV. Families use publisher signer plus recording date, grouping all undated masters conservatively. These guards do not isolate causal signer effects: the same signers appear on both sides. Existing source-split policy and all prior six ASLLRP overlaps, one frame-count mismatch and two RIT overlap exclusions remain unchanged. All 3,121 reported eligible utterances and source identities remain preserved; only the authorized1,500 are fitted here. Counts of externally held examples are metadata reuse, not access to reserved TEST.

The reserved17 TEST set was not enumerated, opened, hashed, fitted, selected on or evaluated by this mission. Existing frozen held-out results were not changed. Production, Android, vocabulary packaging, renderer, APK, GLB and thresholds are snapshotted and unchanged. preservation_verify.log confirms221protected files and1,580feature files. The final read-only verifier checks these exact paths without walking any TEST location.
''')

report('MAPPING_DIFF_REPORT.md','''
# Class mapping and head compatibility

**PASS old mapping preservation; BLOCKED real expansion.** MAPPING_DIFF.csv lists all2,289 original IDs, including blank0 and UNK1. Every current-model ID and literal label is unchanged. The selected A head remains [2289,48], with frozen normalization.

The synthetic extension tests appended three unit labels in memory, checked exact old parameter rows and zero old-logit difference, serialization roundtrip and optimizer moments. A separate synthetic test performs an actual expanded AdamW step. Those models were not saved or used as recognition evidence. Only added rows are initialized; old logits are initially preserved, but the normalization denominator changes when classes are appended. B must demonstrate real old/new retention after actual training before any export or packaging.

Invalid or unsupported admission is rejected; blank/UNK targets are not permitted. Literal morphology, classifiers, spatial indices, punctuation and source conventions are not stripped or merged. No English alias was inserted as a recognition target. No production inference/export path was changed. The future B path requires real-data end-to-end testing after its current data blockers are resolved.
''')

report('ANNOTATION_REVIEW_REQUEST.md','''
# Minimum evidence needed for the next data step

**BLOCKED pending actual records.** The user confirmed records are available and supplied the dataset root. Search did not establish executed permission or qualified label records. Provide their actual filenames/paths, or collect the missing evidence through the project's approved procedure. Do not edit metadata to manufacture consent.

For local HELLO, link permission records to recording IDs under `D:\\ASL VIDEOS DATASETS\\records\\AppHearDataset\\samples\\signs\\hello`, the signer/session metadata, permitted academic model-training scope, access/retention rules and any restrictions on derivatives. Record reviewer identity, qualification, authorization, date, actual ASL form and literal target convention, natural continuous context and boundary uncertainty. Do not assume an isolated recorder prompt is the same as a SignStream occurrence label. Obtain at least a separate recording session for new-class validation; a second signer is needed for an unseen-signer claim. Sixty repetitions in one session are not sixty independent conditions.

Before processing participant media, confirm consent and applicable permissions. Re-extract genuine Candidate01 causal378 features only after the target/timing contract is verified; schema3.0 arrays must not be padded/coerced into the new schema. Freeze manifests and family guards before fitting. Supporting existing labels across distinct natural recordings is also a priority because frequent-label recall remains poor.

For RIT, obtain publisher documentation or independently authorized verified evidence of the60fps native clip origin and the two additional frames relative to published inclusive boundaries. Preserve the archive and all originals. Do not guess offsets or silently trim. Keep all morphology and the existing overlap exclusions.

The earlier TRAIN-only12-event review manifest remains an appropriate small qualified-review target. Its cached computational checks passed; physical observability, semantic target correctness and whole-sign boundaries remain unresolved. No review was invented here. No new participant recording, contact or upload occurred.
''')

report('PROMOTION_DECISION.md',f'''
# Promotion decision

**DO NOT PROMOTE. Overall PARTIAL.** A is a real completed fine-tune, not a validated app-recognition improvement. Full-sequence DEV WER changes95.6701%→94.6392%, but exact sequences remain0/80, the paired descriptive interval includes no improvement, DEV CTC loss worsens, and current streaming WER stays95.6701%. Both recording families remain poorly recognized. B added0classes and C's predeclared gate failed.

The criterion for considering C was frozen before fitting: at least2percentage points of full-sequence DEV gain, no insertion-rate regression over2points, and no TRAIN WER regression over5points. A meets neither a meaningful streaming improvement nor the gain threshold. No thresholds were relaxed. Training stopped after10epochs for lack of gain; continued same-data fitting is not justified by these measurements.

Required independent promotion evidence is absent for new classes, unseen signers, consented no-sign behavior, whole-sign boundaries, adjacent repeats and live phone cadence/latency. No numerical threshold is selected retrospectively from these results. A later trial must predeclare its minimum useful change and independently verify retention, false activations, boundaries and device performance before export/deployment.

Retain the unchanged current app model and lifecycle/missing-hand safeguards. No APK was built/installed, renderer modified, threshold tuned, assets copied, Git operation performed or production checkpoint selected. Selected research artifact only: `{a['selected_checkpoint']['path']}`, SHA256 `{a['selected_checkpoint']['sha256']}`.
''')

# Clarify that TRAIN figures paired with streaming DEV are NOT streaming TRAIN measurements.
path=R/'EXPERIMENT_RESULTS.csv';before=ft.sha(path)
with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
for r in rows:
 r['TRAIN_WER_protocol']='full_sequence_greedy' if r['TRAIN_WER'] else ''
 r['TRAIN_baseline_reused']=r['experiment']=='baseline' and bool(r['TRAIN_WER'])
with path.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ft.save(R/'REPORT_CLARIFICATIONS.json',dict(status='REPORT_ONLY_NO_REEVALUATION',artifact='EXPERIMENT_RESULTS.csv',before_sha256=before,after_sha256=ft.sha(path),reason='Explicit TRAIN protocol and baseline-reuse columns; streaming DEV rows reuse full-sequence TRAIN scores. No metric changed.'))

report('SESSION_HANDOFF.md',f'''
# Completed mission handoff

Directory: {R}. A completed; do not repeat it. Selected epoch2 hash{a['selected_checkpoint']['sha256']}. Ten fine-tuning epochs,82.106s, no promotion. See MISSION_REPORT.md and A/RESULTS.json.

B is blocked, not trained. User says consent/verified-label records exist and supplied D:\\ASL VIDEOS DATASETS; renewed search found only future consent language and480empty reviewer_notes. Need actual executed record paths, qualified target review, independent validation and true compatible features. No participant media was opened. RIT timing remains blocked. No downloads.

C failed the predeclared promising screen; no run. Beam probes are not a consistent remedy. Decoder comparison was performed after A on cached logits and the ordering deviation is documented. No further fit, TEST evaluation, APK build/install, production update or Git operation is needed/authorized by these results. Read-only verification is the next command.
''')

root=str(ft.ROOT);py=str(ft.ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe')
report('commands_and_results.txt',f'''
AppHear isolated fine-tuning mission — commands and measured results
Working directory: {root}
Interpreter: {py}
All commands below are PowerShell. No global Python/CUDA installs, media payload downloads, app builds, phone modifications or Git operations occurred.

Completed commands (chronological stages; full stdout/log files retained):
& '{py}' -B '{R/'fine_tune.py'}' prepare
  PASS:1580features,221protectedpaths, source/IDs/normalization/family guards; baseline exact; report created BEFORE optimizer update. Output prepare.log.
& '{py}' -B '{R/'fine_tune.py'}' A
  PASS execution; PARTIAL recognition:10epochs1880mainupdates,1disposableupdate,82.1064008s, peakallocated1303238656bytes; NO_CREDIBLE_DEV_GAIN_STOP. Output A_training.log; A/RESULTS.json.
& '{py}' -B '{R/'decoder_probe.py'}'
  PASS synthetic exhaustive correctness;6cached-logit comparisons. decoder_probe.log. Performed after A, not before it.
& '{py}' -B '{R/'inventory_expansion.py'}'
  BLOCKED newclassadmission;510proposalrows,463distinctnewlabels,0admitted. inventory_expansion.log.
& '{py}' -B '{R/'vocabulary_extension.py'}'
  PASS synthetic-only append/head/serialization/gradient tests. extension_tests.log.
  Setup failure while preparing B scripts: shell working directory was already RUN while relative paths began experiments/continuous_asl/RUN. Set-Content could not find duplicated paths and the relative interpreter was not found. No model/data changed; command corrected to project-root workdir before execution.
& '{py}' -B '{R/'train_expansion_B.py'}'
  EXPECTED exit2: B_ADMISSION_BLOCKED missing B/APPROVED_DATA_MANIFEST.json. B_preflight.log. No B fit.
& '{py}' -B '{R/'test_expanded_optimizer.py'}'
  PASS one actual synthetic expanded AdamW step; no participant data/model saved. expanded_optimizer_test.log.
& '{py}' -B '{R/'post_evaluate.py'}'
  PASS saved-prediction alignments, per-class/bucket/group analyses, descriptive uncertainty, parameter changes and synthetic host latency. post_evaluation.log. No fit.
& '{py}' -B '{R/'local_permission_scan.py'}'
  PASS read-only metadata/DOCX text search after user supplied root.1,223files,480empty reviewer_notes,no executedpermission/labelrecord established. local_permission_scan.log. No participant pixels/feature arrays opened by scan.
& '{py}' -B '{R/'temporal_diagnostics.py'}'
  PARTIAL publisher interval diagnostics:485occurrences,19zero-sample,100one-sample. temporal_diagnostics.log. No threshold tuning/physical boundary claim.
& '{py}' -B '{R/'fine_tune.py'}' verify
  PASS221protectedfiles+1580featurehashes. preservation_verify.log.
& '{py}' -B '{R/'write_reports.py'}'
  Assemble required reports; clarify TRAIN protocol/reused-baseline columns without changing measured metrics. REPORT_CLARIFICATIONS.json.

Final read-only checks (safe to repeat; no training, writes or TEST access):
& '{py}' -B '{R/'verify_completed.py'}'

Equivalent wrapper if script execution policy blocks unsigned .ps1:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '{R/'Verify-Completed.ps1'}'

B admission check only (currently exits2; do not add --train before valid evidence is frozen):
& '{py}' -B '{R/'train_expansion_B.py'}' --manifest '{R/'B/APPROVED_DATA_MANIFEST.json'}'

Future B launch after all evidence is independently reviewed and preflight passes:
& '{py}' -B '{R/'train_expansion_B.py'}' --manifest '{R/'B/APPROVED_DATA_MANIFEST.json'}' --train
  NOT EXECUTED. Future path is unvalidated on real expanded data; new outputs go to fresh B/admitted_timestamp directories. Source and all old mappings retained; original TRAIN replay only. No automatic promotion.

Do not rerun prepare, measurement scripts or A into this completed directory. Exclusive outputs protect evidence. For a separately authorized experiment use a new uniquely named sibling directory and freeze its config/manifests first. A --resume supports interrupted fitting but detects this completed result and will skip; no resume is needed. Original A script is preserved exactly as executed.

Source checks: ASLLRP official resource/terms/annotation/dataset documents. Some web PDF/CSV references were inaccessible through the browser tool; official explicit page/index URLs were used instead. No dataset payload acquired. An accessible DSP index did not override the publisher's download exclusion.
''')
print('PASS: reports assembled; no new fit, label, production or TEST modification.')
