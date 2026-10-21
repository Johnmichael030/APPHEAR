"""Write the new mission's report set from measured files, preserving the prior run."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from common import ROOT,PREVIOUS,SOURCE,run_root,sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
    def read(path):return json.loads((run/path).read_text(encoding='utf-8-sig'))
    def doc(name,text):
        with (run/name).open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')
    how=read('how2sign_inventory.json');pose=read('how2sign_pose_quality.json');a=read('experiment_A/validation_best.json');legacy=read('experiment_A/baseline.json')
    fp=read('experiment_A/warmstart/baseline_fp32.json');warm=read('experiment_A/warmstart/validation_best.json');repeat=read('experiment_A/warmstart_repeat/validation_best.json')
    b=read('experiment_B/validation_best.json');b0=read('experiment_B/verified_10hz_baseline.json');native_control=read('experiment_B/frozen_model_native_cadence.json')
    comparison=read('paired_comparison.json');export=read('export_results.json');preservation=read('preservation_final.json');gpu=read('gpu_capability_report.json')
    audit=read('experiment_A/data_audit.json');brows=read('experiment_B/data/processed/full/manifest.json');bquality=read('experiment_B/data/processed/full/quality.json')
    statuses={key:read(folder+'/training_status.json') for key,folder in [('scratch','experiment_A'),('warmstart','experiment_A/warmstart'),('repeat','experiment_A/warmstart_repeat'),('continuous','experiment_B')]}
    logs={key:list(csv.DictReader((run/folder/'training_log.csv').open())) for key,folder in [('scratch','experiment_A'),('warmstart','experiment_A/warmstart'),('repeat','experiment_A/warmstart_repeat'),('continuous','experiment_B')]}
    regression=ET.parse(run/'logs/full_regression_final.xml').getroot();s=next(regression.iter('testsuite'));counts={k:int(s.get(k,'0')) for k in ('tests','errors','failures','skipped')};counts['passed']=counts['tests']-counts['errors']-counts['failures']-counts['skipped']
    failure_counts=Counter()
    for t in regression.iter('testcase'):
        for kind in ('failure','error'):
            element=t.find(kind)
            if element is not None:failure_counts[(element.get('message') or '').splitlines()[0][:180]]+=1
    write(run/'logs/regression_summary.json',{'counts':counts,'common_messages':failure_counts.most_common(10),'all_failures_preexisting_confirmed':False,
        'known_preexisting':'missing phase7 Keras artifact and absent MediaPipe in reused environment','other_failures':'snapshot paths/fixtures, API differences and isolated-test constraints; not all classified against original environment'})
    with (run/'data/raw/how2sign/how2sign_original_val.csv').open(encoding='utf-8-sig') as f:annotations=list(csv.DictReader(f,delimiter='\t'))
    with zipfile.ZipFile(run/'data/raw/how2sign/val_rgb_front_clips.zip') as z:video_names={Path(n).stem for n in z.namelist() if Path(n).suffix.lower() in ('.mp4','.mov','.avi')}
    missing=sorted({r['SENTENCE_NAME'] for r in annotations}-video_names)
    write(run/'how2sign_annotation_details.json',{'rows':len(annotations),'archive_video_members':len(video_names),'missing_video_for_annotation':missing,
        'annotation_type':'ENGLISH_TRANSLATION','authentic_gloss_release_verified':False,'public_gloss_download':'NOT_AVAILABLE_IN_INSPECTED_RELEASE',
        'pilot_container_timing_consistent':all(r['timing_status']=='CONSISTENT' for r in how['usable_pilot']),
        'semantic_alignment':'not manually evaluated','manual_realignment_vs_original_clips':'original CSV used for original sentence clips; existing realigned CSV not silently substituted'})
    # Extend the ledger with failed attempts and reused existing data; no successful file is re-downloaded.
    ledger=run/'how2sign_download_manifest.csv';existing_ids={r['id'] for r in csv.DictReader(ledger.open())};append=[]
    for key,url,status in [('stale_official_script','https://drive.google.com/uc?export=download&id=1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1','FAILED_HTTP_404'),
        ('legacy_confirmation_detector','https://drive.google.com/uc?export=download&id=1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO','FALSE_POSITIVE_CSS_RESTRICTION_CORRECTED_IN_NEW_TOOL')]:
        append.append(dict(id=key,source_url=url,path='',status=status,bytes=0,sha256=''))
    for item in how['before']['ENGLISH_TRANSLATION_REALIGNED']:
        append.append(dict(id='existing_'+Path(item['path']).stem,source_url='verified prior receipt',path=item['path'],status='PRESENT_AND_VERIFIED_REUSED',bytes=item['bytes'],sha256=item['sha256']))
    with ledger.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('id','source_url','path','status','bytes','sha256'))
        for row in append:
            if row['id'] not in existing_ids:writer.writerow(row)
    # Manifest IDs explicitly distinguish feature-row indices from original video identities.
    vocab=json.loads((run/'experiment_A/vocabulary.json').read_text());inverse={int(v):k for k,v in vocab.items()}
    fields=('sample_id','sample_kind','source','source_path','modality','label_type','label','signer','split','duration_seconds','duration_status','license_notes','validation_status','acquisition_status')
    with (run/'dataset_manifest.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for split in ('train','val'):
            with np.load(run/'experiment_A/data'/split/'metadata.npz',allow_pickle=False) as data:
                for i,(label,signer) in enumerate(zip(data['y'],data['participant_ids'])):
                    w.writerow(dict(sample_id=f'canonical_{split}_feature_row_{i:05d}',sample_kind='DERIVED_FEATURE_ROW_NOT_RAW_CLIP_ID',source='ASL Citizen-derived canonical bundle',
                        source_path=str(SOURCE/f'data/train/{split}.npz'),modality='LEGACY_NORMALIZED258',label_type='ISOLATED_ASL_GLOSS',label=inverse[int(label)],signer=str(signer),split=split,
                        duration_seconds='',duration_status='UNKNOWN_NO_ORIGINAL_TIMESTAMPS_IN_NPZ',license_notes='Microsoft Research License, private noncommercial research only',
                        validation_status='SHA_SCHEMA_FINITE_MASK_SIGNER_CHECKED; raw sample pairing not independently reconstructed',acquisition_status='EXISTING_VERIFIED_NOT_DOWNLOADED'))
        for row in brows:
            w.writerow(dict(sample_id=row['sample_id'],sample_kind='COMPLETE_UTTERANCE',source='NCSLGR',source_path=row['source_url'],modality='RAW_RGB_AND_CAUSAL378_NATIVE',label_type='ASL_GLOSS_SEQUENCE',
                label=' '.join(row['glosses']),signer='',split=row['split'],duration_seconds=row['container_duration_seconds'],duration_status='CONTAINER_FPS_UNCALIBRATED',license_notes='BU private research only, no redistribution',
                validation_status='PAIR_HASH_FRAME_ORDER_CTC_LENGTH_VERIFIED; signer unknown',acquisition_status='RGB_REUSED_NEW_NATIVE_FEATURES'))
        for row in how['usable_pilot']:
            w.writerow(dict(sample_id=row['sample_id'],sample_kind='SENTENCE_CLIP',source='How2Sign',source_path=row['path'],modality='RGB_FRONT_AND_POSE_PROBE',label_type='ENGLISH_TRANSLATION',
                label=row['reference_english'],signer='',split='official_val',duration_seconds=row['duration'],duration_status=row['timing_status'],license_notes='CC BY-NC 4.0 + research-only; no product clearance',
                validation_status='FULLY_DECODED_AND_ORIGINAL_ANNOTATION_PAIRED; semantic alignment not manually verified',acquisition_status='NEW_PILOT'))
    write(run/'baseline_results.json',{'production_current_RecognitionV2':{'status':'BLOCKED','reason':'new available data are legacy258, current WORD expects350 plus mask; no compatible current-path held-out benchmark'},
        'isolated_legacy_INT8':{k:legacy[k] for k in ('top1','macro_f1','rows','model_path','model_sha256','latency_p50_ms','latency_p95_ms')},
        'isolated_legacy_FP32':{k:fp[k] for k in ('top1','macro_f1','rows','per_signer')},
        'continuous_prior_reproduced':{k:b0[k] for k in ('gloss_wer','substitutions','deletions','insertions','reference_tokens','hypothesis_tokens','checkpoint')},
        'frozen_CTC_on_native_cadence':{k:native_control[k] for k in ('gloss_wer','substitutions','deletions','insertions','reference_tokens','hypothesis_tokens')},
        'test_data_used':False,'software_environment':str(PREVIOUS/'requirements.observed.txt')})
    rp=ROOT.as_posix();rr=run.as_posix();pp=PREVIOUS.as_posix();tools=ROOT/'tools/asl_research_v2'
    doc('recovery_report.md',f'''# Recovery — PASS with explicit limitations

| Item | Status | Evidence |
|---|---|---|
| Exact repository / prior report | PASS | `{rp}`; prior run `{pp}` exists. Report, continuation, configuration, checkpoints, scripts, tests and logs read. |
| Original preservation | PASS | All 2,932 original files retain SHA256 and historical modification time. |
| Completed prior artifacts | PASS | 401 prior artifact hashes verified, plus original manifest; no prior report/log/checkpoint overwritten. |
| Git identity | PARTIAL | No .git metadata; no current branch/commit can be asserted. File manifests are authoritative here. |
| Python / GPU | PASS | Existing Python 3.11.11 CUDA PyTorch reused; actual input/parameter/loss/backward/optimizer and finite checks passed. No environment/driver upgrade. |
| Storage | PASS | D: initially 178,989,453,312 bytes free; explicit 8 GiB run budget, 2 GiB archive cap, 64 MiB pilot extraction cap. New staging about 1.63 GB; retained regression copy about 1.57 GB. |
| RAM / VRAM | PASS | Memory sampled inside actual training workers, correcting the prior launcher-RSS limitation. Logs contain working set, peak working set, private commit, pagefile-backed peak and CUDA peak. |
| How2Sign | PASS for bounded acquisition | Existing realigned translations verified; current official archive and original annotation acquired, CRC checked; eight clips decoded/paired. Not full-dataset acquisition. |
| Other datasets | PASS/PARTIAL | Eligible ASL Citizen-derived features were found on D:, unlike the prior repository-only search; full raw provenance is not reconstructed from row-only NPZs. NCSLGR original RGB reused by hash. |
| Splits / labels | PASS/PARTIAL | A: 35 train / 5 val signer IDs, zero overlap and official split concordance. B: collection-disjoint, signer independence unverified. No TEST tensors loaded for model work. |
| Prior WER / ONNX | PASS | Recomputed 88% = 6S+16D+0I /25 references, 9 hypothesis tokens. Prior checkpoint/export hashes still match; no completed training resumed. |
| Original missing Keras source | BLOCKED | Frozen phase7 source checkpoint not found in relevant repository/data roots. No substitute manufactured. |
| Running work audit | PARTIAL | NVIDIA initially idle/0 MiB; own jobs ran sequentially and finished. No exhaustive initial CPU download-process inventory is asserted. |

Recovery mismatch list is empty. New phases write only `{rr}` and new tools under `{tools.as_posix()}`. Old artifacts and source models remain untouched. New root template training_config.yaml was copied by the run initializer but is **not** the configuration for these candidates; their actual configs are under experiment_A and experiment_B_plan.json.
''')
    doc('how2sign_acquisition_report.md',f'''# How2Sign — bounded acquisition VERIFIED

Official source: [How2Sign project](https://how2sign.github.io/) and its linked downloader. Inventory was scoped to this repository and relevant D: ASL/data/GPU-package directories; unrelated folders were not exhaustively searched. The only verified existing modality was the prior manually re-aligned English TRAIN/VAL tables (32,906 rows total). Those files were reused, not downloaded again. No prior RGB, sentence archive or OpenPose payload was found in searched roots.

The Bash downloader's frontal validation ID returned HTTP404. The current project-page link instead uses `1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO`. This documented link yielded **1,778,074,855 bytes**, SHA256 `6cb48170131fdb7fc64d40381a3c4be41112ba4454444170f5a31af2227b6d36`. No official digest was published in the inspected source; this is an observed integrity hash. All archive members passed ZIP CRC and bounded path/link/size checks. It contains 1,739 video members; these were not all extracted or decoded.

Eight complete validation clips were deterministically selected by paired filename and extracted to a fresh directory: 5,046,739 bytes, 811 fully decoded frames, 1280x720, 24 FPS, 33.792 total container seconds. All eight durations agree with original annotation bounds within two frames/0.1s. Remaining members stay only in the retained archive. The original validation CSV was newly acquired from the current official page (311,131 bytes, SHA `f6f0cef01c3f891ca624a45c71eb7f7748b77362d7418c47567a984384f1537c`), distinct from the existing realigned table. Its 1,741 annotations have {len(missing)} names absent from the archive; see how2sign_annotation_details.json. Semantic ASL-English alignment was not manually assessed.

Storage plan: storage_plan.json; measured free space passed. Archive cap2GiB; extracted pilot64MiB; streaming extraction needs no full expanded copy; checkpoints/logs bounded; no cleanup/deletion. Dry runs occurred before acquisition. The original wrapper falsely rejected a normal public warning because of an account link and hidden CAPTCHA-named CSS. The new version parses visible restrictions, verifies the public form's file ID and host, and still rejects actual challenge widgets/login-only/quota pages. No account, CAPTCHA or authorization control was bypassed. Failure logs and genuine public response are retained. New successful rows, failed attempts and reused items are distinguished in how2sign_download_manifest.csv.

Permissions: research-only project terms and CC BY-NC4.0, attribution/change notices; no commercial deployment, redistribution or model-distribution clearance is asserted. No media was published or uploaded to external services. Full RGB train/test, side view and 2D keypoints were not acquired. Dataset material stays local. The eight-clip pose probe preserves features separately but never turns English into glosses.
''')
    doc('how2sign_annotation_audit.md',f'''# Annotation audit — English is not gloss

Inspected original validation TSV columns: VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME, START, END, SENTENCE. These carry English sentence translations and sentence timing, not authentic ASL gloss token sequences or individual sign boundaries. Existing realigned files have the same translation role. Acquired RGB is VIDEO_ONLY when unpaired; the derived probe is POSE_ONLY plus a separately stored ENGLISH_TRANSLATION reference. Neither is ASL_GLOSS_SEQUENCE or GLOSS_WITH_TIMESTAMPS.

Official project download links and the current linked script were reviewed on 2026-10-20 Asia/Shanghai. The script still has a gloss-annotation TODO; no public authentic-gloss download was found/validated. Status: **NOT AVAILABLE in the inspected official release**, not a claim that private annotations never existed. How2Sign gloss CTC is BLOCKED. No labels were generated from English. Signer identity is absent from the inspected tables; source-video IDs are not fabricated signer IDs.

The archive's 1,739 filenames pair with the original 1,741-row validation table, with {len(missing)} annotations missing corresponding archive videos. Eight extracted clips have verified container timing; full semantic alignment is unverified. Original clips were released with original timing, so the original CSV was used rather than silently pairing realigned timing. All pilot data belong to official validation. Optional translation training was NOT RUN: this pilot is not a valid training partition, and no translation baseline/model was selected. TEST data were not acquired.
''')
    doc('dataset_eligibility_report.md','''# Dataset eligibility and separate learning tasks

| Source | Task / evidence | Research decision | Remaining rights/data limits |
|---|---|---|---|
| ASL Citizen + canonical prepared bundle | Isolated glosses, 236 classes, [96,258], verified NPZ hashes/schema/masks/finite values | USED for A private noncommercial research | Local Microsoft Research License permits noncommercial/nonrevenue research, restricts distribution and personal-data use. No production/distribution clearance. |
| NCSLGR | Authentic official-index gloss sequences; 37 complete RGB utterances | USED for B private research | [BU terms](https://www.bu.edu/asllrp/dai-terms.html) restrict commercial use and redistribution; signer/timed gloss XML unavailable without authorized account access. |
| How2Sign | Eight newly decoded frontal validation clips + English tables + pose probe | Acquisition/quality probe only | [Project](https://how2sign.github.io/) research/CC BY-NC terms. No validated public gloss targets; no training on official validation. |
| OpenASL | ASL-English translation, previously excluded ND terms | EXCLUDED | Training/derivative permissions not cleared; no payload acquired or used. |
| WLASL | Isolated words; C-UDA agreement | EXCLUDED pending eligibility review | No agreement acceptance or verified payload in this run; never substituted as continuous labels. |
| Local word videos / alphabet image folders | Filenames or static-letter labels do not establish sequence supervision/consent | EXCLUDED | Provenance/permissions unresolved, and static alphabet images do not establish streaming fingerspelling labels. |

Raw ASL Citizen inventory: 83,399 video files, 49,600,223,768 bytes, plus official split CSVs and use.txt. This is metadata inventory, not full raw-video decoding/hash validation. The selected bundle has 13,236 TRAIN and 3,264 VAL **feature rows**, potentially multiple cadence views per source clip; raw clip IDs/timing are absent. Do not call them 16,500 independent videos. TRAIN class counts32–80, VAL4–24. All236 labels are retained. The vocabulary matches the unchanged legacy model's source vocabulary. Bundle signer sets match official train/validation membership:35/35 TRAIN,5 of6 official VAL; overlap0. Raw sample lineage/augmentation multiplicity was not independently reconstructed. No local recorder data were silently mixed in.

Prepared feature provenance remains legacy normalized258; current Candidate01/RecognitionV2 uses a different representation. Research authorization does not make either candidate a drop-in production replacement. Retention/distribution obligations remain recorded; no source or derived media/model was redistributed. No additional negative-data consent is invented. Full private/raw content is not included in the reports.
''')
    doc('isolated_sign_training_report.md',f'''# Experiment A — executed, modest repeated validation gain

Eligible existing data were used immediately without waiting for How2Sign. Exact TRAIN/VAL source hashes and tensor schemas: experiment_A/data_audit.json. Signer sets are disjoint and concordant with official ASL Citizen partitions. Class vocab236, feature schema3.0 [T,258], masks and valid sequence lengths4–96. All feature blocks remain; no geometric augmentation or negative relabeling. CPU mmap staging avoids loading the whole corpus into VRAM. No TEST payload was loaded for checkpoint/threshold/augmentation decisions. Metrics below are isolated-clip validation, not signed-phrase or current live-app accuracy.

| Candidate | Completed epochs | Validation top1 | Macro-F1 |
|---|---:|---:|---:|
| Unchanged legacy INT8 TFLite | no training | {legacy['top1']*100:.2f}% | {legacy['macro_f1']*100:.2f}% |
| Unchanged legacy FP32, parity-gated Keras transfer | no training | {fp['top1']*100:.2f}% | {fp['macro_f1']*100:.2f}% |
| Fresh masked TCN | 5 | {a['top1']*100:.2f}% | {a['macro_f1']*100:.2f}% |
| Warm-start clip-objective, seed20261020 | 5 | {warm['top1']*100:.2f}% | {warm['macro_f1']*100:.2f}% |
| Independent repeat, seed20261021 | 5 | {repeat['top1']*100:.2f}% | {repeat['macro_f1']*100:.2f}% |

The scratch model was not competitive within the bounded run. A justified second candidate transfers the unchanged FP32 model, preserving exact causal block/LN/skip/head math. CPU Keras/PyTorch transfer maximum error1.55e-6, no argmax changes. Its new loss is NLL of mean real-frame class probability, matching the existing clip decision; learning rate1e-4, AdamW, microbatch4, accumulation8 (effective32), no AMP, no augmentation, gradient cap5. Each run is capped at5epochs/600s, with atomic per-epoch checkpoints, optimizer/RNG states and immutable numbered checkpoint files. Seeds differ only in the repeat; checkpoint selection uses validation macro-F1 then top1.

Against the matched FP32 baseline, primary top1 changes by {100*(warm['top1']-fp['top1']):.2f} percentage points and macro-F1 by {100*(warm['macro_f1']-fp['macro_f1']):.2f}; repeat changes by {100*(repeat['top1']-fp['top1']):.2f}/{100*(repeat['macro_f1']-fp['macro_f1']):.2f} points. Primary paired signer bootstrap95% interval for top1 difference: [{comparison['paired_signer_bootstrap_top1_delta_95pct'][0]*100:.2f}, {comparison['paired_signer_bootstrap_top1_delta_95pct'][1]*100:.2f}] points,2000 resamples. Only five signer clusters, correlated cadence views and validation checkpoint selection limit inference; no independent final-test claim. Two seeds support a modest validation effect, not production readiness.

Per-class precision/recall/F1, confusion matrices, per-signer outcomes and predictions are retained in validation JSONs and confusion CSVs. Positive-only confidence/coverage curves are in paired_comparison.json; no operating threshold was selected. False activations, real false rejection and hard negatives are unmeasured. New fingerspelling training is BLOCKED: no eligible temporal letter-sequence dataset was verified. Existing spelling paths/tests are preserved.

Primary checkpoint: `{(run/'experiment_A/warmstart'/read('experiment_A/warmstart/checkpoints/best.json')['path']).as_posix()}`. Repeat checkpoint: `{(run/'experiment_A/warmstart_repeat'/read('experiment_A/warmstart_repeat/checkpoints/best.json')['path']).as_posix()}`. Exact configurations, hashes, per-epoch time/worker RAM/VRAM and logs remain in their separate directories. Current production WORD expects350 features and is not directly comparable; its accuracy improvement is NOT MEASURED. No production checkpoint selected.
''')
    doc('continuous_sign_training_report.md',f'''# Experiment B — executed, native cadence not selected

How2Sign gloss training is BLOCKED because acquired labels are English. Independently, verified NCSLGR authentic gloss sequences supported CTC. Old RGB/checkpoint artifacts were reused, not re-downloaded or re-trained as the same artifact. The pre-registered change derives all genuine native frames rather than the old max10Hz observations: 2,798 versus944 observations across the same37 complete utterances. New schema apphear.continuous_asl.candidate01_causal378_native.v1 retains all hand/body blocks, masks, true timestamps and causal deltas; no invented frames, depth or gloss alignment. TRAIN30/VAL7 by unchanged recording collection; signer independence remains UNVERIFIED. Test payload not used.

The prior 10Hz checkpoint re-evaluates at88% gloss WER:6 substitutions,16 deletions,0 insertions over25 reference tokens;9 hypothesis tokens. These are authentic case-preserved index gloss tokens; English is not involved. The denominator is reference-token count, not hypothesis count or number of clips. Four references are OOV under TRAIN-only vocabulary. 88% WER does not mean88% accuracy.

New native-rate CTC: {statuses['continuous']['completed_epochs']} epochs, early stop,48channels,RF61observations, FP32 AdamW lr0.001, batch4, seed20261019. Best WER **{b['gloss_wer']*100:.1f}%** = {b['substitutions']}S+{b['deletions']}D+{b['insertions']}I /{b['reference_tokens']} refs, {b['hypothesis_tokens']} hypothesis tokens. Frozen old model on native inputs WER {native_control['gloss_wer']*100:.1f}%. New candidate does not improve the88% original diagnostic. Its higher cadence also shortens RF61 in physical time, so this is a representation/cadence diagnostic, not an isolated causal conclusion that more frames are worse. No larger tuning grid is justified on25 reference tokens.

Preprocessing passed all37 clips, monotonic/nonempty/finite/CTC-length and causal-prefix checks. Full native extraction took {bquality['elapsed_seconds']:.2f}s; per-frame p50/p95 {bquality['extraction_ms_p50']:.2f}/{bquality['extraction_ms_p95']:.2f}ms. This CPU processing cannot establish native30FPS camera operation. An initial launch before quality.json was ready failed safely; the retained retry trained only after PASS. CUDA loss/gradients were finite; real worker RAM and VRAM are logged. CUDA CTC is not promised bitwise deterministic.

No valid individual-boundary or non-sign labels exist, and the user previously confirmed no additional authorized negatives/timing data. No rejection/boundary heads, threshold calibration, false-activation claim or fluent fingerspelling merger was fabricated. Candidate stays unselected; new B export was NOT RUN because it was not promising. Prior export hash remains verified, without assuming parity for a new checkpoint. Best new checkpoint: `{(run/'experiment_B'/read('experiment_B/checkpoints/best.json')['path']).as_posix()}`.
''')
    ncs_observed=[sum(r['hand_observed_fraction'][i]*r['sequence_length'] for r in brows)/2798 for i in range(2)]
    h_observed=[sum(r['hand_observed'][i]*r['observations'] for r in pose['clips'])/pose['observations'] for i in range(2)]
    doc('failure_analysis.md',f'''# Feature, temporal and rejection failure analysis

NCSLGR native-frame hand-observed fractions: {ncs_observed[0]:.3f}/{ncs_observed[1]:.3f}. How2Sign720p probe: {h_observed[0]:.3f}/{h_observed[1]:.3f}, body available1.0,273 selected genuine observations from811 frames. Probe extraction p50/p95 {pose['extraction_ms_p50']:.2f}/{pose['extraction_ms_p95']:.2f}ms. A100ms minimum sampling interval on24FPS video yields roughly8Hz, not exact10Hz. These are extractor evidence flags, not manual hand-occlusion truth. Higher resolution alone did not uniformly fix missing hands. Per-side proposal presence, refined joint validity and observed fractions are saved per clip; no confidence threshold/ROI expansion was adopted merely to inflate detector presence.

Native temporal density adds1,854 genuine observations but did not improve seven-clip gloss WER; the context duration changes with RF61. Camera timing, fingers crossing, crop/distance/orientation and left/right identity remain insufficiently annotated for causal attribution. Container timestamps and upright/unmirrored source semantics are retained; no interpolation, carry-forward, fabricated depth or relabeling. No externally visible ground-truth landmarks were invented. The8 How2Sign clips contain English references only; pose features were preserved under data/derived/how2sign_pose_probe, separate from any gloss dataset.

For legacy A, an all-zero21x3 hand block is used only as a missing-hand proxy. Paired-comparison quality groups report model accuracy by both-hands-absent fraction; this is observational association, not proof of manual occlusion or changed recognition performance. Source views, lighting/distance annotations and exact original sample timing are absent from the prepared NPZ; do not infer them from row order. Signer IDs are validated; no identities beyond pseudonymous IDs are published.

The initial scratch model underfit within5epochs. Starting from parity-verified existing weights and matching clip loss to its inference aggregation produced modest repeated validation gains. This does not prove larger capacity, arbitrary augmentations or changed feature geometry would help. Supported236classes remain unchanged; class imbalance is documented32–80 TRAIN rows/class. No missing pose/hand block was dropped for speed.

Non-sign and hard-negative truth are absent. Positive-only confidence/coverage and class-margin ideas cannot establish false activations/hour or safe rejection. No threshold is calibrated for deployment. Boundary F1, transitions, pauses/incomplete signs, real false accepts/rejects and live-camera frame-loss behavior remain BLOCKED. The isolated input validator rejects wrong schema/width, nonfinite/empty masks and lengths outside1–96; a160-step synthetic per-frame causal/RF probe passes without being scored as ASL. CTC long natural streams and reliable boundary behavior remain unvalidated. Original long-ACTIVE buffer expiry, top1 commitment and unused flag findings persist because production logic was not changed.
''')
    doc('export_and_performance_report.md',f'''# New export and measured performance

Primary warm-start ONNX: {export['status']}, {export['bytes']:,} bytes, SHA `{export['onnx_sha256']}`. Input FP32[B,T,258] plus boolean mask; output236clip probabilities in unchanged vocabulary order. CPU PyTorch/ORT same-input parity on3264validation rows: max {export['max_abs_probability_error']:.9g}, mean {export['mean_abs_probability_error']:.9g}, zero decoded class changes. Dynamic batch/time tested on genuine lengths; synthetic160-frame per-frame RF test maximum {export['61_observation_receptive_field_max_error']:.3g}. Caller contract restricts isolated windows1–96 and rejects empty/nonfinite input; empty masks must not be argmax-committed. No Android adapter or flag was changed.

ORT CPU invocation p50/p95 {export['CPU_ORT_p50_ms']:.3f}/{export['CPU_ORT_p95_ms']:.3f}ms; synchronized PyTorch CUDA {export['CUDA_PyTorch_p50_ms']:.3f}/{export['CUDA_PyTorch_p95_ms']:.3f}ms,100 warmed real inputs, transfers excluded. ORT construction {export['cold_ORT_session_ms']:.2f}ms single cold-process sample. Legacy TFLite CPU p50/p95 {legacy['latency_p50_ms']:.3f}/{legacy['latency_p95_ms']:.3f}ms; different runtime/precision/head output means this is not an architecture-only speed comparison. Model-only timings do not establish end-to-end FPS.

Primary training peak allocated VRAM {max(int(r['peak_vram_allocated']) for r in logs['warmstart']):,} bytes; true worker peak working set {statuses['warmstart']['memory']['peak_working_set_bytes']:,} bytes, private commit {statuses['warmstart']['memory']['private_commit_bytes']:,}. Repeat and B independent memory records are saved. First scratch training {statuses['scratch']['seconds']:.2f}s; primary warm-start {statuses['warmstart']['seconds']:.2f}s; repeat {statuses['repeat']['seconds']:.2f}s; B {statuses['continuous']['seconds']:.2f}s plus167s preprocessing. GPU monitors record device utilization/total memory/temperature during A jobs; no concurrent GPU training jobs were launched. Short laptop tests do not establish sustained Android thermal behavior.

Current WORD350 and fingerspelling405 contracts are incompatible with this legacy258 candidate. The seed-repeat and nonpromising B checkpoints were not separately exported; no parity is claimed for unexported artifacts. No Android FPS, delegate/operator compatibility or signed-phrase accuracy claim. Production integration: NOT INTEGRATED.
''')
    configs={'scratch':read('experiment_A/config.json'),'warmstart':read('experiment_A/warmstart/config.json'),'repeat':read('experiment_A/warmstart_repeat/config.json'),'continuous':read('experiment_B_plan.json')}
    write(run/'experiment_config.json',{'run_id':run.name,'repository':str(ROOT),'previous_run':str(PREVIOUS),'python':str(PREVIOUS/'runtime/venv/Scripts/python.exe'),
        'production_enabled':False,'configs':configs,'storage_plan':read('storage_plan.json'),'task_separation':['A isolated signs','B authentic gloss CTC','C English translation NOT_RUN'],
        'How2Sign_training':False,'test_payload_used_for_model_work':False,'root_template_config_used':False,'source_scripts':{str(f.relative_to(ROOT)):sha(f) for f in tools.glob('*') if f.is_file()}})
    # Repeat is appended as a distinct candidate, never confused with a rerun of the completed artifact.
    with (run/'comparison_results.csv').open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('task','candidate','metric','value','comparison','status'))
        for metric in ('top1','macro_f1'):w.writerow(dict(task='isolated_gloss',candidate='warmstart_repeat_seed20261021',metric=metric,value=repeat[metric],comparison='same validation signer IDs, independent training seed',status='DIAGNOSTIC_VALIDATION'))
    phases='''param([ValidateSet('Verify','Acquire','ResumeScratch','ResumeWarm','ResumeRepeat','ResumeContinuous')][string]$Phase='Verify')
$ErrorActionPreference='Stop'
'''+f"$repo='{rp}'\n$run='{rr}'\nSet-Location -LiteralPath $repo\n"+'''switch($Phase){
 'Verify' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Verify }
 'Acquire' { & "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run -DryRun; & "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run }
 'ResumeScratch' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainA -Resume }
 'ResumeWarm' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase WarmStart -Resume }
 'ResumeRepeat' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase WarmRepeat -Resume }
 'ResumeContinuous' { & "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainB -Resume }
}
'''
    doc('reproduction_commands.ps1',phases)
    doc('reproduction_notes.md',f'''# Exact executed commands and safe continuation

All commands used working directory `{rp}` and interpreter `{pp}/runtime/venv/Scripts/python.exe`. No dependency reinstall, global Python or driver change occurred. Primary commands:

```powershell
$repo='{rp}'
$run='{rr}'
Set-Location -LiteralPath $repo
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Recover
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainA
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase BaselineA
& "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run -DryRun
& "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase How2Verify
& '{pp}/runtime/venv/Scripts/python.exe' -B "$repo/tools/asl_research_v2/prepare_b_features.py" --run-root $run --tag full
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainB
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Transfer
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase WarmStart
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Compare
& '{pp}/runtime/venv/Scripts/python.exe' -B "$repo/tools/asl_research_v2/warmstart_repeat.py" --run-root $run
```

These commands are execution history, not an instruction to overwrite completed outputs. Fresh-output guards refuse existing stages. `reproduction_commands.ps1` provides safe continuation: completed-run resumes verify the latest checkpoint and start no extra epochs; acquisition reuses a verified complete archive without network download. For a new replication, create a new named run via the existing New-Experiment.ps1 initializer, stage the verified TRAIN/VAL sources and transfer fixtures there, keep its source/feature/config hashes, and reuse existing verified media instead of downloading it again. Do not choose a new seed/checkpoint based on TEST data. Restore the exact missing phase7 source artifact from an authoritative archive to its recorded path/hash; a similarly named legacy model is not a substitute.

Broader tests used regression_sandbox.py on an independent copy with external writes/subprocess/network denied. Native Windows long-path prefix handling and pytest's default null log target needed harness corrections; retained earlier attempts are not model-regression evidence. Full tests still have missing MediaPipe and historical fixture/API/snapshot failures. Use a separately validated regression environment if resolving dependencies; do not modify the reused completed-run environment. Exact error messages are in logs/full_regression_final.txt and XML. No command can manufacture missing consented negatives, signer/timing XML or Android device evidence.
''')
    acceptance={'GPU environment and training execution':'PASS','How2Sign inventory and download verification':'PASS_BOUNDED_PILOT; two annotation/video gaps documented',
        'How2Sign gloss annotation availability':'NOT AVAILABLE_IN_INSPECTED_PUBLIC_RELEASE','Dataset eligibility and permissions':'PARTIAL_PRIVATE_RESEARCH_ONLY',
        'Isolated-sign experiment':'PARTIAL_REPEATED_VALIDATION_GAIN; new fingerspelling blocked','Continuous-sign experiment':'PARTIAL_TRAINED_NO_GAIN',
        'Accuracy improvement against matched legacy validation baseline':'PASS','Accuracy improvement against current production live path':'NOT MEASURED',
        'False-activation improvement':'NOT MEASURED','Long-sequence and boundary behavior':'PARTIAL_ENGINEERING_CHECKS; natural boundary evaluation BLOCKED',
        'Speed and memory':'PARTIAL_LAPTOP_MEASURED_NO_ANDROID','Export parity':'PASS_PRIMARY_NEW_EXPORT','Regression suite':'FAIL_ENVIRONMENT_FIXTURE_AND_UNCLASSIFIED_SNAPSHOT_FAILURES',
        'Production-asset preservation':preservation['status'],'Production integration':'NOT INTEGRATED'}
    write(run/'acceptance_status.json',acceptance)
    doc('final_status.md',f'''# AppHear separate ASL experiments — PARTIAL

**Actual training executed on the RTX3050: three separate isolated-sign runs (5epochs each) and one new continuous-gloss run (11epochs). No production model, threshold, flag, Android behavior, app data or avatar asset changed.**

The matched isolated FP32 baseline scores65.13% top1 /64.12% macro-F1. Primary warm-start scores67.10% /66.57%; repeat66.54% /65.78%. These are modest repeated validation gains on3,264 feature rows from5 held-out signer IDs, not independent final-test or live-app evidence. The initial scratch model scored19.45% /15.49% and was not selected. Current production RecognitionV2 accuracy remains NOT MEASURED on these incompatible legacy258 features.

Continuous baseline WER reproduces88% (6S+16D+0I /25 refs,9hyp tokens), while the native-cadence candidate scores92%. It stays unselected. Recording collection separation does not prove signer independence. Negative/boundary truth remains unavailable; no false-activation/rejection improvement is claimed. CTC blank/UNK is not calibrated rejection.

How2Sign newly acquired:1,778,074,855-byte official validation clip archive and311,131-byte original English table. ZIP CRC/path checks passed;1,739 archive clips pair to1,741 annotation rows with2missing videos. Eight clips (5,046,739bytes,811frames,720p24FPS) were fully decoded/timing checked. Existing re-aligned English TRAIN/VAL annotations were verified/reused. Eight-clip pose features (273observations) were preserved. Authentic gloss labels were NOT AVAILABLE in the inspected public release, so no How2Sign gloss training occurred. The official Bash script had a stale404 identifier; the current website link worked after a public-confirmation detector correction. No access controls were bypassed. No TEST or full RGB training dataset was acquired, and no translation model was trained on official validation.

Primary new ONNX700,036bytes: max probability error2.09e-6, zero class changes across3264validation rows. CPU ORT p50/p95 0.463/0.784ms, model-only. Landmark extraction remains roughly47–68ms depending on source/cadence; no Android FPS claim. Actual worker memory and VRAM are recorded, with primary warm-start peak working set {statuses['warmstart']['memory']['peak_working_set_bytes']:,}bytes and allocated VRAM {max(int(r['peak_vram_allocated']) for r in logs['warmstart']):,}bytes. No simultaneous GPU training jobs were launched.

Focused suite:65passed, including9new model/contract/checkpoint checks. Public confirmation tests pass. Full protected-copy suite: {counts['passed']}passed, {counts['failures']}failed, {counts['errors']}errors, {counts['skipped']}skipped across {counts['tests']}JUnit entries. Missing MediaPipe, missing historical files, Windows snapshot paths/API assumptions and some unclassified failures prevent a green regression gate. The prior missing phase7 Keras artifact remains absent; no substitute was fabricated. All2,932 original files retain hashes and timestamps; all401 completed prior artifacts plus their manifest remain preserved. New sources and external training input hashes are recorded. Git metadata is absent.

Acceptance statuses:

| Gate | Status |
|---|---|
'''+''.join(f'| {key} | {value} |\n' for key,value in acceptance.items())+f'''

Exact paths:

- New run: `{rr}`
- Tooling: `{tools.as_posix()}`
- Primary checkpoint: `{(run/'experiment_A/warmstart'/read('experiment_A/warmstart/checkpoints/best.json')['path']).as_posix()}`
- Repeat checkpoint: `{(run/'experiment_A/warmstart_repeat'/read('experiment_A/warmstart_repeat/checkpoints/best.json')['path']).as_posix()}`
- Continuous checkpoint: `{(run/'experiment_B'/read('experiment_B/checkpoints/best.json')['path']).as_posix()}`
- Training/resource/evaluation logs: `{rr}/logs`, with per-candidate training_log.csv and validation_best.json.
- Safe continuation commands: `{rr}/reproduction_commands.ps1`; executed command history and fresh-run cautions: reproduction_notes.md.

Remaining blockers: current350-feature production benchmark/transfer; real non-sign/incomplete-sign labels and rejection operating curves; reliable timed gloss boundaries and NCSLGR signer identities; new temporal fingerspelling data eligibility; full raw-feature lineage; independent final evaluation and more signer clusters; full regression dependency/fixture restoration; deployment/distribution rights and representative Android/device/delegate tests. There is no outstanding approval to integrate or install an APK in this mission.

**Safest next action:** preserve these candidates unselected; obtain compatible current-path signer-disjoint streaming and consented negative/boundary data. Validate the modest isolated gain on a separately reserved benchmark before considering any adapter/integration. Further larger-model or cadence sweeps on25 gloss references are not justified. Every required deliverable exists in this new run, with blocked work stated explicitly.
''')
    # Portable artifact contract for the one exported candidate.
    write(run/'experiment_A/warmstart/export/contract.json',{'schema_version':'3.0','input':'float32 B,T,258 plus bool mask','length_bounds':[1,96],
        'output':'236 clip probabilities','vocabulary':[inverse[i] for i in range(236)],'model_sha256':export['onnx_sha256'],
        'input_validator':str(ROOT/'tools/asl_research_v2/input_contract.py'),'empty_or_nonfinite':'REJECT_INPUT','production_enabled':False,'Android_integrated':False,
        'non_sign_rejection_calibrated':False,'distribution_rights_cleared':False,'normalization':'existing canonical legacy258, no new fitted normalizer'})
    print(json.dumps({'reports_written':True,'A_top1':warm['top1'],'repeat_top1':repeat['top1'],'B_WER':b['gloss_wer'],'full_regressions':counts,'preservation':preservation['status']}))
if __name__=='__main__':main()
