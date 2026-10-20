"""Close the continuation package; no network, training or participant processing."""
from core import RUN, ROOT, TREE, PYTHON, read, sha, write
from pathlib import Path
from collections import Counter
import csv,json,shutil,subprocess

def textfile(name,text):
    with (RUN/name).open('x',encoding='utf-8') as f: f.write(text.rstrip()+'\n')

def main():
    status=read(RUN/'pilot_fresh48_v2/training_status.json')
    dev=read(RUN/'pilot_fresh48_v2/best_checkpoint_dev.json')
    train=read(RUN/'pilot_fresh48_v2/best_checkpoint_full_train.json')
    plan=read(RUN/'dataset_plan.json'); quality=read(RUN/'eligible_corpus/quality.json')
    goal=read(RUN/'goal_update.json'); probe=read(RUN/'vocab_cardinality_probe/results.json')
    receipt_bytes=sum(read(p)['bytes'] for p in RUN.rglob('*.receipt.json'))
    prior_bytes=4052597224
    token_counts=Counter(t for p in dev['predictions'] for t in p['hypothesis'])
    gpu=[]
    for line in (RUN/'gpu_20261020T185941_209.csv').read_text().splitlines():
        parts=line.split(',')
        try:gpu.append(int(parts[2].strip().split()[0]))
        except (ValueError,IndexError):pass
    preservation={}
    for directory,script in [('20261020T051631Z_devwer_diagnosis01','verify_outputs.py'),('20261020T102748Z_continuation_pilot01','verify_pilot.py')]:
        result=subprocess.run([str(PYTHON),'-B',str(TREE/directory/script)],capture_output=True,text=True)
        preservation[directory]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        if result.returncode:raise RuntimeError('FROZEN_PARENT_VERIFICATION_FAILED:'+directory+result.stdout+result.stderr)
    write(RUN/'parent_preservation_verification.json',preservation)
    write(RUN/'outcomes.json',{'active_goal':goal,'eligible_by_label_pool':quality['eligible_by_label_pool'],
        'training':status,'selected_dev_errors':{k:dev[k] for k in ['loss','substitutions','deletions','insertions','reference_tokens','empty_hypotheses']},
        'hypothesis_token_counts':dict(token_counts),'acquisition_bytes_this_phase':receipt_bytes,
        'acquisition_bytes_total_including_recorded_prior':prior_bytes+receipt_bytes,'cap_bytes':107374182400,
        'free_disk_bytes_at_close':shutil.disk_usage(ROOT).free,'sampled_device_peak_MiB':max(gpu) if gpu else None,
        'sampled_device_limitations':'200ms nvidia-smi samples; whole device usage, not a continuous peak; unknown utilization values retained in original CSV.',
        'readiness':'DATA_GOAL_PASS_RECOGNITION_PARTIAL_NO_PROMOTION','further_acquisition_authorized_by_this_script':False})
    with (RUN/'prediction_audit.csv').open('x',encoding='utf-8-sig',newline='') as f:
        names=['status','sample_id','split','reference','hypothesis','substitutions','deletions','insertions','reference_tokens']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for p in dev['predictions']:
            w.writerow({**{k:p[k] for k in names if k in p},'status':'MEASURED_RESEARCH_DEV','split':'dev','reference':json.dumps(p['reference'],ensure_ascii=False),'hypothesis':json.dumps(p['hypothesis'],ensure_ascii=False)})
    textfile('EXECUTION_SUMMARY.md',f'''# AppHear continuation outcome

**Data goal: PASS. Recognition readiness: PARTIAL. Production promotion: BLOCKED.**

The latest user instruction makes **2,000** the active data target. There are **3,121 verified eligible utterances**, including **96 newly admitted NCSLGR utterances**. Further acquisition has stopped. Earlier 5,000-target/1,879-shortfall fields in frozen intermediate artifacts are historical and superseded by `goal_update.json`; their bytes were preserved.

| Finding | Classification | Evidence and scope |
|---|---|---|
| Active 2,000-example target | PASS | 3,121 unique eligible IDs; 1,002 NCSLGR2 and 2,119 ASLLRP3, separate label pools |
| Additional authentic timed data | PASS | 96/98 deferred NCSLGR utterances; 2 overlapping intervals excluded without edits; two exact publisher EAF/media joins |
| Evaluation partition | PASS, scoped | 1,706 TRAIN / 413 DEV; three publisher TRAIN signers / one held-out signer; 39 / 8 source recordings; no ID/signer/source/raw/pixel hash intersection |
| Fresh CUDA pilot execution | PASS | Five completed epochs, saved/reloaded checkpoints, optimizer and RNG; RTX 3050; actual CUDA tensors and gradients |
| Recognition quality | PARTIAL | Selected epoch 4 TRAIN {100*train['gloss_wer']:.4f}% / DEV {100*dev['gloss_wer']:.4f}% WER; learning gate failed, run stopped |
| RIT timing admission | BLOCKED | 278 exact annotation/media joins, but every container has two extra frames and native 60 fps; no authoritative crop/frame-origin adapter; zero eligible |
| Hard CTC/token mapping defect | No evidence found | Prior 50 checks reused; both controlled eight-TRAIN-sample probes memorize; this does not prove production correctness |
| Dominant cause of poor output | PARTIAL hypothesis | 98.40% blank frames and {token_counts.get('IX-1p',0)}/283 emitted tokens IX-1p; frequent-token collapse/underfitting observed; precise optimization/data/feature cause unproven |
| Frozen artifacts | PASS | Both existing read-only validators passed initially and at closure; protected input hashes and 20,745 metadata records checked |
| Streaming false activation, boundaries, latency | BLOCKED for product claims | No new consented no-sign/streaming evaluation recordings; these utterance WER results cannot establish streaming behavior |
| Production readiness | BLOCKED | No checkpoint promoted; Android, thresholds, APK, GLB and reserved 17 TEST untouched |

The fit uses only the ASLLRP3 literal occurrence pool. The 3,121 count is an eligible catalog, **not** a merged training manifest. New 96 NCSLGR examples were not added to the already frozen pilot. Do not normalize away publisher label distinctions to improve scores.

All 413 new DEV utterances had historical TRAIN exposure in prior models. The new model was therefore initialized randomly, with new TRAIN-only vocabulary and normalization and no prior ASL recognition weights. This is a research development partition, not a pristine TEST or a multi-signer accuracy estimate. Its WER is not numerically comparable to the old 85.96% baseline on nine different NCSLGR DEV utterances.

`best_checkpoint_dev.json` retains the reused metric helper's generic `SIGNER UNVERIFIED` scope string. The actual new partition's publisher signer verification is documented in `signer_metadata.json`, `dataset_plan.json` and `EVALUATION_SPLIT_AUDIT.csv`; the saved prediction file was preserved rather than edited.

Acquisition accounting: {receipt_bytes:,} new receipt bytes; {prior_bytes+receipt_bytes:,} including recorded prior payload; cap 107,374,182,400 bytes. Extracted archive members are not counted as a second network acquisition. Disk free at closure is recorded in `outcomes.json`.

Read `PILOT_RESULTS.md`, `LABEL_COMPATIBILITY.md`, `DATA_PROGRESS.csv`, `EVALUATION_SPLIT_AUDIT.csv`, `prediction_audit.csv` and `commands_and_results.txt`. The checksum index covers closed new artifacts; its own checksum and itself are not recursively included. The read-only verifier checks the index and frozen parent evidence without training, acquisition or TEST inference.

Next action: verify this package using the command below. No automatic continuation of the failed pilot is scheduled. If further model research is requested, the smallest useful test is a predeclared intermediate-size **TRAIN-only** learnability probe (for example 64 diverse utterances), preserving literal labels and changing one optimization variable at a time; require meaningful TRAIN learning before another generalization run. Eight-sample memorization alone is insufficient. No more data is needed to meet the active goal.

```powershell
& '{RUN / 'Verify-Continuation.ps1'}'
```
''')
    textfile('LABEL_COMPATIBILITY.md','''# Label and temporal compatibility

| Pool | Status | Labels | Features / timing | Split limitations |
|---|---|---|---|---|
| NCSLGR2 literal main gloss | PASS for its existing utterance-recognition tasks; PARTIAL for universal boundary evaluation | Authentic publisher main gloss sequences; no English-to-gloss conversion | 378-dimensional causal schema. Legacy original records retain uncalibrated timing limitations; newer EAF/ms records use exact source bounds, 10 Hz native observations and reset causal deltas | 1,002 eligible; no independently verified cross-corpus signer identity map |
| ASLLRP3 literal dominant realized occurrence | PASS for scoped same-pool research WER | Literal occurrence labels, including hand, spatial, classifier, repetition and fingerspelling distinctions | Same 378-dimensional causal feature schema; exact publisher/native frame alignment from existing verified clips | 2,119 eligible; publisher participant metadata joined for all; new 1,706/413 signer/source-disjoint partition |
| RIT November 2025 CSV | BLOCKED for training admission | Authentic publisher event annotations; two invalid utterance-level cases and two overlapping cases excluded | 278 media joins and CRC-valid archive; 60 fps and two extra decoded frames per clip. No guessed origin, trimming, resampling or label edits | P02/P03 and four collections available; temporal adapter must be established before any fit |
| How2Sign existing package | BLOCKED for gloss recognition | English translations are not authentic ASL gloss labels | Prior acquisition evidence reused; no new acquisition | No gloss labels invented; not merged |

Sharing a feature dimension does not make NCSLGR main-gloss and ASLLRP realized-occurrence target ontologies interchangeable. A vocabulary union would preserve spelling but would not resolve annotation semantics or legacy boundary uncertainty. They remain separate pools; current fit uses ASLLRP3 only. No alias mapping was guessed.

New NCSLGR recovery used the unique basename in the authentic EAF MEDIA_DESCRIPTOR to select the matching member of each already published semicolon-separated URL field. HTTP was upgraded to HTTPS on the same official host. Original annotations, times and existing feature adapter were unchanged. Both newly found overlapping intervals were rejected. The previous entire-source overlap exclusion, six ASLLRP overlap exclusions and 100-vs-99 frame-count rejection remain excluded.

The RIT blocker needs publisher confirmation of master frame units, exact encoded clip origin and distribution of the two extra frames. Archive availability alone is insufficient. No data has been silently repaired to meet the count target.

Public research provenance and applicable noncommercial restrictions are retained in source receipts and the saved documents: https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html , https://dai.cs.rutgers.edu/asllvd/signbank/ , https://www.bu.edu/asllrp/dai-terms.html , https://www.bu.edu/asllrp/signbank-terms.pdf and https://www.bu.edu/asllrp/about-datasets.pdf . Public publisher participant queries establish signer IDs; no facial identity inference or account/login bypass was used. No participant recording, upload or redistribution was performed.
''')
    rows=list(csv.DictReader((RUN/'pilot_fresh48_v2/training_log.csv').open()))
    table='\n'.join(f"| {r['epoch']} | {float(r['train_optimization_loss']):.4f} | {100*float(r['train_monitor_wer']):.4f}% | {100*float(r['dev_wer']):.4f}% | {float(r['epoch_seconds']):.3f} | {r['dev_prediction_tokens']} |" for r in rows)
    examples='\n'.join(f"- `{p['sample_id']}`: reference `{json.dumps(p['reference'],ensure_ascii=False)}`; hypothesis `{json.dumps(p['hypothesis'],ensure_ascii=False)}`." for p in dev['predictions'][:6])
    textfile('PILOT_RESULTS.md',f'''# Measured fresh research pilot

**Execution PASS; recognition PARTIAL; no promotion.** Five CUDA epochs completed. The predeclared learning gate stopped the run; it was not extended after failure.

Data: frozen 1,706 TRAIN / 413 DEV manifest SHA256 `{plan['manifest_sha256']}`. Single target pool; three TRAIN publisher signers, one DEV signer. 2,517 literal TRAIN glosses plus blank ID 0 and UNK ID 1. 357/3,236 DEV tokens are out of vocabulary (11.03% irreducible exact-label coverage limitation with this fixed vocabulary). DEV reference labels remain literal, not replaced with UNK for scoring.

Architecture/config: causal 48-channel CTC, receptive field 61, 378-dimensional features; fresh random initialization; FP32, AdamW lr 0.001, weight decay 0.0001, batch 4, accumulation 2, gradient clip 5, seed 20261020. New TRAIN-only normalization and vocabulary. Bounds: 12 epochs / 180 seconds with a five-epoch learning window and saved optimizer/RNG state. See the exact gate in `pilot_fresh48_v2/config.json`.

| Epoch | Optimization loss | Fixed 64-TRAIN monitor WER | DEV WER | Epoch seconds | DEV emitted tokens |
|---|---|---|---|---|---|
{table}

Selected epoch 4: full TRAIN WER **{100*train['gloss_wer']:.4f}%**, loss **{train['loss']:.6f}**; DEV WER **{100*dev['gloss_wer']:.4f}%**, loss **{dev['loss']:.6f}**; 218 substitutions, 2,953 deletions, 0 insertions / 3,236 reference tokens. 131/413 hypotheses empty; 98.3956% argmax frames blank. Emitted token counts: `{dict(token_counts)}`. This is frequent-token collapse with limited learning, not working general recognition. Epoch 5 full TRAIN WER is recorded separately; it is not the selected epoch's TRAIN score.

Measured runtime: epoch loop {status['epoch_loop_seconds']:.3f}s; fit/reload/full-TRAIN diagnostics {status['fit_reload_full_train_diagnostics_seconds']:.3f}s; successful wrapper wall 100.971s including startup and logging. 8,530 TRAIN presentations; epoch-loop throughput {status['utterance_presentations']/status['epoch_loop_seconds']:.2f} utterance presentations/s (includes DEV/monitor evaluation, not pure forward throughput). Per-epoch durations above are measured, not estimates.

RTX 3050 Laptop 4 GiB: framework peak allocated {status['peak_allocated_vram_bytes']:,} bytes ({status['peak_allocated_vram_bytes']/1073741824:.3f} GiB), reserved {status['peak_reserved_vram_bytes']:,} bytes. Sampled whole-device memory peak {max(gpu) if gpu else 'UNKNOWN'} MiB, from 200 ms `nvidia-smi` logs; sampling can miss peaks. Unknown utilization values remain in the raw CSV. Actual input/logits/loss CUDA device and finite-gradient smoke are saved. Checkpoint reload reproduced selected DEV predictions.

Checkpoint: `{RUN / 'pilot_fresh48_v2/checkpoints/epoch_004.pt'}`

SHA256: `{status['best_checkpoint']['sha256']}`

Trainer SHA256: `{status['script_sha256']}`.

The first new driver failed during smoke logging after `zero_grad(set_to_none=True)`, before any optimizer update. V2 captures gradient device before clearing it. Failure and original script are retained in a different output directory; no completed fit was repeated or overwritten. This is a diagnostic logging fix, not evidence of a prior model bug.

Prediction examples, publisher literal conventions:

{examples}

All 413 reference/hypothesis pairs are in `prediction_audit.csv` and the selected DEV JSON. The generic inherited `SIGNER UNVERIFIED` metric scope string is superseded for partition provenance by the new publisher-backed split audit; it does not change measured predictions.

## Controlled TRAIN-only cardinality probe

Eight fixed new-TRAIN utterances from three TRAIN signers and eight source groups; same features, literal targets, TRAIN normalization, matching initialized backbone/common output rows. Only output vocabulary cardinality changed: 2,519 versus 62 classes. Both fresh fits memorized: full vocabulary 2.7778% TRAIN WER after 100 updates / 3.040s; restricted vocabulary 1.3889% after 90 updates / 2.458s. Saved predictions/checkpoints/curves verify these figures. No DEV or TEST, no new examples, no generalization claim. CUDA CTC nondeterminism and CuBLAS workspace warnings are retained; these small results are not a deterministic causal estimate for the full fit.

This argues against an absolute CTC/token-mapping inability to learn. It does not establish the cause of full-data underfitting or justify merging label pools. A larger TRAIN-only learnability test is the smallest next research step if requested. The active data goal is already satisfied.

The old baseline 85.96% and old main 100% use the original nine NCSLGR DEV utterances and cannot be ranked numerically against this new ASLLRP3 413-utterance DEV. No new TEST scoring or production selection occurred.
''')
    commands=[]
    for name in ['intake_metadata.py','signer_map.py','freeze_split.py','train_fresh.py','train_fresh_v2.py','vocab_cardinality_probe.py','rit_catalog.py','audit_rit_boundaries.py','rit_archive.py','prepare_rit.py','rit_timing_audit.py','resolve_ncslgr_urls.py','resolve_ncslgr_urls_v2.py','prepare_new_ncslgr.py','freeze_expanded_corpus.py']:
        commands.append(f"& '{PYTHON}' -B '{RUN/name}'")
    textfile('commands_and_results.txt',f'''CLOSED CONTINUATION PACKAGE — active goal 2000, achieved 3121; no further acquisition/training scheduled.
Repo: {ROOT}
Run: {RUN}

Initial existing validators: Verify-Diagnostics.ps1 PASS; Verify-Pilot.ps1 PASS.
Closure validators: both PASS through underlying Python checkers; exact captured stdout/stderr/exit codes in parent_preservation_verification.json. These rehash existing evidence; no 50-check diagnostic rerun or old training repeated. The first closure invocation via a new powershell -File process was denied by its unsigned-script execution policy before any checker ran; switched to Python without changing execution policy.

Executed script entry points (historical record, NOT an instruction to rerun completed work; CLI modes and exact successful fit wrapper follow):
{chr(10).join(commands)}

Exact successful training wrapper:
& '{RUN/'Run-PilotV2.ps1'}'
Exact underlying training command:
{read(RUN/'command_wall_20261020T185941_209.json')['command']}
Result: exit 0; wrapper 100.9710405 seconds; five actual epochs; TRAIN_BLANK_OR_NO_LEARNABILITY_GATE. Frozen configuration/checkpoints/logs in pilot_fresh48_v2; no pending fit.

New NCSLGR preparation:
$env:CUDA_VISIBLE_DEVICES='-1'
& '{PYTHON}' -B '{RUN/'prepare_new_ncslgr.py'}'
Result: 96 eligible, 0 processing failures; 152.5490514 seconds; exact unchanged adapter SHA b78ba79af96aaa5e0c06b79f787c62e2643fb46edba15fb307f5e8461da029d4.

Failures retained, not hidden:
- signer_metadata_console.txt: downloader returned None after successful download; fixed return path, verified existing receipts reused, successful v2 log retained.
- training_20261020T185838_461.log / command_wall_20261020T185838_461.json: exit 1 before optimizer update; gradient-device logging after clearing grads. V2 isolated fix and new output directory.
- rit_feature_pilot_console.txt: missing module search path, fixed new helper path; v2 reached strict FPS gate and rejected native60 vs expected29.97.
- rit2025_catalog/container_timing_audit.json and full_decode_probe.json: all278 clips have two extra frames; no guessed repairs. RIT eligible0.
- ncslgr_new_sources_console.txt: first resolver rejected actual overlapping intervals before download; v2 excludes both records, preserving original labels and media. v2 completed96.
- Earlier How2Sign RGB failed URL/error and other blocked-source acquisition evidence remain unchanged in parent artifacts. No retry or bypass in this closure.

Evidence files: dataset_plan.json; signer_metadata.json; data/processed/train_dev/manifest.json; eligible_corpus/manifest.json; DATA_PROGRESS.csv; all receipts; pilot_fresh48_v2/training_log.csv; training_status.json; gpu_20261020T185941_209.csv; vocab_cardinality_probe/results.json.
Reproduction outputs are exclusive/new-or-verified; completed runs must not be overwritten. Any future fitting needs a separate uniquely named research directory and predeclared protocol. Do not rerun acquisition commands above merely to validate.

NEXT READ-ONLY COMMAND:
& '{RUN/'Verify-Continuation.ps1'}'
No training/acquisition/TEST inference occurs in this validator. PASS means artifact integrity and scoped protocol checks, not recognition acceptance.
''')
    # Index only closed artifacts, with no active training/download/log writer.
    paths=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_hashes.sha256' and '__pycache__' not in p.parts and p.suffix!='.partial')
    textfile('artifact_hashes.sha256','\n'.join(f"{sha(p)}  {p.relative_to(RUN).as_posix()}" for p in paths))
    print(json.dumps({'status':'REPORTS_AND_CLOSED_ARTIFACT_INDEX_WRITTEN','indexed_files':len(paths),'eligible':3121,'active_target':2000,'training_status':status['status']}))

if __name__=='__main__':main()
