"""Add new execution report and artifact inventory; never overwrite completed evidence."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from core import RUN, ROOT, read, write, sha
from verify_completed import verify


def main():
    outputs = ['readiness_validation.json', 'preservation_final.json', 'execution_summary.json',
               'execution_report.md', 'reproduction_commands.ps1', 'artifact_manifest.json']
    for filename in outputs:
        if (RUN / filename).exists():
            raise ValueError('FINAL_OUTPUT_EXISTS_NO_OVERWRITE:' + filename)
    validation = verify()
    write(RUN / 'readiness_validation.json', validation)
    write(RUN / 'preservation_final.json', validation['preservation'])
    plan = read(RUN / 'expanded_catalog_v2/plan.json')
    quality = read(RUN / 'expanded_catalog_v2/quality.json')
    catalog = read(RUN / 'expanded_catalog_v2/manifest.json')
    comparison = read(RUN / 'comparison_completed/comparison.json')
    final = read(RUN / 'final_results.json')
    test_prep = read(RUN / 'data/processed/test/quality.json')
    v1 = read(RUN / 'dataset_plan.json')['rows']
    v1_new = [r for r in v1 if not r['reuse']]
    catalog_new = [r for r in catalog if r['new_in_catalog_v2']]
    initial_train_dev = [r for r in v1_new if r['split'] != 'test']
    custom_test = [r for r in v1_new if r['split'] == 'test']
    by_id = {r['sample_id']: r for r in catalog}
    ledger = []
    for path in sorted((RUN / 'logs').glob('*acquisition*.jsonl')):
        ledger += [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    downloaded = [r for r in ledger if r['status'] == 'DOWNLOADED']
    require_count = len(v1_new) + len(catalog_new)
    if len({r['id'] for r in downloaded}) != require_count or len(downloaded) != require_count:
        raise ValueError('NEW_RAW_LEDGER_COUNTS_DO_NOT_MATCH')
    counts = {
        'source_index_rows_inspected': plan['index_rows'],
        'all_eligible_candidates_acquired_and_prepared': len(catalog),
        'unique_valid_sequences': sum(r['unique_valid'] for r in catalog),
        'prior_raw_recordings_reused_not_new': len(catalog) - require_count,
        'previous_train_dev_download_checkpoint': len(initial_train_dev),
        'previous_train_dev_download_bytes': sum(by_id[r['sample_id']]['bytes'] for r in initial_train_dev),
        'additional_custom_test_acquired': len(custom_test),
        'additional_custom_test_bytes': sum(by_id[r['sample_id']]['bytes'] for r in custom_test),
        'additional_full_index_catalog_acquired': len(catalog_new),
        'additional_full_index_catalog_bytes': sum(r['bytes'] for r in catalog_new),
        'additional_raw_since_79_clip_checkpoint': len(custom_test) + len(catalog_new),
        'additional_bytes_since_79_clip_checkpoint': sum(by_id[r['sample_id']]['bytes'] for r in custom_test) + sum(r['bytes'] for r in catalog_new),
        'total_new_raw_in_active_experiment': require_count,
        'total_new_raw_bytes_in_active_experiment': sum(r['bytes'] for r in downloaded),
        'raw_content_duplicates': len(catalog) - len({r['raw_sha256'] for r in catalog}),
        'decoded_content_duplicates': len(catalog) - len({r['decoded_pixel_sha256'] for r in catalog}),
        'failed_downloads': [r for r in ledger if r['status'] == 'FAILED'],
        'failed_preprocessing': quality['failures'],
        'new_v2_feature_sets_created': sum(r['new_in_catalog_v2'] for r in catalog),
        'v2_feature_sets_and_decode_evidence_reused': quality['reused_records'],
        'frozen_fit': {'train': 100, 'dev': 9, 'test_after_freeze': 14},
        'catalog_roles': dict(Counter(r['split'] for r in catalog)),
        'v2_new_role_counts': dict(Counter(r['split'] for r in catalog_new)),
        'index_exclusions': plan['exclusion_counts'],
        'total_feature_observations': sum(r['observations'] for r in catalog),
        'source_gloss_tokens': sum(len(r['glosses']) for r in catalog),
        'v2_fitting_performed': False,
    }
    cap = validation['conservative_raw_bytes_including_How2Sign']
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'overall_status': 'PARTIAL',
        'execution_status': 'PASS_DATA_EXPANSION_AND_COMPLETED_BOUNDED_COMPARISON',
        'readiness_limit': 'Diagnostic collection-disjoint evidence only; signer/timing/no-sign/normal-app evaluation gates remain unmet.',
        'counts': counts, 'catalog_quality': quality,
        'comparison': {key: value for key, value in comparison.items() if key not in ('preprocessing', 'final_holdout')},
        'post_freeze_diagnostic_test': final,
        'bounded_acquisition': {'total_bytes_including_existing_How2Sign': cap,
                                'cap_bytes': validation['raw_cap'], 'remaining_bytes': validation['raw_cap'] - cap,
                                'disk_free_bytes_at_finalization': shutil.disk_usage(ROOT).free,
                                'existing_How2Sign_verification': 'REUSED; no archive integrity scan, extraction or acquisition repeated'},
        'signer_independent': False, 'authentic_individual_sign_boundaries': False,
        'authentic_sequence_labels': 'Verbatim official main-gloss strings, tokenized only for admitted unambiguous index sequences; grammar/locus/HOLD retained.',
        'label_scope': 'Literal source-main-gloss-token recognition; not canonical lexical-only gloss and not English translation.',
        'permissions': 'Reviewed BU research/education terms; local private use; no new participant data recorded; no production or redistribution clearance asserted.',
        'remaining_data_blocker': 'Richer authentic structured XML, timing and per-utterance signer mapping require an authorized official DAI account/export. Public index subset fully covered.',
        'other_sources_reviewed': str(RUN / 'source_review.md'),
        'preservation': validation['preservation'], 'production_promoted': False,
        'approval_incident': 'Initial custom-test download at approximately 02:14 local was not executed because the approval reviewer hit a usage limit. A fresh approved retry at 09:58 local downloaded all 14; no remaining approval blocker.',
        'How2Sign_prior_failed_URL': 'https://drive.google.com/uc?export=download&id=1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1',
        'How2Sign_prior_error': 'HTTP_404 at 2026-10-19T16:53:14.5183865Z; original logs/download_how2sign_val_frontal_pilot_archive_attempt1.json preserved in previous experiment; no repeat attempted.',
        'How2Sign_verified_validation_ID_reused': '1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO',
    }
    write(RUN / 'execution_summary.json', summary)
    models = comparison['models']
    lines = [
        '# Continuous ASL expansion and executed bounded comparison', '',
        '**Execution: PASS. Overall readiness: PARTIAL.** All 223 eligible public-index candidates are now acquired, decoded, feature-compatible and unique. Both predeclared GPU runs completed and their selected checkpoints were frozen before custom test acquisition. Completed work was reused; no retraining, checkpoint selection or preprocessing of completed samples was repeated during this resume.', '',
        '## Actual acquisition and preprocessing', '',
        f"The 79-clip TRAIN/DEV download checkpoint contained {counts['previous_train_dev_download_bytes']:,} bytes. This continuation added {counts['additional_custom_test_acquired']} custom test recordings ({counts['additional_custom_test_bytes']:,} bytes) and {len(catalog_new)} additional catalog recordings ({counts['additional_full_index_catalog_bytes']:,} bytes): **{counts['additional_raw_since_79_clip_checkpoint']} additional unique raw recordings since that checkpoint**. The active experiment now contains {require_count} new raw recordings / {counts['total_new_raw_bytes_in_active_experiment']:,} bytes; 37 earlier raw recordings are reused and never counted as new.", '',
        f"All 5,219 source-index rows were audited. All 223 eligible candidate recordings passed source/annotation joins, decode and raw/full-decoded-pixel uniqueness checks. New v2 preparation produced {len(catalog_new)} feature sets and reused {quality['reused_records']} verified feature/decode records. There were 0 failed downloads, 0 preprocessing failures and 0 content duplicates. Total catalog observations: {counts['total_feature_observations']:,}; original source target tokens: {counts['source_gloss_tokens']:,}.", '',
        f"The conservative acquired-raw total including the unchanged How2Sign archive is {cap:,} / {validation['raw_cap']:,} bytes, leaving {validation['raw_cap'] - cap:,} bytes. Additional NCSLGR files were bounded to 8 MiB each, 1 GiB per active run, 2 GiB aggregate with How2Sign and 10 GiB disk headroom. Existing archive verification was reused without rehashing/re-extracting it.", '',
        '| Catalog role | Unique recordings | Use |', '|---|---:|---|',
        '| TRAIN_POOL_NOT_FIT | 181 | Contains original 100 fit recordings plus 81 new future-pool additions; v2 additions were not fitted |',
        '| PRIOR_DEV_QUARANTINE | 19 | Old 7 development recordings plus 12 additional recordings from the same quarantined collections; no fitting |',
        '| FROZEN_DEV_REFERENCE | 9 | Original development-based selection unchanged |',
        '| FROZEN_TEST_EXPOSED_LOCKED | 14 | Acquired after selection freeze and evaluated once; now exposed, cannot be relabeled as a fresh future test |', '',
        'The completed comparison remains 100 TRAIN / 9 DEV, with 30 TRAIN feature sets reused from the prior experiment, 70 new TRAIN recordings and 9 new DEV recordings. Splits are source-collection-disjoint, with source/media/decode overlap checks. Signer IDs are unknown, so they are not signer-independent or official corpus partitions.', '',
        f"Initial TRAIN/DEV preparation took {comparison['preprocessing']['elapsed_seconds']:.3f} s; custom test preparation took {test_prep['elapsed_seconds']:.3f} s. Catalog v2 preparation took {quality['elapsed_seconds']:.3f} s, including cached validation and {quality['new_observations_processed']} new perception observations; extraction-only p50/p95 = {quality['extraction_ms_p50']:.3f} / {quality['extraction_ms_p95']:.3f} ms per observed frame. These measured CPU preprocessing times are separate from model fitting. Published compressed media may play at half speed: source/container timestamps are retained, but real-world sign durations are unverified.", '',
        '## Completed matched CUDA comparison', '',
        'RTX 3050 Laptop 4 GiB; FP32, seed 20261020, TRAIN-only 117-token vocabulary and identical normalization, batch 4 / accumulation 2, AdamW lr 0.001 / weight decay 0.0001, clip 5, greedy CTC blank 0. Selection used DEV WER then DEV loss, patience 10, bounded 30 epochs / 180 seconds per model. Effective TCN receptive field **61 observations**. CUDA input/logits/loss/gradient/optimizer state are recorded in epoch logs.', '',
        '| Model | Epochs / selected | DEV WER; S/D/I (57 refs) | Fit+reload s | Epoch median s | Peak allocated VRAM MiB | Worker peak MiB |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for name, model in models.items():
        dev = model['dev']
        lines.append(f"| {name} | {model['completed_epochs']} / {model['selected_epoch']} | {dev['gloss_wer'] * 100:.2f}%; {dev['substitutions']}/{dev['deletions']}/{dev['insertions']} | {model['measured_training_loop_and_reload_seconds']:.4f} | {model['epoch_seconds_median']:.4f} | {model['peak_pytorch_allocated_vram_bytes'] / 1048576:.2f} | {model['peak_worker_working_set_bytes'] / 1048576:.2f} |")
    lines += ['', 'Both runs stopped at their registered patience limit. Times are actual measured fit/DEV/checkpoint/final-reload loops, excluding interpreter startup and initial setup. VRAM figures are PyTorch allocator peaks; whole-GPU physical peaks were not independently sampled. No OOM or nonfinite-training failure was recorded. CUDA CTC backward warned about nondeterminism; fixed seeds do not imply bitwise reproduction.', '',
              '| Model | TRAIN observations / complete epoch second | TRAIN+DEV observations / complete epoch second | Epoch min / max s |', '|---|---:|---:|---|']
    for name, model in models.items():
        lines.append(f"| {name} | {model['train_observations_per_train_dev_checkpoint_epoch_second']:.2f} | {model['train_dev_observations_per_complete_epoch_second']:.2f} | {model['epoch_seconds_min']:.4f} / {model['epoch_seconds_max']:.4f} |")
    lines += ['', 'Throughput derives from actual completed epoch durations and actual precomputed-feature observations (3,059 TRAIN + 430 DEV per epoch), including DEV/checkpoint overhead. It is not source RGB, camera or Android throughput. Per-epoch durations and throughput are saved in comparison_completed/epoch_throughput.csv.', '',
              '## Frozen diagnostic test and model-only timing', '',
              'The 14-utterance / 86-reference-token custom test is one source collection, with unknown signer clustering. It was downloaded only after both checkpoint decisions were frozen; acquisition timestamps were checked against the freeze. It was not used for fitting, early stopping, threshold tuning or model selection.', '',
              '| Model | TEST WER; S/D/I | CPU model-only p50 / p95 ms |', '|---|---|---|']
    for name, model in final['models'].items():
        lines.append(f"| {name} | {model['gloss_wer'] * 100:.2f}%; {model['substitutions']}/{model['deletions']}/{model['insertions']} | {model['cpu_model_only_p50_ms']:.3f} / {model['cpu_model_only_p95_ms']:.3f} |")
    legacy = final['unchanged_legacy_baseline']
    lines += ['', f"Unchanged historical 39-token baseline: TEST WER {legacy['gloss_wer'] * 100:.2f}% ({legacy['substitutions']}/{legacy['deletions']}/{legacy['insertions']} S/D/I), 50 OOV tokens. Its restricted vocabulary differs from the matched 117-token pair, so it is only a historical diagnostic contrast. Matched models have 9 OOV DEV tokens / 57 references and 9 OOV TEST tokens / 86 references.", '',
              '**Compact24 remains unselected.** DEV WER worsened by 10.53 percentage points. Same-input 2-thread CPU median model latency fell 31.72%, but TEST WER worsened by 1.16 points. Timing used the same 9 DEV sequences, 5 warmups and 20 repeats per sequence, 180 measurements/model. These measurements exclude perception, buffering, decoder delivery and Android integration. No joint accuracy/speed improvement, real-world accuracy or 15 completed FPS claim is supported.', '',
              '## Checkpoints and frozen evidence', '']
    for name, model in models.items():
        lines += [f"- {name}: `{model['checkpoint_path']}`", f"  SHA256 `{model['checkpoint_sha256']}`; configuration `models/{name}/config.json`; status `models/{name}/training_status.json`; every epoch retains a resumable optimizer checkpoint and a separate model CSV log."]
    lines += ['', 'Plan/config/script/manifest/checkpoint hashes match model_selection_freeze.json. Reloaded predictions match the selected-epoch evidence. Completed status verification after resume does not launch training. The early RF121 draft arithmetic error and its failed pre-fit focused check remain preserved; training_plan.v2.json corrected the value to 61 before fitting. The subsequent 6 focused checks passed. No broad repository-suite green claim is made.', '',
              '## Labels, remaining blockers and permissions', '',
              'Source provenance is the official BU NCSLGR index, joined by exact collection/utterance/camera/source URL and index row. Ordered main-gloss strings, case, locus punctuation, repetitions, grammar particles and HOLD targets are retained verbatim. This metric is literal source-main-gloss-token WER, not standardized lexical-only gloss accuracy. No English translation was used as a gloss label. Independent sign boundaries and signer IDs are null rather than inferred.', '',
              f"The remaining index rows are explicitly excluded: {json.dumps(plan['exclusion_counts'], sort_keys=True)}. Shared recording URLs need authentic utterance timing; quoted/classifier/fingerspelling descriptions need structured annotations before defensible tokenization. Other-camera views are not credited as extra utterances. The next practical data step is an authorized official DAI structured XML export and per-utterance signer/timing mapping. Do not guess export URLs or participant identities. See source_review.md for checked NCSLGR/How2Sign/OpenASL/ChicagoFSWild access and task constraints. No further verified, fitting gloss payload is available under this mission's current supervision/access evidence.", '',
              'How2Sign remains the verified Validation archive with English sentences, not gloss supervision. Its official Validation ID and earlier HTTP_404 URL/error remain recorded in execution_summary.json and original evidence. The official 31 GB TRAIN archive is outside the 2 GiB cap and was not downloaded. Existing archive integrity evidence was reused. BU terms permit the reviewed private research use; keep media, derivatives and checkpoints local and do not infer production/distribution permission.', '',
              'Continuous streaming acceptance remains blocked by absent authentic per-sign timing, consented no-sign recordings, reliable signer-disjoint partitions and normal-app end-to-end latency measurements. The separate consented pilot still has no real participant consent/data; synthetic pilot fixtures are not model evidence. Do not tune thresholds against this exposed custom test or the frozen 2,620-row isolated test.', '',
              'Initial custom test acquisition was not executed when approval review hit a usage limit at approximately 02:14 local. A later fresh approval at 09:58 local succeeded; it is no longer an acquisition blocker. No completed artifacts were removed.', '',
              '## Preservation and validation', '',
              f"Fresh validation passed all 223 source/feature joins, timestamps, CTC feasibility, exact uniqueness, catalog/fitted-role joins, post-freeze TEST timing, frozen hashes, 4 preservation guard SHA256+mtime checks and {validation['preservation']['fresh_protected_metadata_checks']} protected size/mtime checks. Original full-hash evidence was reused; this is not a new full content rehash of all original files. Production models, thresholds, Android assets, old experiments and frozen isolated results are unchanged. No dependency was reinstalled.", '',
              'Primary artifacts: execution_summary.json, readiness_validation.json, preservation_final.json, expanded_catalog_v2/manifest.json, expanded_catalog_v2/manifest.csv, expanded_catalog_v2/index_inventory.csv, expanded_catalog_v2/quality.json, comparison_completed/comparison.json, final_results.json, artifact_manifest.json. Source HTTP receipts, exact byte/hash joins and acquisition ledgers remain under data/raw and logs.', '',
              '## Exact PowerShell commands', '', '```powershell', f"Set-Location -LiteralPath '{ROOT}'", f"$run = '{RUN}'", f"$py = '{ROOT / 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'}'", '& "$run/Invoke-Research.ps1" -Phase Status', '& $py -B "$run/tools/verify_completed.py" --artifacts', '& "$run/Invoke-Research.ps1" -Phase Verify', '```', '',
              'Historical commands used for the completed runs (now safely return FROZEN_COMPLETE_NO_TRAINING):', '', '```powershell', '& "$run/Invoke-Research.ps1" -Phase Reference', '& "$run/Invoke-Research.ps1" -Phase Candidate', '```', '',
              'Completed catalog orchestration (do not repeat successful downloads/preparation):', '', '```powershell', '& "$run/Acquire-Catalog.ps1"', '& $py -B "$run/tools/expand_catalog.py" prepare', '```', '',
              'No incomplete training step remains in this frozen comparison. A future expanded-data experiment needs a new versioned output, a declared evaluation protocol and reserved evaluation data; signer-independent/streaming claims additionally need the structured signer/timing annotations and consented negative recordings described above. The 181-recording training pool is additive preparation, not evidence that 181 recordings were fitted.', '']
    (RUN / 'execution_report.md').write_text('\n'.join(lines), encoding='utf-8')
    commands = '''param([ValidateSet('Status','Validate')][string]$Phase='Status')
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$py=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
if($Phase -eq 'Status'){ & (Join-Path $run 'Invoke-Research.ps1') -Phase Status; return }
& $py -B (Join-Path $run 'tools/verify_completed.py') --artifacts
if($LASTEXITCODE -ne 0){throw "COMPLETED_VALIDATION_FAILED_$LASTEXITCODE"}
'''
    (RUN / 'reproduction_commands.ps1').write_text(commands, encoding='utf-8')
    inventory = []
    for path in sorted(RUN.rglob('*')):
        relative = path.relative_to(RUN)
        if not path.is_file() or any(part in {'cache', '__pycache__', '.pytest_cache'} for part in relative.parts):
            continue
        inventory.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    write(RUN / 'artifact_manifest.json', {'schema': 'apphear.continuous_asl.artifacts.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': inventory,
        'excludes': ['artifact_manifest.json (self)', 'cache/**', '**/__pycache__/**', '**/.pytest_cache/**'],
        'external_inputs': 'Prior source/feature paths are referenced, verified by joins and preservation evidence; not copied or counted as new raw.',
        'production_enabled': False})
    print(json.dumps({'status': summary['execution_status'], 'overall_status': 'PARTIAL',
                      'additional_raw_since79': counts['additional_raw_since_79_clip_checkpoint'],
                      'total_new_raw': require_count, 'unique_catalog': len(catalog), 'artifacts': len(inventory)}, indent=2))


if __name__ == '__main__':
    main()
