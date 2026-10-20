"""Finalize executed expansion/training evidence, with read-only preservation checks."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from core import RUN, ROOT, SOURCE, ACQUIRED, read, write, sha


def preservation():
    baseline = read(ACQUIRED / 'preservation_baseline.json')
    issues = []
    source_inventory = Path(baseline['source_artifact_manifest'])
    if sha(source_inventory) != baseline['source_artifact_manifest_sha256'] or source_inventory.stat().st_mtime_ns != baseline['source_artifact_manifest_mtime_ns']:
        issues.append({'path': str(source_inventory), 'reason': 'SOURCE_INVENTORY_CHANGED'})
    hashed = 0
    for row in read(source_inventory)['files']:
        path = SOURCE / row['path']
        if not path.is_file() or sha(path) != row['sha256']:
            issues.append({'path': str(path), 'reason': 'FROZEN_EXPAND03_ARTIFACT_HASH'})
        hashed += 1
    for row in baseline['metadata_snapshot']:
        path = Path(row['path'])
        if not path.is_file() or path.stat().st_size != row['size'] or path.stat().st_mtime_ns != row['mtime_ns']:
            issues.append({'path': str(path), 'reason': 'PROTECTED_SIZE_OR_MTIME'})
    guards = read(SOURCE / 'preservation_baseline.json')['guards']
    for row in guards:
        path = Path(row['path'])
        if sha(path) != row['sha256'] or path.stat().st_mtime_ns != row['mtime_ns']:
            issues.append({'path': str(path), 'reason': 'PRIOR_EVIDENCE_GUARD'})
    for root in (ACQUIRED, RUN):
        frozen = read(root / 'model_selection_freeze.json')
        if sha(root / 'data/processed/train_dev/manifest.json') != frozen['manifest_sha256'] or sha(root / 'tools/train_expanded.py') != frozen['trainer_sha256']:
            issues.append({'path': str(root), 'reason': 'NEW_FROZEN_TRAINER_OR_MANIFEST'})
        for name, model in frozen['models'].items():
            base = root / 'models' / name
            for relative, expected in (('training_status.json', model['status_sha256']), ('config.json', model['config_sha256']),
                                       (model['pointer']['path'], model['pointer']['sha256'])):
                if sha(base / relative) != expected:
                    issues.append({'path': str(base / relative), 'reason': 'NEW_FROZEN_MODEL'})
    if issues:
        raise ValueError('PRESERVATION_FAILURE:' + json.dumps(issues))
    return {'status': 'PASS', 'fresh_expand03_content_hashes': hashed, 'fresh_guard_hashes_and_mtimes': len(guards),
            'protected_metadata_checks': len(baseline['metadata_snapshot']), 'failures': issues,
            'scope': 'Fresh hashes of expand03 artifact inventory and all its listed files; four prior evidence guard hashes; protected size/mtime checks. Existing full original-content/How2Sign integrity evidence reused without archive rescan.'}


def dataset_statistics(rows):
    output = {}
    for split in sorted({r['split'] for r in rows}):
        selected = [r for r in rows if r['split'] == split]
        spans = []
        for row in selected:
            with np.load(row['feature_path'], allow_pickle=False) as f:
                ts = f['timestamp_ms']
            spans.append(float(ts[-1] - ts[0]) / 1000)
        tokens = [token for row in selected for token in row['glosses']]
        output[split] = {'utterances': len(selected), 'observations': sum(r['observations'] for r in selected),
            'gloss_tokens': len(tokens), 'unique_gloss_labels': len(set(tokens)),
            'fingerspelled_word_target_tokens': sum(t.startswith('fs-') for t in tokens),
            'container_duration_from_decoded_frames_over_native_fps_seconds': sum(r['decoded_frames'] / r['native_fps'] for r in selected),
            'observed_timestamp_span_seconds': sum(spans), 'known_signer_rows': sum(r.get('signer_id') is not None for r in selected),
            'known_signer_ids': sorted({r['signer_id'] for r in selected if r.get('signer_id') is not None})}
    return output


def inventory(root):
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or path.name == 'artifact_manifest.json' or any(p in ('cache', '__pycache__', '.pytest_cache') for p in relative.parts):
            continue
        files.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    write(root / 'artifact_manifest.json', {'schema': 'apphear.continuous_asl.artifacts.v1', 'files': files,
          'created_utc': datetime.now(timezone.utc).isoformat(),
          'excludes': ['self', 'cache/**', '**/__pycache__/**', '**/.pytest_cache/**'], 'production_enabled': False})
    return len(files)


def main():
    if (RUN / 'execution_summary.json').exists() or (RUN / 'artifact_manifest.json').exists():
        raise ValueError('MISSION_REPORT_ALREADY_EXISTS_NO_OVERWRITE')
    protected = preservation()
    catalog = read(ACQUIRED / 'full_index_v2/preparation/manifest.json')
    quality = read(ACQUIRED / 'full_index_v2/preparation/quality.json')
    plan = read(ACQUIRED / 'full_index_v2/plan.json')
    fit = read(RUN / 'data/processed/train_dev/manifest.json')
    test = read(RUN / 'data/reserved_test/manifest.json')
    stats = dataset_statistics(fit + test)
    all_stats = dataset_statistics([r for r in catalog if r['unique_valid']])
    four = read(ACQUIRED / 'comparison_dev.json')
    five = read(RUN / 'comparison_dev.json')
    final = read(RUN / 'final_reserved_test.json')
    raw_paths = list((ACQUIRED / 'data/raw').rglob('*.mov'))
    receipts = [read(str(path) + '.receipt.json') for path in raw_paths]
    raw_unique = len({r['sha256'] for r in receipts})
    payloads = [p for p in (ACQUIRED / 'data/raw').rglob('*') if p.is_file() and not p.name.endswith(('.receipt.json', '.http.json'))]
    payload_bytes = sum(p.stat().st_size for p in payloads)
    acquired_total = 1895291702 + payload_bytes
    if acquired_total > 2147483648:
        raise ValueError('GLOBAL_ACQUISITION_CAP')
    failures = [read(path) for path in sorted((ACQUIRED / 'logs').glob('download_*_attempt*.json'))]
    gpu = []
    for path in sorted((RUN / 'logs').glob('gpu_samples_*.csv')):
        for row in csv.reader(path.open()):
            if len(row) == 5:
                try:
                    gpu.append({'timestamp': row[0].strip(), 'name': row[1].strip(), 'total_mib': float(row[2]), 'used_mib': float(row[3]), 'utilization_percent': float(row[4])})
                except ValueError:
                    continue
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'execution_status': 'PASS', 'readiness': 'PARTIAL',
        'new_raw_files': len(raw_paths), 'new_unique_raw_sha256': raw_unique, 'new_raw_bytes': sum(r['bytes'] for r in receipts),
        'new_usable_unique_utterances': sum(r['new_raw_required'] and r['unique_valid'] for r in catalog),
        'total_usable_unique_utterances': quality['unique_valid'], 'previous223_features_reused': quality['reused_records'],
        'source_candidate_count': len(plan['rows']), 'processing_quality': quality, 'fitted_statistics': stats, 'all_catalog_statistics': all_stats,
        'new181fit_comparison': four, 'new_broader_fit_comparison': five, 'post_freeze_new_test': final,
        'quarantined_records': read(RUN / 'data/quarantined.json'), 'download_failures': failures,
        'acquired_payload_bytes_in_this_continuation': payload_bytes, 'aggregate_payload_bytes_including_prior_How2Sign': acquired_total,
        'aggregate_cap_bytes': 2147483648, 'remaining_bytes': 2147483648 - acquired_total,
        'gpu_external_sampling': {'interval_ms': 200, 'samples': len(gpu),
            'maximum_sampled_used_mib': max((r['used_mib'] for r in gpu), default=None),
            'maximum_sampled_utilization_percent': max((r['utilization_percent'] for r in gpu), default=None),
            'scope': 'Whole-GPU sampled occupancy/utilization, may include desktop/driver/context;200ms sampling can miss brief peaks. PyTorch allocator peaks also saved per model.'},
        'preservation': protected, 'production_promoted': False, 'signer_independent': False,
        'new_reserved_test_now_exposed': True, 'old14_and_isolated2620_results_unchanged': True,
        'real_sign_duration_or_streaming_boundary_truth': 'NOT_VERIFIED; retain original source coordinates without conversion',
        'normal_app_latency': 'NOT_MEASURED', 'phone_fps': 'NOT_MEASURED',
        'remaining_blockers': ['Remaining official-index records need calibrated utterance boundaries or unambiguous structured event labels.',
            'Cokely official EAF links returnedHTTP403; no video downloaded or access restriction bypassed.',
            'Purdue RVL-SLLL requires signed agreement and issued credentials; none fabricated or submitted.',
            'DAI structured XML/participant exports require authorized access.',
            'Signer-independent streaming evaluation still needs complete signer mapping, authentic timed boundaries and consented no-sign recordings.'],
        'remaining_staged_work': 'NONE; all completed bounded fits/checkpoints/dev and reserved-test evaluation saved.',
        'disk_free_bytes': shutil.disk_usage(ROOT).free}
    write(RUN / 'preservation_final.json', protected)
    write(RUN / 'execution_summary.json', summary)
    write_reports(summary, stats, four, five, final, plan)
    a = inventory(ACQUIRED)
    b = inventory(RUN)
    print(json.dumps({'status': 'PASS_EXECUTED', 'readiness': 'PARTIAL', 'new_unique_raw': raw_unique,
          'new_usable': summary['new_usable_unique_utterances'], 'total_usable': quality['unique_valid'],
          'fit_counts': {k: v['utterances'] for k, v in stats.items()}, 'artifacts04': a, 'artifacts05': b}))


def write_reports(summary, stats, four, five, final, plan):
    # Filled from measured artifacts only, after all fitting and post-freeze evaluation complete.
    lines = ['# Executed continuous ASL expansion and model comparison', '',
        '**Execution PASS; overall readiness PARTIAL.** All reported training is backed by epoch logs, hash-verified checkpoints and matching reloaded predictions. No production model or application integration changed.', '',
        f"Added **{summary['new_unique_raw_sha256']} new unique raw recordings** ({summary['new_raw_bytes']:,} bytes); {summary['new_usable_unique_utterances']} new utterances passed feature/duplicate/CTC checks. Total usable catalog: {summary['total_usable_unique_utterances']}, including 223 reused recordings/features that are not counted as new. Newly extracted features and timings are under expansion04; staged fitting in expansion05 references those files without copying or reprocessing them.", '',
        f"The original expansion03 100/9 comparison remains complete and unchanged. A separate 181/9 comparison completed in expansion04, followed by the new broader fit in expansion05. Both use the same original nine DEV recordings and fixed greedy CTC evaluator. Final fit counts: {stats['train']['utterances']} TRAIN, {stats['dev']['utterances']} DEV, {stats['test']['utterances']} newly reserved TEST. Quarantined records: {len(summary['quarantined_records'])}; no old exposed14 or isolated2620 result was modified.", '',
        '## Data and label evidence', '',
        '[Official BU NCSLGR index/source](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html); [BU research terms](https://www.bu.edu/asllrp/dai-terms.html). The new versioned target policy retains authentic fingerspelled-word glosses such as fs-JOHN as complete source word labels. It does not create letter sequences or substitute English captions. Case, punctuation, locus notation, source index rows, source URLs and checksums are preserved. Multiword/quoted labels are admitted only with exact matching structured event evidence.', '',
        'Official older BU display pages yielded166 SignStream v5 exports, documented by the [publisher guide](https://www.bu.edu/asllrp/asllrpr9.pdf). Strict parsing recovered17 newly eligible recordings and14 existing metadata joins;135 exports were rejected for ambiguous event/time representation, invalid triples or annotation-version conflicts. Those17 new recordings are included in the broader total, never counted twice. Index and legacy utterance IDs are kept separately; media+collection+full-gloss joins avoid assuming an ID offset. Original timing integers and signer names/IDs are retained where present. Legacy coordinates are uncalibrated to current media, so no per-sign millisecond or boundary-accuracy claim is made.', '',
        '| Partition | Utterances | Observations | Source gloss tokens | Distinct labels | Container duration estimate s |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in stats.items():
        lines.append(f"| {name} | {row['utterances']} | {row['observations']} | {row['gloss_tokens']} | {row['unique_gloss_labels']} | {row['container_duration_from_decoded_frames_over_native_fps_seconds']:.3f} |")
    lines += ['', 'Duration estimates use full decoded frame counts / native container FPS; observed timestamp spans are saved separately. The publisher warns some compressed media play at half speed. These are not independently verified real signing durations. Every accepted feature file has schema apphear.continuous_asl.candidate01_causal378.v1,378float32 features, finite values, strictly increasing genuine observation timestamps and feasible CTC target length. Source/group/raw/full-decoded-pixel overlap was checked; invalid candidates were rejected, not repaired.', '',
        'Training and DEV are collection-disjoint. New TEST recordings have distinct source groups, raw and decoded content, but share historical10d/10q source collections and have incomplete signer mapping. Known signer metadata does not establish complete signer-disjointness. The new TEST was reserved before fitting, prepared separately and scored only after model decisions froze. It is now exposed and cannot serve as a fresh future test. No signer-independent, end-to-end or production accuracy claim is made.', '',
        '## Measured development comparison', '',
        '| Regime / model | TRAIN | Epochs / selected | DEV WER; S/D/I | Fit+reload s | Median epoch s | Peak allocated VRAM MiB |',
        '|---|---:|---:|---|---:|---:|---:|']
    baseline = five['frozen_expand03_reference48']
    lines.append(f"| frozen expand03 reference48 | 100 | 14 / 4 | {baseline['gloss_wer']*100:.2f}%; {baseline['substitutions']}/{baseline['deletions']}/{baseline['insertions']} | 5.7732 | 0.3803 | 68.08 |")
    checkpoints = []
    for label, result in (('expand04', four), ('expand05', five)):
        for name, model in result['models'].items():
            lines.append(f"| {label} {name} | {result['train']} | {model['epochs']} / {model['selected_epoch']} | {model['gloss_wer']*100:.2f}%; {model['substitutions']}/{model['deletions']}/{model['insertions']} | {model['measured_fit_reload_seconds']:.4f} | {model['epoch_median_seconds']:.4f} | {model['peak_pytorch_allocated_vram_bytes']/1048576:.2f} |")
    lines += ['', 'All DEV scores use57 original reference tokens. The pair within each new regime has identical TRAIN partitions, vocabulary and normalization. Vocabulary/normalization are refit on TRAIN when adding data, so historical comparisons are expanded-training-regime contrasts rather than architecture-only ablations. WER and loss, not training loss alone, determine checkpoints. Reported time includes fitting, DEV, checkpoints and reload, excluding interpreter/startup setup. Command wall times for the broader fit are separately logged.', '',
        '| Regime / model | Vocab | DEV OOV tokens | TRAIN observations / complete epoch s | CPU model p50 / p95 ms | Checkpoint SHA256 |',
        '|---|---:|---:|---:|---|---|']
    for label, result in (('expand04', four), ('expand05', five)):
        for name, model in result['models'].items():
            lines.append(f"| {label} {name} | {model['vocabulary_size']} | {model['out_of_vocabulary_tokens']} | {model['train_observations_per_complete_epoch_second']:.2f} | {model['cpu_model_p50_ms']:.3f} / {model['cpu_model_p95_ms']:.3f} | {model['checkpoint_sha256']} |")
            checkpoints.append(f"- {label} {name}: `{model['checkpoint_path']}`")
    lines += [''] + checkpoints
    lines += ['', f"Freshly timed frozen reference48 CPU model p50/p95: {baseline['cpu_model_p50_ms']:.3f}/{baseline['cpu_model_p95_ms']:.3f}ms. Timing uses identical9DEV feature sequences,2CPU threads,5warmups/20repeats per sequence,180measurements/model. It excludes decoding/perception/buffering/application delivery. Throughput uses actual completed epoch wall times including DEV/checkpoint work; it is not camera or phone FPS.", '',
        '## Post-freeze reserved test', '', '| Model | TEST WER | S/D/I | Reference tokens | OOV tokens |', '|---|---:|---|---:|---:|']
    for name, row in final['models'].items():
        lines.append(f"| {name} | {row['gloss_wer']*100:.2f}% | {row['substitutions']}/{row['deletions']}/{row['insertions']} | {row['reference_tokens']} | {row['out_of_vocabulary_tokens']} |")
    lines += ['', f"Selected within expansion05 by DEV only: {five['selected_within_new_pair']}. Beats the frozen reference on DEV: {five['beats_frozen_baseline']}. TEST did not change this decision. Descriptive corpus diagnostics do not establish current-app performance. No experimental model was promoted.", '',
        '## Resources, failures and stopping condition', '',
        f"Acquired payload total including the previous How2Sign Validation archive: {summary['aggregate_payload_bytes_including_prior_How2Sign']:,}/{summary['aggregate_cap_bytes']:,}bytes; remaining {summary['remaining_bytes']:,}bytes. Existing How2Sign CRC/hash/decode evidence was reused, and its English translations were not used as gloss labels. New feature preparation took {summary['processing_quality']['elapsed_seconds']:.3f}s; extraction-only frame p50/p95 {summary['processing_quality']['extraction_ms_p50']:.3f}/{summary['processing_quality']['extraction_ms_p95']:.3f}ms, separate from fitting.", '',
        f"GPU sampling: {json.dumps(summary['gpu_external_sampling'])}. Per-batch free-memory guards and allocator peaks are logged. No repeated OOM experiment was attempted. Fixed seeds were used; CUDA CTC backward nondeterminism warnings remain recorded. Python console-destructor WinError0xc0000008 warnings occurred after successful work/zero exits; checkpoints were reloaded and validated. One initialization status-print NameError was corrected; generated manifests were hash-verified and reused, with the original failure preserved in expansion04/logs/initialization_print_failure.json.", '',
        '[Cokely Parallel Corpus](https://encompass.eku.edu/cokely_videos/) explicitly offers genuine ELAN ID-gloss data under CC BY-NC-SA4.0, but all three attempted official EAF URLs returnedHTTP403. Exact URLs/errors remain in expansion04/logs; no videos were acquired and no restriction was bypassed. [Purdue RVL-SLLL](https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm) requires a signed agreement and issued credentials. No agreement was signed/submitted and no messages were sent. Previously reviewed OpenASL/How2Sign are English-translation tasks, and ChicagoFSWild is letter transcription; none was relabeled as sentence-gloss data.', '',
        f"Full-index remaining exclusions: {json.dumps(plan['exclusion_counts'], sort_keys=True)}. All eligible recordings under the documented label/access policy were attempted. The next data step is authorized structured annotation access with calibrated utterance timings, or restored official Cokely downloads. Streaming/signer-independent claims additionally require complete signer mapping and consented negative recordings. These blockers do not undo the completed bounded training.", '',
        f"Preservation PASS: {summary['preservation']['fresh_expand03_content_hashes']} original expansion03 artifact content hashes, four prior evidence guards and {summary['preservation']['protected_metadata_checks']} protected metadata checks. Existing full original-content evidence and How2Sign integrity verification were reused. Android/GLB/application data, thresholds, production selection, frozen isolated2620 results, prior checkpoints, manifests and reports remain unchanged. No dependencies were reinstalled.", '',
        '## Exact PowerShell commands', '', '```powershell', f"$run = '{RUN}'", f"$py = '{ROOT / 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'}'", '& "$run/Validate-Completed.ps1"', '& "$run/Invoke-Research.ps1" -Phase Status', '```', '',
        'Historical executed training commands (the wrapper verifies completed checkpoints on repeat):', '', '```powershell', f"& '{ACQUIRED / 'Invoke-Research.ps1'}' -Phase Tests", f"& '{ACQUIRED / 'Invoke-Research.ps1'}' -Phase Smoke", f"& '{ACQUIRED / 'Invoke-Research.ps1'}' -Phase Reference", f"& '{ACQUIRED / 'Invoke-Research.ps1'}' -Phase Candidate", '& "$run/Run-Bounded.ps1"', '& $py -B "$run/tools/evaluate_reserved.py"', '```', '',
        'No incomplete optimizer stage remains. In an interruption before completion, Invoke-Research.ps1 -Phase Reference -Resume or -Phase Candidate -Resume restores the last complete epoch and optimizer/RNG state. Further training belongs in a new versioned experiment; do not mutate these frozen selections or tune against either exposed test set.', '']
    (RUN / 'execution_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (ACQUIRED / 'execution_report.md').write_text('# Expansion04 execution evidence\n\nInitial181/9 CUDA comparison completed and frozen; comparison_dev.json contains measured WER, resources and checkpoint hashes.\n\nAdditional official-index acquisition and feature preparation are under full_index_v2. Complete combined execution report and immutable inventories are in expansion05:\n\n' + str(RUN / 'execution_report.md') + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
