"""Freeze expanded181 selections and compare on original9DEV only. No TEST reads."""
import csv
import json
import statistics
import time
import numpy as np
import torch
from core import RUN, ROOT, SOURCE, read, write, sha
from train_expanded import dataset
from ctc_experiment import CausalCTC, evaluate, seeds


def measured_model(base, pointer, rows, data):
    checkpoint = base / pointer['path']
    if sha(checkpoint) != pointer['sha256']:
        raise ValueError('CHECKPOINT_HASH')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = CausalCTC(len(state['vocabulary']), state['config']['channels'], state['model']['mean'].numpy(), state['model']['std'].numpy())
    model.load_state_dict(state['model'])
    result = evaluate(model, rows, data, state['vocabulary'], 'cpu')
    durations = []
    with torch.inference_mode():
        for row in rows:
            x = torch.as_tensor(data[row['sample_id']][None])
            for _ in range(5):
                model(x)
            for _ in range(20):
                start = time.perf_counter()
                model(x)
                durations.append((time.perf_counter() - start) * 1000)
    result.update(cpu_model_p50_ms=float(np.percentile(durations, 50)), cpu_model_p95_ms=float(np.percentile(durations, 95)),
                  timing_n=len(durations), cpu_threads=torch.get_num_threads(), checkpoint_path=str(checkpoint), checkpoint_sha256=pointer['sha256'],
                  vocabulary_size=len(state['vocabulary']), parameters=sum(p.numel() for p in model.parameters()))
    return result, state


def main():
    if (RUN / 'comparison_dev.json').exists():
        print('COMPARISON_ALREADY_COMPLETE_NO_REPEAT')
        return
    seeds(20261020)
    rows, data = dataset()
    dev = [r for r in rows if r['split'] == 'dev']
    fit = [r for r in rows if r['split'] == 'train']
    frozen = {'manifest_sha256': sha(RUN / 'data/processed/train_dev/manifest.json'),
              'plan_sha256': sha(RUN / 'training_plan.v2.json'), 'trainer_sha256': sha(RUN / 'tools/train_expanded.py'),
              'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              'selection': 'DEV WER then loss; no final test accessed', 'models': {}}
    for name in ('reference48', 'compact24'):
        base = RUN / 'models' / name
        status = read(base / 'training_status.json')
        pointer = read(base / 'checkpoints/best.json')
        if status['script_sha256'] != frozen['trainer_sha256'] or status['manifest_sha256'] != frozen['manifest_sha256']:
            raise ValueError('FITTING_CONTRACT_CHANGED')
        frozen['models'][name] = {'pointer': pointer, 'status_sha256': sha(base / 'training_status.json'), 'config_sha256': sha(base / 'config.json')}
    if not (RUN / 'model_selection_freeze.json').exists():
        write(RUN / 'model_selection_freeze.json', frozen)
    output, states = {}, {}
    for name, info in frozen['models'].items():
        base = RUN / 'models' / name
        scores, states[name] = measured_model(base, info['pointer'], dev, data)
        if scores['predictions'] != read(base / 'dev_best.json')['predictions']:
            raise ValueError('CPU_GPU_DEVELOPMENT_PARITY')
        status = read(base / 'training_status.json')
        epochs = list(csv.DictReader((base / 'training_log.csv').open(newline='')))
        if len(epochs) != status['completed_epochs']:
            raise ValueError('EPOCH_LOG_COUNT')
        times = [float(r['epoch_seconds']) for r in epochs]
        scores.update(epochs=status['completed_epochs'], selected_epoch=info['pointer']['epoch'], status=status['status'],
                      measured_fit_reload_seconds=status['seconds'], epoch_median_seconds=statistics.median(times),
                      epoch_min_seconds=min(times), epoch_max_seconds=max(times),
                      train_observations_per_complete_epoch_second=sum(len(data[r['sample_id']]) for r in fit) * len(times) / sum(times),
                      peak_pytorch_allocated_vram_bytes=status['peak_allocated_vram_bytes'],
                      peak_pytorch_reserved_vram_bytes=status['peak_reserved_vram_bytes'],
                      peak_worker_working_set_bytes=status['worker_memory']['peak_working_set_bytes'],
                      final_training_ctc_loss=float(epochs[-1]['train_ctc_loss']), no_test_access=not status['test_accessed'])
        output[name] = scores
    for key in ('mean', 'std'):
        if not torch.equal(states['reference48']['model'][key], states['compact24']['model'][key]):
            raise ValueError('MATCHED_NORMALIZATION')
    if states['reference48']['vocabulary'] != states['compact24']['vocabulary']:
        raise ValueError('MATCHED_VOCABULARY')
    baseline_pointer = read(SOURCE / 'models/reference48/checkpoints/best.json')
    baseline, _ = measured_model(SOURCE / 'models/reference48', baseline_pointer, dev, data)
    if baseline['predictions'] != read(SOURCE / 'models/reference48/dev_best.json')['predictions']:
        raise ValueError('FROZEN_BASELINE_PARITY')
    result = {'status': 'PASS_EXECUTED_DEV_COMPARISON', 'models': output, 'frozen_expand03_reference48': baseline,
              'train': len(fit), 'dev': len(dev), 'test_evaluated': 0, 'signer_independent': False,
              'fairness': 'Same9DEV, raw targets, schema, decoder and evaluation code; new matched pair shares181TRAIN/vocab/normalization. Historical100TRAIN baseline has117rather than143tokens, so it is an expanded-regime contrast, not architecture-only.',
              'normal_app_latency': 'NOT_MEASURED', 'timing_scope': '2CPU threads, same9DEV feature sequences,5warmup+20repeat/model/sequence; excludes preprocessing and delivery',
              'receptive_field': 61, 'production_promoted': False,
              'selected_within_new_pair': min(output, key=lambda name: (output[name]['gloss_wer'], output[name]['loss'])),
              'beats_frozen_baseline': min(m['gloss_wer'] for m in output.values()) < baseline['gloss_wer']}
    write(RUN / 'comparison_dev.json', result)
    print(json.dumps({'status': result['status'], 'dev_wer': {k: v['gloss_wer'] for k, v in output.items()}, 'frozen_baseline_dev_wer': baseline['gloss_wer']}))


if __name__ == '__main__':
    main()
