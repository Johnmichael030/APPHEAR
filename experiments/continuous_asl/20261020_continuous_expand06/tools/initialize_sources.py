"""Capture preservation evidence before any new source acquisition or regression work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
THREE = ROOT / 'experiments/continuous_asl/20261020_continuous_expand03'
FOUR = ROOT / 'experiments/continuous_asl/20261020_continuous_expand04'
FIVE = ROOT / 'experiments/continuous_asl/20261020_continuous_expand05'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = RUN / 'preservation_baseline.json'
    if output.exists():
        print('SOURCE_PHASE_BASELINE_EXISTS_NO_REWRITE')
        return
    frozen = []
    for base in (THREE, FOUR, FIVE):
        for relative in ('model_selection_freeze.json', 'data/processed/train_dev/manifest.json', 'training_plan.v2.json'):
            p = base / relative
            frozen.append({'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha(p)})
        for name in ('reference48', 'compact24'):
            folder = base / 'models' / name
            pointer = json.loads((folder / 'checkpoints/best.json').read_text())
            for p in (folder / 'config.json', folder / 'training_status.json', folder / pointer['path']):
                frozen.append({'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha(p)})
    old = json.loads((FOUR / 'preservation_baseline.json').read_text())
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'metadata_snapshot': old['metadata_snapshot'],
              'fresh_frozen_artifact_hashes': frozen, 'source_inventory_guard': old['source_artifact_manifest_sha256'],
              'minimum_utterance_target': 2000, 'stretch_target': 5000, 'verified_count_at_start': 552,
              'acquisition_cap_bytes_unchanged': 2147483648, 'production_enabled': False}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'PASS_SOURCE_PHASE_INITIALIZED', 'frozen_hashes': len(frozen), 'minimum_shortfall': 1448}))


if __name__ == '__main__':
    main()
