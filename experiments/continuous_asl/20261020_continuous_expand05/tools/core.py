"""New fit on broader authentic ASL word-gloss data; frozen sources are read-only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
SOURCE = ROOT / 'experiments/continuous_asl/20261020_continuous_expand03'
ACQUIRED = ROOT / 'experiments/continuous_asl/20261020_continuous_expand04'
OLD = ROOT / 'experiments/continuous_asl/20261019T155428Z_cslr01'
SCHEMA = 'apphear.continuous_asl.candidate01_causal378.v1'
SEED = 20261020
sys.path.insert(0, str(ROOT / 'tools/continuous_asl'))
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path).resolve()
    if RUN not in path.parents:
        raise ValueError('OUTPUT_OUTSIDE_NEW_EXPERIMENT')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
