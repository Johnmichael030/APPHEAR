"""Additive official timed-corpus preparation and regression diagnostics."""
from pathlib import Path
import hashlib
import json
import sys

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
THREE = ROOT/'experiments/continuous_asl/20261020_continuous_expand03'
FOUR = ROOT/'experiments/continuous_asl/20261020_continuous_expand04'
FIVE = ROOT/'experiments/continuous_asl/20261020_continuous_expand05'
SCHEMA = 'apphear.continuous_asl.candidate01_causal378.v1'
sys.path.insert(0, str(ROOT/'tools/continuous_asl'))
sys.path.insert(0, str(ROOT))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(path, obj):
    path = Path(path).resolve()
    if RUN not in path.parents:
        raise ValueError('OUTPUT_OUTSIDE_NEW_EXPERIMENT')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')
