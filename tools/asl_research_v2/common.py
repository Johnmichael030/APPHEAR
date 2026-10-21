"""Read-only access to prior experiments; new outputs confined to the new run."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
SOURCE=Path('D:/classmate_e1_matched_rtx3050_package/classmate_e1_matched_rtx3050_package')
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def run_root(path: Path) -> Path:
    run=path.resolve();run.relative_to(ROOT/'experiments/continuous_asl')
    if run==PREVIOUS or not (run/'preservation_baseline.json').is_file():raise ValueError('NEW_RUN_REQUIRED')
    return run
def write(path: Path,value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
