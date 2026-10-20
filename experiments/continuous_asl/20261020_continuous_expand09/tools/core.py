"""Immediate bounded pilot on the complete verified906 NCSLGR pool."""
from pathlib import Path
import hashlib,json,sys
RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[2]
SOURCE=ROOT/'experiments/continuous_asl/20261020_continuous_expand03'
ACQUIRED=ROOT/'experiments/continuous_asl/20261020_continuous_expand04'
FIVE=ROOT/'experiments/continuous_asl/20261020_continuous_expand05'
SIX=ROOT/'experiments/continuous_asl/20261020_continuous_expand06'
SEVEN=ROOT/'experiments/continuous_asl/20261020_continuous_expand07'
OLD=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01'
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1';SEED=20261020
sys.path.insert(0,str(ROOT/'tools/continuous_asl'));sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p).resolve()
 if RUN not in p.parents:raise ValueError('OUTPUT_OUTSIDE_NEW_EXPERIMENT')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
