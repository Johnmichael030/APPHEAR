"""Supplementary actual192 feature-mask checks; no model import/inference or TEST access."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_framing',RUN/'framing_control.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg=m.read(RUN/'config.json');rows=m.read(RUN/'subset_manifest.json')
position=np.r_[np.arange(84),[176,177,179,180],np.arange(182,198)]
other=np.setdiff1d(np.arange(378),position)
frames=0
for r in rows:
 p=Path(r['feature_path']);assert m.sha(p)==r['feature_sha256']
 with np.load(p,allow_pickle=False) as z:
  assert str(z['schema_version'].item())==m.SCHEMA;x=z['X'].copy()
 before=x.copy();frames+=len(x)
 for dx,dy in cfg['offsets']:
  y=m.translate(x,dx,dy)
  assert np.array_equal(y[:,other],x[:,other])
  assert np.all(y[:,:84].reshape(-1,42,2)[x[:,84:126]==0]==0)
  assert np.all(y[:,182:198].reshape(-1,8,2)[x[:,206:214]==0]==0)
  for side in range(2):assert np.all(y[x[:,174+side]==0,176+side*3:179+side*3]==0)
 assert np.array_equal(x,before)
 assert m.sha(p)==r['feature_sha256']
result=dict(status='PASS',TRAIN_utterances=len(rows),frames=frames,offsets=4,invalid_geometry_zeros_preserved=True,nonposition_channels_exact=True,original_arrays_and_files_unchanged=True,model_inference=False,TEST_used=False)
with (RUN/'actual_feature_mask_checks.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
