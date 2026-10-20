"""Read-only artifact/data/selection verification. Never trains or scores TEST."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from core import RUN,ROOT,SOURCE,FIVE,SIX,SEVEN,SCHEMA,read,sha
from finalize import NEW_ROOTS,preservation

def main():
 indexes={}
 for root in NEW_ROOTS:
  index=read(root/'artifact_manifest.json')
  for entry in index['files']:
   path=(root/entry['path']).resolve()
   if root not in path.parents or not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('ARTIFACT_INTEGRITY:'+str(path))
  indexes[root.name]=len(index['files'])
 summary=read(RUN/'execution_summary.json');plan=read(RUN/'dataset_plan.json')
 canonical=read(SIX/'canonical/manifest.json');quality=read(SIX/'canonical/quality.json')
 if sha(SIX/'canonical/manifest.json')!=quality['manifest_sha256']:raise ValueError('CANONICAL_HASH')
 if sha(RUN/'data/processed/train_dev/manifest.json')!=plan['manifest_sha256']:raise ValueError('FROZEN_FIT_MANIFEST')
 rows=read(RUN/'data/processed/train_dev/manifest.json');fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
 if dict(Counter(r['split'] for r in rows))!=plan['fit_counts']:raise ValueError('SPLIT_COUNTS')
 if len({r['training_pool'] for r in fit})!=1:raise ValueError('PUBLISHER_LABEL_POOLS_MIXED')
 original=[r for r in read(SOURCE/'data/processed/train_dev/manifest.json') if r['split']=='dev']
 if [(r['sample_id'],r['glosses'],r['feature_sha256']) for r in dev]!=[(r['sample_id'],r['glosses'],r['feature_sha256']) for r in original]:raise ValueError('FROZEN9DEV_CHANGED')
 forbidden=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
 if forbidden&{r['sample_id'] for r in fit}:raise ValueError('OVERLAPPING_PUBLISHER_UTTERANCE_ADMITTED')
 protected=[r for r in canonical if r['canonical_role'] in ('fixed_dev','reserved_test','quarantine')]
 for key in ('sample_id','raw_sha256'):
  if {r[key] for r in fit}&{r[key] for r in protected}:raise ValueError('PROTECTED_LEAKAGE:'+key)
 for row in rows:
  if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
  with np.load(row['feature_path'],allow_pickle=False) as f:
   x=f['X'];ts=f['timestamp_ms']
   if str(f['schema_version'].item())!=SCHEMA or x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
  if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_FEASIBILITY')
 freeze=read(RUN/'model_selection_freeze.json')
 if sha(RUN/'training_plan.v2.json')!=freeze['training_plan_sha256'] or sha(RUN/'tools/train_main.py')!=freeze['trainer_sha256']:raise ValueError('MAIN_TRAINING_CONTRACT_CHANGED')
 model=read(RUN/'models/reference48/training_status.json')
 if not model['used_cuda'] or not model['checkpoint_reload_predictions_match'] or model['unique_train_utterances_consumed']!=len(fit):raise ValueError('ACTUAL_TRAINING_EVIDENCE')
 if summary['new_test_evaluated']!=0 or summary['reserved_test_untouched']!=17 or summary['production_promoted']:raise ValueError('EVALUATION_OR_PROMOTION_PROTOCOL')
 if sha(FIVE/'data/reserved_test/manifest.json')!=plan['reserved_test_manifest_sha256']:raise ValueError('RESERVED_TEST_MANIFEST_CHANGED')
 guard=preservation()
 print(json.dumps({'status':'PASS_READ_ONLY_VERIFICATION','training_performed_by_verifier':False,'test_evaluated_by_verifier':False,
       'artifact_hash_counts':indexes,'fit_counts':plan['fit_counts'],'main_manifest_sha256':plan['manifest_sha256'],
       'checkpoint_path':summary['main_comparison']['candidate']['checkpoint_path'],'checkpoint_sha256':summary['main_comparison']['candidate']['checkpoint_sha256'],
       'preservation':guard,'signer_independent':False,'production_promoted':False},indent=2),flush=True)

if __name__=='__main__':main()
