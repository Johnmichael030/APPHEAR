"""New development split only; never edits original manifests or uses old weights."""
from collections import Counter
import csv,json,time
from core import RUN,OLD,ROOT,TREE,read,write,sha,SCHEMA

def main(watch=False):
 path=RUN/'data/processed/train_dev/manifest.json'
 if path.exists():
  if sha(path)!=read(RUN/'dataset_plan.json')['manifest_sha256']:raise ValueError('FROZEN_NEW_MANIFEST_CHANGED')
  print('NEW_SPLIT_COMPLETE_NO_REPEAT');return
 deadline=time.monotonic()+900
 while not (RUN/'signer_metadata.json').exists():
  if not watch:raise ValueError('PUBLISHER_SIGNER_METADATA_PENDING')
  if time.monotonic()>deadline:raise ValueError('METADATA_WAIT_BUDGET')
  time.sleep(2)
 metadata=read(RUN/'signer_metadata.json')
 if metadata['status']!='PASS_PUBLISHER_SIGNER_JOINS' or metadata['missing']:raise ValueError('UNVERIFIED_SIGNER_MAP')
 canonical=read(OLD/'canonical/manifest.json')
 source=[r for r in canonical if r['canonical_role']=='eligible_train' and r['training_pool']=='ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE']
 heldout='asllrp_dai_participant_4';rows=[]
 for original in source:
  evidence=metadata['joined_eligible_records'][original['sample_id']]
  row={**original,'signer_id':evidence['signer_id'],'signer_provenance':evidence,'split':'dev' if evidence['signer_id']==heldout else 'train'}
  rows.append(row)
 fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
 if len(rows)!=2119 or len(dev)<100 or len(fit)<1000:raise ValueError('NEW_SPLIT_SIZE_GATE')
 overlap={}
 for key in ('sample_id','signer_id','collection','source_group','raw_sha256','decoded_pixel_sha256'):
  overlap[key]=sorted({r[key] for r in fit}&{r[key] for r in dev})
 if any(overlap.values()):raise ValueError('TRAIN_DEV_GROUP_OR_CONTENT_OVERLAP')
 forbidden=set(read(OLD/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
 if forbidden&{r['sample_id'] for r in rows} or 'asllrp3_8d6cbe4ca1fc7986' in {r['sample_id'] for r in rows}:raise ValueError('PRIOR_EXCLUSIONS_CHANGED')
 labels=Counter(t for r in fit for t in r['glosses']);oov=Counter(t for r in dev for t in r['glosses'] if t not in labels)
 write(path,rows)
 write(RUN/'data/processed/train_dev/quality.json',{'status':'PASS_PUBLISHER_SIGNER_SOURCE_DISJOINT','overlap':overlap,'same_label_pool':True,'feature_revalidation':'Previously passed feature/schema/hash/CTC evidence reused; membership/partition changes only. Trainer verifies selected frozen feature hashes before fitting.','excluded_ids':sorted(forbidden|{'asllrp3_8d6cbe4ca1fc7986'})})
 plan={'status':'PASS_RESEARCH_DEV_PROTOCOL','manifest_sha256':sha(path),'old_main_manifest_path':str(TREE/'20261020_continuous_expand08/data/processed/train_dev/manifest.json'),
  'old_main_manifest_sha256':'adebeb9e2f20179eb59b7f79805ea9226ad6c9c0594e6cba193c771947cdaf40',
  'fit_counts':dict(Counter(r['split'] for r in rows)),'heldout_publisher_signer':heldout,'signers_by_split':{s:sorted({r['signer_id'] for r in rows if r['split']==s}) for s in ('train','dev')},
  'source_recordings_by_split':{s:len({r['source_group'] for r in rows if r['split']==s}) for s in ('train','dev')},'overlap':overlap,
  'target_pool':'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE','vocabulary_fit':'New TRAIN only, blank0/UNK1; literal publisher occurrence labels unchanged',
  'train_labels':len(labels),'dev_reference_tokens':sum(len(r['glosses']) for r in dev),'dev_oov_tokens':sum(oov.values()),'dev_oov_labels':dict(oov),
  'selection_policy':'Publisher participant4 fixed before fitting/new DEV predictions; use every eligible utterance for that signer, no score-based record selection.',
  'prior_exposure':'All records previously TRAIN in old runs. This new model MUST start from random weights and a new optimizer. DEV is development-exposed, not a pristine TEST. No prior ASL recognition checkpoint may be warm-started.',
  'signer_independent_scope':'One publisher held-out signer vs three publisher TRAIN signers, disjoint source recordings within this corpus. Not a multiple-signer statistical accuracy claim; no undocumented cross-corpus identity inference.',
  'test_accessed':False,'new_weight_initialization':'RANDOM_ONLY','production_enabled':False}
 if sha(plan['old_main_manifest_path'])!=plan['old_main_manifest_sha256']:raise ValueError('ORIGINAL_MAIN_MANIFEST_CHANGED')
 write(RUN/'dataset_plan.json',plan)
 with (RUN/'EVALUATION_SPLIT_AUDIT.csv').open('x',encoding='utf-8-sig',newline='') as f:
  names=['status','split','sample_id','publisher_signer_id','collection','source_recording','feature_sha256','label_pool','prior_exposure','signer_source_overlap']
  w=csv.DictWriter(f,fieldnames=names);w.writeheader()
  for row in rows:w.writerow(dict(zip(names,['PASS',row['split'],row['sample_id'],row['signer_id'],row['collection'],row['source_group'],row['feature_sha256'],row['training_pool'],'PreviouslyTRAIN; new model random initialization required',False])))
 print(json.dumps({k:plan[k] for k in ('status','fit_counts','train_labels','dev_reference_tokens','dev_oov_tokens','manifest_sha256')}),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');main(p.parse_args().watch)
