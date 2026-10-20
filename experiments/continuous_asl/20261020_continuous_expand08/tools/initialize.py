"""Freeze largest internally consistent verified TRAIN label pool and fixed9DEV."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import numpy as np
from core import RUN,ROOT,SOURCE,FIVE,SIX,SEVEN,SCHEMA,read,write,sha

def main():
    if (RUN/'dataset_plan.json').exists():
        p=read(RUN/'dataset_plan.json')
        if sha(RUN/'data/processed/train_dev/manifest.json')!=p['manifest_sha256']:raise ValueError('FROZEN_MANIFEST_CHANGED')
        print('INITIALIZED_FROZEN_SNAPSHOT_NO_REWRITE');return
    gate=read(SEVEN/'overfit_diagnostic_v2/result.json')
    if gate['status']!='PASS_LEARNABILITY_GATE' or not gate['checkpoint_reload_predictions_match']:raise ValueError('OVERFIT_LEARNABILITY_GATE')
    if gate['best_train']['gloss_wer']>0.05 or gate['loss_ratio']>0.25:raise ValueError('OVERFIT_NUMERICAL_ACCEPTANCE')
    if sha(gate['checkpoint_path'])!=gate['checkpoint_sha256']:raise ValueError('OVERFIT_CHECKPOINT_HASH')
    pilot_path=ROOT/'experiments/continuous_asl/20261020_continuous_expand09/pilot_outcome.json'
    if not pilot_path.exists() or read(pilot_path)['status']!='PASS_LEARNING_SIGNAL':raise ValueError('BOUNDED_PILOT_LEARNING_GATE: inspect expand09/pilot_outcome.json before main fit')
    overlap=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
    if overlap&{r['sample_id'] for r in read(SEVEN/'overfit_diagnostic_v2/manifest.json')}:raise ValueError('OVERFIT_GLOBAL_ELIGIBILITY')
    quality=read(SIX/'canonical/quality.json');canonical=read(SIX/'canonical/manifest.json')
    if quality['manifest_sha256']!=sha(SIX/'canonical/manifest.json'):raise ValueError('CANONICAL_FROZEN_HASH')
    pools=defaultdict(list)
    for r in canonical:
        if r['canonical_role']=='eligible_train':pools[r['training_pool']].append(r)
    pool=max(pools,key=lambda k:(len(pools[k]),k.startswith('ASLLRP3')))
    train=[{**r,'split':'train'} for r in pools[pool]]
    dev=[r for r in read(FIVE/'data/processed/train_dev/manifest.json') if r['split']=='dev']
    olddev=[r for r in read(SOURCE/'data/processed/train_dev/manifest.json') if r['split']=='dev']
    if [(r['sample_id'],r['glosses'],r['feature_sha256']) for r in dev]!=[(r['sample_id'],r['glosses'],r['feature_sha256']) for r in olddev] or len(dev)!=9:raise ValueError('FIXED9DEV_CHANGED')
    held=[r for r in canonical if r['canonical_role'] in ('fixed_dev','reserved_test','quarantine')]
    for key in ('sample_id','raw_sha256'):
        if {r[key] for r in train}&{r[key] for r in held}:raise ValueError('PROTECTED_OVERLAP:'+key)
    if {r['collection'] for r in train}&{r['collection'] for r in dev}:raise ValueError('COLLECTION_OVERLAP')
    if any(r['sample_id'] in overlap for r in train):raise ValueError('PUBLISHER_BOUNDARY_OVERLAP')
    if len({r['sample_id'] for r in train+dev})!=len(train+dev):raise ValueError('IDENTITY_DUPLICATES')
    # Exact raw/video duplicates are invalid for whole utterance clips. Genuine
    # timed NCSLGR segments have already passed the canonical nonoverlap audit.
    for r in train+dev:
        if sha(r['feature_path'])!=r['feature_sha256']:raise ValueError('FEATURE_HASH')
        with np.load(r['feature_path'],allow_pickle=False) as f:
            x=f['X'];ts=f['timestamp_ms']
            if str(f['schema_version'].item())!=SCHEMA or x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
        if len(x)<len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:])):raise ValueError('CTC_FEASIBILITY')
    rows=train+dev;write(RUN/'data/processed/train_dev/manifest.json',rows)
    write(RUN/'data/processed/train_dev/quality.json',{'status':'PASS','source_media_hash_duplicates':0,'exact_decoded_duplicates':[],
          'source_canonical_manifest_sha256':quality['manifest_sha256'],'single_label_pool':pool,'signer_independent':False})
    plan=read(FIVE/'training_plan.v2.json')
    plan.update(models={'smoke':{'channels':48,'max_epochs':1,'wall_seconds':60},'reference48':{'channels':48,'max_epochs':60,'wall_seconds':600}},
                patience=15,task='Literal authentic occurrence-gloss CTC in the largest verified single publisher label pool',
                label_pool=pool,selection='Exact original9DEV WER then loss; max60epochs/600seconds,patience15; no TEST reads',
                baseline='Frozen expand03 reference48, same9DEV labels/features/decoder. Different training corpus/vocabulary/normalization; diagnostic cross-corpus contrast, not architecture-only.',
                overfit_gate_sha256=sha(SEVEN/'overfit_diagnostic_v2/result.json'),
                protocol_sha256=sha(RUN/'research_protocol.md'),normalization='This frozen TRAIN pool only; no DEV/test fitting')
    write(RUN/'training_plan.v2.json',plan)
    write(RUN/'dataset_plan.json',{'fit_counts':{'train':len(train),'dev':len(dev)},'training_pool':pool,
          'manifest_sha256':sha(RUN/'data/processed/train_dev/manifest.json'),'canonical_manifest_sha256':quality['manifest_sha256'],
          'fixed_dev_ids':[r['sample_id'] for r in dev],'train_ids':[r['sample_id'] for r in train],
          'reserved_test_count':17,'reserved_test_manifest_sha256':sha(FIVE/'data/reserved_test/manifest.json'),
          'signer_independent':False,'target2000_reached_in_this_fit_pool':len(train)>=2000,'minimum_shortfall':max(0,2000-len(train)),
          'excluded_other_label_pool_counts':{k:len(v) for k,v in pools.items() if k!=pool},'test_features_read':False})
    source_path=FIVE/'tools/train_expanded.py';source=source_path.read_text(encoding='utf-8')
    changes=[('SOURCE, ACQUIRED, OLD, SCHEMA','SOURCE, ACQUIRED, FIVE, SIX, OLD, SCHEMA'),
      ('or ACQUIRED in path.parents or OLD in path.parents','or ACQUIRED in path.parents or FIVE in path.parents or SIX in path.parents or OLD in path.parents'),
      ("if name=='compact24' and not (RUN/'models/reference48/training_status.json').is_file(): raise ValueError('REFERENCE_MUST_BE_FROZEN_FIRST')",''),
      ("choices=('smoke','reference48','compact24')","choices=('smoke','reference48')"),
      ("'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__))","'unique_train_utterances_consumed':len(fit),'train_utterance_presentations':len(fit)*completed,'train_observations_per_complete_epoch':sum(len(data[r['sample_id']]) for r in fit),'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__))")]
    for before,after in changes:
        if before not in source:raise ValueError('TRAINER_PORT_ANCHOR:'+before)
        source=source.replace(before,after)
    with (RUN/'tools/train_main.py').open('x',encoding='utf-8') as f:f.write(source)
    write(RUN/'code_provenance.json',{'source':str(source_path),'source_sha256':sha(source_path),'new_trainer_sha256':sha(RUN/'tools/train_main.py'),
          'changes':'New verified cache paths, one48-channel model plus smoke, actual consumed counts; unchanged fitting/model/CTC/greedy decoder. New bounded protocol and manifest.',
          'model_decoder_sha256':sha(ROOT/'tools/continuous_asl/ctc_experiment.py')})
    print(json.dumps({'status':'PASS_FROZEN_MAIN_TRAINING_SNAPSHOT','train':len(train),'dev':9,'pool':pool,'target_minimum_shortfall':max(0,2000-len(train))}),flush=True)

if __name__=='__main__':main()
