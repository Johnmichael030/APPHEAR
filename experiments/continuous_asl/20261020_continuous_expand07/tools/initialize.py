"""Freeze the next verified timed NCSLGR batch; never fit during initialization."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from core import RUN, ROOT, SOURCE, FIVE, SIX, SCHEMA, read, write, sha

def main():
    if (RUN/'dataset_plan.json').exists():
        plan=read(RUN/'dataset_plan.json')
        if sha(RUN/'data/processed/train_dev/manifest.json')!=plan['manifest_sha256']:raise ValueError('FROZEN_MANIFEST_CHANGED')
        print('ALREADY_INITIALIZED_VERIFIED');return
    diagnosis=read(SIX/'wer_diagnosis.json')
    if diagnosis['test_accessed']:raise ValueError('DIAGNOSIS_TEST_CONTAMINATION')
    source_rows=read(SIX/'timed_preparation/manifest.json')
    quality=read(SIX/'timed_preparation/quality.json')
    if quality['status']!='PASS_VALID_SUBSET' or not source_rows:raise ValueError('NEW_TIMED_DATA_GATE')
    rows=read(FIVE/'data/processed/train_dev/manifest.json')+source_rows
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('DUPLICATE_IDENTITY')
    dev=[r for r in rows if r['split']=='dev'];fit=[r for r in rows if r['split']=='train']
    olddev=[r for r in read(SOURCE/'data/processed/train_dev/manifest.json') if r['split']=='dev']
    if [(r['sample_id'],r['glosses'],r['feature_sha256']) for r in dev]!=[(r['sample_id'],r['glosses'],r['feature_sha256']) for r in olddev]:raise ValueError('DEV_CHANGED')
    protected=read(FIVE/'data/reserved_test/manifest.json')
    for key in ('sample_id','raw_sha256','source_group'):
        if {r[key] for r in fit}&{r[key] for r in protected+dev}:raise ValueError('PROTECTED_OVERLAP:'+key)
    if {r['collection'] for r in fit}&{r['collection'] for r in dev}:raise ValueError('COLLECTION_OVERLAP')
    sources=defaultdict(list);pixels=set()
    for row in rows:
        sources[row['raw_sha256']].append(row)
        pixel=row.get('decoded_segment_observation_sha256',row.get('decoded_pixel_sha256'))
        if not pixel or pixel in pixels:raise ValueError('EXACT_DECODED_CONTENT_DUPLICATE')
        pixels.add(pixel)
        if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
        with np.load(row['feature_path'],allow_pickle=False) as f:
            x=f['X'];ts=f['timestamp_ms']
            if str(f['schema_version'].item())!=SCHEMA or x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
        if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_LENGTH')
    shared=0
    for checksum,group in sources.items():
        if len(group)<2:continue
        shared+=len(group)-1
        if any('start_ms' not in r or 'end_ms' not in r or r['split']!='train' for r in group):raise ValueError('UNVERIFIED_SHARED_MEDIA')
        ordered=sorted(group,key=lambda r:r['start_ms'])
        if any(a['end_ms']>b['start_ms'] for a,b in zip(ordered,ordered[1:])):raise ValueError('OVERLAPPING_SEGMENTS')
    write(RUN/'data/processed/train_dev/manifest.json',rows)
    write(RUN/'data/processed/train_dev/quality.json',{'status':'PASS','invalid_source_overlap':0,'exact_decoded_duplicates':[],
          'intentional_shared_media_utterances_beyond_first':shared,'shared_media_rule':'Only verified nonoverlapping publisher spans entirely inside TRAIN',
          'source_quality':str(SIX/'timed_preparation/quality.json'),'signer_independent':False})
    plan=read(FIVE/'training_plan.v2.json')
    plan.update(models={'smoke':{'channels':48,'max_epochs':1,'wall_seconds':60},'warm48':{'channels':48,'max_epochs':20,'wall_seconds':180}},
                lr=0.0001,patience=5,normalization='Frozen originalTRAIN100 mean/std in expand03 reference checkpoint',
                baseline='Frozen expand03 reference48; exact mapped-logit/greedy parity required before fitting',
                selection='Fixed9DEV WER then loss, including initialized epoch0; max20epochs/180seconds/patience5.',
                task='Literal authentic main-gloss-event CTC on prior455 plus verified timed NCSLGR utterances',
                diagnosis_sha256=sha(SIX/'wer_diagnosis.json'),protocol_sha256=sha(RUN/'research_protocol.md'),
                warm_start=True,new_output_bias=-10.)
    write(RUN/'training_plan.v2.json',plan)
    write(RUN/'dataset_plan.json',{'fit_counts':dict(Counter(r['split'] for r in rows)),
          'manifest_sha256':sha(RUN/'data/processed/train_dev/manifest.json'),
          'reused_train455':455,'new_verified_timed_train':len(source_rows),'reserved_test_count':len(protected),
          'reserved_test_manifest_sha256':sha(FIVE/'data/reserved_test/manifest.json'),
          'source_manifest_sha256':sha(SIX/'timed_preparation/manifest.json'),'signer_independent':False,'test_features_read':False})
    source=(FIVE/'tools/train_expanded.py').read_text(encoding='utf-8')
    changes=[
      ('SOURCE, ACQUIRED, OLD, SCHEMA','SOURCE, ACQUIRED, FIVE, SIX, OLD, SCHEMA'),
      ("quality['source_media_hash_duplicates']","quality['invalid_source_overlap']"),
      ('or ACQUIRED in path.parents or OLD in path.parents','or ACQUIRED in path.parents or FIVE in path.parents or SIX in path.parents or OLD in path.parents'),
      ("if name=='compact24' and not (RUN/'models/reference48/training_status.json').is_file(): raise ValueError('REFERENCE_MUST_BE_FROZEN_FIRST')",''),
      ("vocabulary=['<BLANK>','<UNK>']+sorted({s for r in fit for s in r['glosses']})","vocabulary=extended_vocabulary(fit)"),
      ("model=CausalCTC(len(vocabulary),config['channels'],mean,std).to('cuda:0')","model=build(vocabulary,'cuda:0')"),
      ("write(out/'untrained_dev.json',evaluate(model,dev[:2] if name=='smoke' else dev,data,vocabulary,'cuda:0'))", """write(out/'initialization_parity.json',parity(model,dev,data,vocabulary,'cuda:0'))
        initial=evaluate(model,dev[:2] if name=='smoke' else dev,data,vocabulary,'cuda:0')
        write(out/'untrained_dev.json',initial)
        best=(initial['gloss_wer'],initial['loss'])
        pointer=checkpoint(out,0,model,optimizer,config,vocabulary,manifest_hash,best,0)
        atomic_json(out/'checkpoints/best.json',pointer);atomic_json(out/'dev_best.json',initial)"""),
      ("choices=('smoke','reference48','compact24')","choices=('smoke','warm48')")]
    # Remove the now-inapplicable TRAIN-expanded normalization calculation entirely.
    first="    values=np.concatenate([data[r['sample_id']] for r in fit]).astype(np.float64)"
    last="    del values\n"
    start=source.index(first);end=source.index(last,start)+len(last)
    source=source[:start]+source[end:]
    for before,after in changes:
        if before not in source:raise ValueError('TRAINER_PORT_ANCHOR:'+before)
        source=source.replace(before,after)
    source=source.replace('from replay_profile import memory','from replay_profile import memory\nfrom warm_start import extended_vocabulary,build,parity')
    source=source.replace("'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__))", "'unique_train_utterances_consumed':len(fit),'train_utterance_presentations':len(fit)*completed,'train_observations_per_complete_epoch':sum(len(data[r['sample_id']]) for r in fit),'initialized_epoch0_may_win':True,'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__))")
    destination=RUN/'tools/train_warm.py'
    with destination.open('x',encoding='utf-8') as f:f.write(source)
    write(RUN/'code_provenance.json',{'source':str(FIVE/'tools/train_expanded.py'),'source_sha256':sha(FIVE/'tools/train_expanded.py'),
          'new_trainer_sha256':sha(destination),'warm_start_sha256':sha(RUN/'tools/warm_start.py'),
          'changes':'Baseline warm-start, frozen norm, exact-token extended head, initialized epoch0 may win, new timed-source paths and nonoverlap-aware quality gate. Unchanged causal model/CTC/decoder/FP32 loop.',
          'model_decoder_sha256':sha(ROOT/'tools/continuous_asl/ctc_experiment.py')})
    print(json.dumps({'status':'PASS_FROZEN_FOR_BOUNDED_ADAPTATION','train':len(fit),'dev':len(dev),'new_timed':len(source_rows)}),flush=True)

if __name__=='__main__':main()
