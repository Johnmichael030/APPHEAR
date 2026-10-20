"""Frozen TRAIN-only compact CTC control; prepare first, then execute one bounded fit."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,math,os,random,shutil,sys,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RUN=Path(__file__).resolve().parent;BASE=RUN.parent
PREV=BASE/'20261020T143609Z_learning_diagnostic01';AUDIT=BASE/'20261020T135119Z_read_only_failure_audit01'
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
def snap(paths):
 ans={}
 for p in sorted(set(map(Path,paths))):
  s=p.stat();ans[str(p)]=dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 return ans
def family(r):
 import re
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
 return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def csvwrite(p,rows):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
 assert not (RUN/'control_config.json').exists(),'PREPARATION_ALREADY_FROZEN'
 # Check prior artifact index as explicit non-TEST scope; no broad discovery.
 priorpaths=[PREV/'artifact_hashes.sha256']
 for line in (PREV/'artifact_hashes.sha256').read_text().splitlines():
  digest,rel=line.split('  ',1);p=(PREV/rel).resolve();assert PREV in p.parents and sha(p)==digest,rel;priorpaths.append(p)
 rows=read(PREV/'resolved_manifest.json');train=[r for r in rows if r['internal_role']=='TRAIN']
 assert len(train)==1500 and len({r['sample_id'] for r in train})==1500
 fold=read(PREV/'fold_manifest.json');assert {r['sample_id'] for r in train}==set(fold['train_sample_ids'])
 assert all(r['split']=='train' for r in train)
 meta={r['sample_id']:r for r in read(AUDIT/'feature_and_target_checks.json') if r['role']=='TRAIN'}
 assert set(meta)=={r['sample_id'] for r in train}
 support=Counter(t for r in train for t in set(r['glosses']));occ=Counter(t for r in train for t in r['glosses'])
 tq=np.quantile([meta[r['sample_id']]['observations'] for r in train],[.25,.5,.75]);uq=np.quantile([len(r['glosses']) for r in train],[.25,.5,.75])
 def fields(r):
  m=meta[r['sample_id']]
  return dict(family=family(r),signer=r['signer_id'],master=r['source_group'],input_quartile=int(np.searchsorted(tq,m['observations'],side='left')),target_quartile=int(np.searchsorted(uq,len(r['glosses']),side='left')),rare_bearing=any(support[t]<5 for t in r['glosses']),common_bearing=any(support[t]>=20 for t in r['glosses']),observation_category='NO_FRAME' if m['missing_event_observation_count'] else 'NO_HAND' if m['sampled_events_no_hand_count'] else 'OTHER',missing_frame_events=m['missing_event_observation_count'],no_hand_events=m['sampled_events_no_hand_count'])
 desc={r['sample_id']:fields(r) for r in train};families=sorted({family(r) for r in train});assert len(families)==6
 selected=[];used=set();strata=Counter();globalcounts=Counter();masters=Counter()
 # Fixed equal16-per-family quota; diversity, not ease or prior predictions.
 for roundnum in range(16):
  for fam in families:
   choices=[r for r in train if family(r)==fam and r['sample_id'] not in used]
   def key(r):
    d=desc[r['sample_id']];sk=(fam,d['input_quartile'],d['target_quartile'],d['observation_category'],d['rare_bearing'])
    balance=sum(globalcounts[(k,d[k])] for k in ['input_quartile','target_quartile','observation_category','rare_bearing','common_bearing'])
    return strata[sk],masters[d['master']],balance,hashlib.sha256(('compact96-seed20261020|'+r['sample_id']).encode()).hexdigest()
   r=min(choices,key=key);d=desc[r['sample_id']]
   selected.append(r);used.add(r['sample_id']);strata[(fam,d['input_quartile'],d['target_quartile'],d['observation_category'],d['rare_bearing'])]+=1;masters[d['master']]+=1
   for k in ['input_quartile','target_quartile','observation_category','rare_bearing','common_bearing']:globalcounts[(k,d[k])]+=1
 assert len(selected)==96 and len({r['signer_id'] for r in selected})==3
 vocab=read(PREV/'vocabulary.json')['tokens'];assert vocab==['<BLANK>','<UNK>']+sorted(support)
 paths=priorpaths+[AUDIT/'feature_and_target_checks.json',Path(__file__),ROOT/'tools/continuous_asl/ctc_experiment.py',ROOT/'tools/continuous_asl/prepare_features.py']
 cfgprev=read(PREV/'config.json');assert sha(ROOT/'tools/continuous_asl/ctc_experiment.py')==cfgprev['source_model_decoder_sha256']
 for r in selected:
  p=Path(r['feature_path']);assert sha(p)==r['feature_sha256'];paths.append(p)
  with np.load(p,allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA
   x=z['X'];ts=z['timestamp_ms'];m=json.loads(str(z['metadata_json'].item()))
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
   assert m['sample_id']==r['sample_id'] and m['glosses']==r['glosses']
   assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
 selectedsupport=Counter(t for r in selected for t in set(r['glosses']))
 coverage=dict(utterances=96,families=dict(Counter(family(r) for r in selected)),signers=dict(Counter(r['signer_id'] for r in selected)),masters=len(masters),unique_labels=len(selectedsupport),token_occurrences=sum(len(r['glosses']) for r in selected),full_train_unique_labels=len(support),full_train_token_occurrences=sum(occ.values()),selected_labels_by_fullTRAIN_support={name:sum(fn(support[t]) for t in selectedsupport) for name,fn in [('1',lambda n:n==1),('2_4',lambda n:2<=n<=4),('5_9',lambda n:5<=n<=9),('10_19',lambda n:10<=n<=19),('20plus',lambda n:n>=20)]},utterances_no_retained_frame_events=sum(desc[r['sample_id']]['missing_frame_events']>0 for r in selected),utterances_sampled_no_hand_events=sum(desc[r['sample_id']]['no_hand_events']>0 for r in selected),input_quartile_cuts=tq.tolist(),target_quartile_cuts=uq.tolist(),input_quartiles=dict(Counter(desc[r['sample_id']]['input_quartile'] for r in selected)),target_quartiles=dict(Counter(desc[r['sample_id']]['target_quartile'] for r in selected)),input_length_range=[min(meta[r['sample_id']]['observations'] for r in selected),max(meta[r['sample_id']]['observations'] for r in selected)],target_length_range=[min(len(r['glosses']) for r in selected),max(len(r['glosses']) for r in selected)],limitation='Equal family quotas deliberately overrepresent smaller families; coverage-balanced diagnostic, not a population-random sample or unseen-signer evaluation.')
 cfg=dict(schema=SCHEMA,seed=20261020,selection_seed='compact96-seed20261020',selection='16 per TRAIN family; greedy unused (family,input quartile,target quartile,observation category,rare-bearing) cells, unused masters, underrepresented marginal categories, stable SHA; no predictions/heldout used',channels=48,classes=len(vocab),batch_size=4,accumulation=2,accumulation_tail='actual microbatch group count',lr=.001,weight_decay=.0001,gradient_clip=5.,precision='FP32',max_epochs=120,max_optimizer_steps=1440,wall_stop_request_seconds=90,evaluation_every_epochs=5,pass_WER=.30,pass_loss_fraction_of_initial=.50,minimum_absolute_WER_drop=.20,max_family_WER=.50,max_rare_bearing_WER=.50,max_long_input_WER=.50,success='All global+family+rare-bearing+upper-input-quartile criteria must pass; reference/hypothesis literal tokens unchanged',collapse_stop='After epoch30: two evaluations with blank>=.995 or top-token fraction>=.98/diversity<=3 AND loss reduction<30%; otherwise stop on numerical/resource failure or declared budget',vocabulary_sha256=sha(PREV/'vocabulary.json'),normalization_sha256=sha(PREV/'normalization.npz'),frozen_fold_sha256=sha(PREV/'fold_manifest.json'),model_decoder_sha256=sha(ROOT/'tools/continuous_asl/ctc_experiment.py'),control_script_sha256=sha(__file__),TRAIN_only=True,DEV_used_for_selection_or_training=False,TEST_used=False,promotion=False)
 save(RUN/'subset_manifest.json',[dict(**r,compact_selection=desc[r['sample_id']]) for r in selected])
 save(RUN/'subset_coverage.json',coverage);save(RUN/'control_config.json',cfg)
 for name in ['vocabulary.json','normalization.npz','fold_manifest.json']:shutil.copyfile(PREV/name,RUN/name)
 save(RUN/'control_input_hashes_before.json',snap(paths))
 save(RUN/'preflight.json',dict(status='PASS',prior_artifact_hashes_verified=len(priorpaths)-1,subset_features_verified=96,vocabulary_size=len(vocab),numerical_integrity=True,fold_membership=True))
 print(json.dumps(dict(status='PREPARED',coverage=coverage,predeclared_bounds=dict(epochs=120,steps=1440,seconds=90)),indent=2),flush=True)
def fit():
 import torch
 sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
 from ctc_experiment import CausalCTC,batch,decode,edit_counts,seeds
 assert not (RUN/'execution_started.json').exists(),'ONE_COMPACT_FIT_ONLY'
 cfg=read(RUN/'control_config.json');assert sha(__file__)==cfg['control_script_sha256']
 inputs=read(RUN/'control_input_hashes_before.json')
 for p,e in inputs.items():assert sha(p)==e['sha256'],p
 assert (RUN/'PHASE1_FINDINGS.md').exists(),'WAIT_FOR_PHASE1_COMPLETION'
 diag=read(RUN/'diagnosis_summary.json');assert diag.get('pipeline_valid_for_compact_control',False),'PHASE1_GATE_BLOCKED'
 save(RUN/'execution_started.json',dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),config_sha256=sha(RUN/'control_config.json')))
 rows=read(RUN/'subset_manifest.json');vocab=read(RUN/'vocabulary.json')['tokens'];mapping={t:i for i,t in enumerate(vocab)}
 assert len(rows)==96 and all(all(mapping[t]>=2 for t in r['glosses']) for r in rows)
 data={}
 for r in rows:
  with np.load(r['feature_path'],allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA;data[r['sample_id']]=z['X'].copy()
 with np.load(RUN/'normalization.npz',allow_pickle=False) as z:mean=z['mean'].copy();std=z['std'].copy()
 assert torch.cuda.is_available(),'CUDA_UNAVAILABLE'
 seeds(cfg['seed']);model=CausalCTC(len(vocab),48,mean,std).to('cuda:0');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 save(RUN/'environment.json',dict(GPU=torch.cuda.get_device_name(0),CUDA=torch.version.cuda,torch=torch.__version__,VRAM_total=torch.cuda.get_device_properties(0).total_memory,VRAM_free=torch.cuda.mem_get_info()[0],CPU_threads=os.cpu_count()))
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter();deadline=started+cfg['wall_stop_request_seconds'];steps=0;current_epoch=0;last_eval_epoch=0;history=[];reason='MAX_EPOCHS';collapsed=0;passed=False
 def aggregate(pred):
  c=Counter();counts=Counter()
  for p in pred:
   c.update({k:p[k] for k in ['substitutions','deletions','insertions','reference_tokens']});counts.update(p['hypothesis'])
  return dict(WER=sum(c[k] for k in ['substitutions','deletions','insertions'])/c['reference_tokens'],CTC_loss=float(np.mean([p['CTC_loss'] for p in pred])),**dict(c),blank_fraction=sum(p['blank_frames'] for p in pred)/sum(p['output_frames'] for p in pred),token_diversity=len(counts),top_token_fraction=counts.most_common(1)[0][1]/sum(counts.values()) if counts else 0.,prediction_reference_length_ratio=sum(len(p['hypothesis']) for p in pred)/c['reference_tokens'],empty_outputs=sum(not p['hypothesis'] for p in pred),predictions=pred)
 def evaluate():
  model.eval();pp=[]
  with torch.inference_mode():
   for r in rows:
    x,y,l,tl=batch([r],data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);assert torch.isfinite(loss),'NONFINITE_EVAL'
    h=decode(z[0],vocab);ids=z[0].argmax(-1)
    pp.append(dict(sample_id=r['sample_id'],reference=r['glosses'],hypothesis=h,**edit_counts(r['glosses'],h),CTC_loss=float(loss),blank_frames=int((ids==0).sum()),output_frames=len(ids),family=family(r),rare_bearing=r['compact_selection']['rare_bearing'],input_quartile=r['compact_selection']['input_quartile']))
  a=aggregate(pp);a['family_WER']={f:aggregate([p for p in pp if p['family']==f])['WER'] for f in sorted({p['family'] for p in pp})}
  a['rare_bearing_WER']=aggregate([p for p in pp if p['rare_bearing']])['WER']
  a['upper_input_quartile_WER']=aggregate([p for p in pp if p['input_quartile']==3])['WER']
  return a
 def checkpoint(name,epoch,extra=None):
  p=RUN/'checkpoints'/name;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
  state=dict(schema=SCHEMA,epoch=epoch,optimizer_steps=steps,model=model.state_dict(),optimizer=opt.state_dict(),vocabulary=vocab,config=cfg,config_sha256=sha(RUN/'control_config.json'),subset_sha256=sha(RUN/'subset_manifest.json'),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),elapsed_seconds=time.perf_counter()-started)
  if extra:state.update(extra)
  torch.save(state,p);return dict(path=str(p.relative_to(RUN)),sha256=sha(p),epoch=epoch,optimizer_steps=steps)
 def metric(epoch,score,norm,fitsec):
  row=dict(epoch=epoch,optimizer_steps=steps,elapsed_seconds=time.perf_counter()-started,CTC_loss=score['CTC_loss'],WER=score['WER'],substitutions=score['substitutions'],deletions=score['deletions'],insertions=score['insertions'],reference_tokens=score['reference_tokens'],blank_fraction=score['blank_fraction'],prediction_reference_length_ratio=score['prediction_reference_length_ratio'],token_diversity=score['token_diversity'],empty_outputs=score['empty_outputs'],top_token_fraction=score['top_token_fraction'],worst_family_WER=max(score['family_WER'].values()),rare_bearing_WER=score['rare_bearing_WER'],upper_input_quartile_WER=score['upper_input_quartile_WER'],gradient_norm_mean=norm,fit_seconds=fitsec,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  p=RUN/'diagnostic_metrics.csv'
  with p.open('a',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(row))
   if f.tell()==0:w.writeheader()
   w.writerow(row)
  print(json.dumps(row),flush=True)
 try:
  initial=evaluate();save(RUN/'initial_predictions.json',initial);metric(0,initial,None,0.);checkpoint('initialization.pt',0);last=initial;lastcp=None
  for epoch in range(1,cfg['max_epochs']+1):
   model.train();order=list(rows);random.shuffle(order);micro=[order[i:i+4] for i in range(0,len(order),4)];norms=[];tick=time.perf_counter();interrupted=False
   for gs in range(0,len(micro),2):
    if time.perf_counter()>=deadline or steps>=cfg['max_optimizer_steps']:
     reason='WALL_OR_STEP_BUDGET';interrupted=True
     lastcp=checkpoint(f'partial_epoch_{epoch:03d}.pt',current_epoch,dict(partial_epoch=epoch,next_group_microbatch_index=gs,shuffle_order_ids=[r['sample_id'] for r in order],pending_gradients=False));break
    free,_=torch.cuda.mem_get_info();assert free>=256*2**20,'FREE_VRAM_STOP';assert shutil.disk_usage(RUN).free>=10*2**30,'DISK_STOP'
    assert torch.cuda.max_memory_allocated()<=int(2.5*2**30),'VRAM_CAP'
    group=micro[gs:gs+2];opt.zero_grad(set_to_none=True)
    for rr in group:
     x,y,l,tl=batch(rr,data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);assert torch.isfinite(loss),'NONFINITE_LOSS';(loss/len(group)).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm),'NONFINITE_GRADIENT';opt.step();steps+=1;norms.append(float(norm))
   if interrupted:break
   current_epoch=epoch
   if epoch==1 or epoch%cfg['evaluation_every_epochs']==0:
    last=evaluate();last_eval_epoch=epoch;save(RUN/f'predictions/epoch_{epoch:03d}.json',last);metric(epoch,last,float(np.mean(norms)),time.perf_counter()-tick)
    passed=last['WER']<=cfg['pass_WER'] and last['CTC_loss']<=cfg['pass_loss_fraction_of_initial']*initial['CTC_loss'] and initial['WER']-last['WER']>=cfg['minimum_absolute_WER_drop'] and max(last['family_WER'].values())<=cfg['max_family_WER'] and last['rare_bearing_WER']<=cfg['max_rare_bearing_WER'] and last['upper_input_quartile_WER']<=cfg['max_long_input_WER']
    collapse=last['blank_fraction']>=.995 or last['top_token_fraction']>=.98 and last['token_diversity']<=3
    collapsed=collapsed+1 if collapse and last['CTC_loss']>.7*initial['CTC_loss'] else 0
    lastcp=checkpoint(f'epoch_{epoch:03d}.pt',epoch)
    if passed:reason='PREDECLARED_LEARNING_GATE_PASS';break
    if epoch>=30 and collapsed>=2:reason='PERSISTENT_COLLAPSE_WITH_STAGNANT_LOSS';break
  # Always score the actual final weights, including any budget-limited partial epoch.
  final=evaluate();save(RUN/'final_predictions.json',final)
  if current_epoch!=last_eval_epoch or reason=='WALL_OR_STEP_BUDGET':metric(current_epoch,final,None,0.)
  if not lastcp:lastcp=checkpoint('final_state.pt',current_epoch)
  save(RUN/'decision.json',dict(status='PASS' if passed else 'FAIL',reason=reason,compact_control_learned=passed,complete_epochs=current_epoch,last_periodic_evaluated_epoch=last_eval_epoch,optimizer_steps=steps,initial_WER=initial['WER'],final_WER=final['WER'],initial_CTC=initial['CTC_loss'],final_CTC=final['CTC_loss'],final_blank_fraction=final['blank_fraction'],final_prediction_reference_length_ratio=final['prediction_reference_length_ratio'],final_token_diversity=final['token_diversity'],family_WER=final['family_WER'],rare_bearing_WER=final['rare_bearing_WER'],upper_input_quartile_WER=final['upper_input_quartile_WER'],elapsed_including_initial_and_final_scoring_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint=lastcp,full_training_started=False,TEST_used=False,promote=False))
 except BaseException as e:
  save(RUN/'failure.json',dict(error=str(e),type=type(e).__name__,optimizer_steps=steps,complete_epochs=current_epoch,preserve_all_artifacts=True));raise
 finally:
  before=read(RUN/'control_input_hashes_before.json');after=snap(before)
  changed=[p for p in before if before[p]!=after[p]]
  save(RUN/'control_input_integrity.json',dict(status='PASS' if not changed else 'FAIL',files=len(before),changed=changed,after=after));assert not changed
if __name__=='__main__':
 args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');args.add_argument('--train',action='store_true');a=args.parse_args()
 if a.prepare:prepare()
 elif a.train:fit()
 else:args.error('Specify --prepare or --train')

