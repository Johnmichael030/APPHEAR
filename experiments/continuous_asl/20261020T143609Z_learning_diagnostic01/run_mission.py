"""Research-only gated CUDA diagnostic/candidate. No acquisition, production writes or TEST paths."""
from pathlib import Path
from collections import Counter
import argparse,csv,ctypes,hashlib,json,math,os,random,shutil,sys,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8') # process-local only
import numpy as np
import torch
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];TREE=RUN.parent
OLD=TREE/'20261020T133148Z_source_transfer_pilot01';REVIEW=TREE/'20261020T141110Z_alignment_observability_review01'
sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
from ctc_experiment import CausalCTC,batch,decode,edit_counts,seeds
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'
BEFORE={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def track(p,expected=None):
 p=Path(p)
 if str(p) not in BEFORE:
  s=p.stat();v=dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
  if expected:assert v['sha256']==expected,('HASH',str(p))
  BEFORE[str(p)]=v
 return p
def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def save(name,value):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def append(name,row):
 p=RUN/name
 with p.open('a',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row))
  if f.tell()==0:w.writeheader()
  w.writerow(row)
def emit(v):print(json.dumps(v),flush=True)
def family(r):
 import re
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group']);return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def aggregate(pred):
 c=Counter();tokens=Counter()
 for p in pred:
  c.update({k:p[k] for k in ['substitutions','deletions','insertions','reference_tokens']});tokens.update(p['hypothesis'])
 frames=sum(p['output_frames'] for p in pred)
 return dict(WER=sum(c[k] for k in ['substitutions','deletions','insertions'])/c['reference_tokens'],CTC_loss=float(np.mean([p['CTC_loss'] for p in pred])),**dict(c),blank_fraction=sum(p['blank_frames'] for p in pred)/frames,token_diversity=len(tokens),top_token_fraction=tokens.most_common(1)[0][1]/sum(tokens.values()) if tokens else 0.,empty_outputs=sum(not p['hypothesis'] for p in pred),multi_token_outputs=sum(len(p['hypothesis'])>=2 for p in pred),utterances=len(pred),predictions=pred)
def safe():
 free,_=torch.cuda.mem_get_info()
 assert free>=256*2**20,'FREE_VRAM_STOP'
 assert torch.cuda.max_memory_allocated()<=int(2.5*2**30),'VRAM_CAP_STOP'
 assert shutil.disk_usage(RUN).free>=10*2**30,'DISK_HEADROOM_STOP'
def score(model,rows,data,vocab,criterion):
 model.eval();pred=[]
 with torch.inference_mode():
  for r in rows:
   x,y,l,tl=batch([r],data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
   assert torch.isfinite(loss)
   hyp=decode(z[0],vocab);ids=z[0].argmax(-1)
   pred.append(dict(sample_id=r['sample_id'],reference=r['glosses'],hypothesis=hyp,**edit_counts(r['glosses'],hyp),CTC_loss=float(loss),blank_frames=int((ids==0).sum()),output_frames=len(ids)))
 return aggregate(pred)
def cp(path,model,opt,epoch,extra):
 assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),schema=SCHEMA,vocabulary=VOCAB,config_sha256=sha(RUN/'config.json'),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),**extra),path)
 return dict(path=str(path.relative_to(RUN)),sha256=sha(path),epoch=epoch)
def fit_epoch(model,opt,rows,data,vocab,criterion,deadline):
 model.train();order=list(rows);random.shuffle(order);losses=[];norms=[];groups=0;tick=time.perf_counter()
 micro=[order[i:i+4] for i in range(0,len(order),4)]
 for start in range(0,len(micro),2):
  if time.perf_counter()>=deadline:return None
  safe();group=micro[start:start+2];opt.zero_grad(set_to_none=True)
  for rr in group:
   x,y,l,tl=batch(rr,data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
   assert torch.isfinite(loss),'NONFINITE_CTC'
   (loss/len(group)).backward();losses.append(float(loss.detach()))
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm),'NONFINITE_GRAD'
  opt.step();groups+=1;norms.append(float(norm))
 torch.cuda.synchronize()
 return dict(optimization_CTC=float(np.mean(losses)),gradient_norm_mean=float(np.mean(norms)),optimizer_updates=groups,fit_seconds=time.perf_counter()-tick)
def newmodel(mean,std):
 m=CausalCTC(len(VOCAB),48,mean,std).to('cuda:0');o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);return m,o
def main():
 global VOCAB
 args=argparse.ArgumentParser();args.add_argument('--resume-candidate',type=Path);args=args.parse_args()
 tick=time.perf_counter()
 for p in [REVIEW/'ALIGNMENT_OBSERVABILITY_REPORT.md',OLD/'PILOT_REPORT.md',OLD/'training_status.json',OLD/'config.json',OLD/'common.py',OLD/'train.py',ROOT/'tools/continuous_asl/ctc_experiment.py',ROOT/'tools/continuous_asl/prepare_features.py',Path(__file__)]:track(p)
 oldcfg=read(OLD/'config.json');status=read(OLD/'training_status.json')
 sourcecp=track(OLD/status['selected_checkpoint']['path'],status['selected_checkpoint']['sha256'])
 fold=read(OLD/'fold_manifest.json');assert sha(OLD/'fold_manifest.json')==oldcfg['fold_manifest_sha256']
 resolved=read(OLD/'resolved_manifest.json');lookup={r['sample_id']:r for r in resolved}
 train=[lookup[i] for i in fold['train_sample_ids']];dev=[lookup[i] for i in fold['compatible_eval_sample_ids']];strong=set(fold['support5_eval_sample_ids'])
 assert (len(train),len(dev),len(strong))==(1500,80,27)
 assert all(r['split']=='train' for r in train+dev)
 assert all(r['internal_role']=='TRAIN' for r in train) and all(r['internal_role']=='HELD_OUT' for r in dev)
 assert not ({family(r) for r in train}&{family(r) for r in dev})
 for key in ['sample_id','source_group','raw_sha256','decoded_pixel_sha256']:
  assert not ({r[key] for r in train}&{r[key] for r in dev}),key
 u=track(fold['universe_manifest'],fold['universe_manifest_sha256'])
 original={r['sample_id']:r for r in read(u) if r['split']=='train'}
 assert all(r['sample_id'] in original and r['glosses']==original[r['sample_id']]['glosses'] for r in train+dev)
 vocabfile=track(OLD/'vocabulary.json',oldcfg['vocabulary_sha256']);VOCAB=read(vocabfile)['tokens']
 support=Counter(t for r in train for t in set(r['glosses']))
 assert VOCAB==['<BLANK>','<UNK>']+sorted(support) and len(VOCAB)==2289
 assert all(t in support and VOCAB.index(t)>=2 for r in train+dev for t in r['glosses'])
 norm=track(OLD/'normalization.npz',oldcfg['normalization_sha256'])
 with np.load(norm,allow_pickle=False) as z:mean=z['mean'].copy();std=z['std'].copy()
 assert mean.shape==std.shape==(378,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std>0)
 data={}
 for r in train+dev:
  p=track(r['feature_path'],r['feature_sha256'])
  with np.load(p,allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA
   x=z['X'].copy();ts=z['timestamp_ms'];meta=json.loads(str(z['metadata_json'].item()))
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
   assert meta['sample_id']==r['sample_id'] and meta['glosses']==r['glosses']
  minimum=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]));assert len(x)>=minimum
  data[r['sample_id']]=x
 assert torch.cuda.is_available(),'CUDA_UNAVAILABLE'
 seeds(20261020);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 env=dict(GPU=torch.cuda.get_device_name(0),GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,GPU_free_bytes=torch.cuda.mem_get_info()[0],torch=torch.__version__,CUDA=torch.version.cuda,CUDA_available=True,CPU_threads=os.cpu_count(),disk_free_bytes=shutil.disk_usage(RUN).free)
 if not args.resume_candidate:
  save('environment.json',env)
  # Source-/family-balanced deterministic 16-example subset, independent of predictions.
  pools={}
  for r in train:pools.setdefault(family(r),[]).append(r)
  for rr in pools.values():rr.sort(key=lambda r:hashlib.sha256(('subset-20261020|'+r['sample_id']).encode()).hexdigest())
  subset=[]
  while len(subset)<16:
   for k in sorted(pools):
    if pools[k] and len(subset)<16:subset.append(pools[k].pop(0))
  monitorids=[r['sample_id'] for r in read(OLD/'train_monitor_manifest.json')];assert len(monitorids)==128 and set(monitorids)<=set(fold['train_sample_ids'])
  cfg=dict(schema=SCHEMA,seed=20261020,channels=48,batch=4,accumulation=2,accumulation_fix='loss divided by actual number of microbatches in preplanned group; singleton tail divisor1',lr=.001,weight_decay=.0001,gradient_clip=5.,mixed_precision=False,subset_seed='subset-20261020',subset_epochs=200,subset_seconds=60,subset_evaluate_every=10,subset_pass='WER<=20%, loss<=50%initial, and >=10 percentage-point WER improvement; or initial/final WER both<=20% with no deterioration and loss halved',candidate_max_epochs=20,candidate_seconds=180,candidate_patience_evaluations=4,candidate_min_improvement=.005,candidate_eval_epochs=[1,2,4,6,8,10,12,14,16,18,20],collapse_gate='from epoch6, two evaluations with TRAIN and DEV blank>=.995 or top-token fraction>=.98 and diversity<=3',selection='primary80 tuple(WER,CTC loss); strong27 descriptive only',credible_improvement='>=.005 absolute WER better than prior primary80 AND no failure of TRAIN screen; smaller gains descriptive',training_screen='selected fullTRAIN WER<=50%, diversity>=10, multi-token outputs>=25%',vocabulary_sha256=sha(vocabfile),normalization_sha256=sha(norm),fold_sha256=sha(OLD/'fold_manifest.json'),source_checkpoint_sha256=sha(sourcecp),prior_epoch12_WER=status['primary80_WER'],source_model_decoder_sha256=sha(ROOT/'tools/continuous_asl/ctc_experiment.py'),script_sha256=sha(__file__),TEST_used=False,production_promoted=False)
  save('config.json',cfg);save('subset_manifest.json',[r['sample_id'] for r in subset]);save('train_monitor_ids.json',monitorids)
  for name in ['fold_manifest.json','vocabulary.json','normalization.npz']:shutil.copyfile(OLD/name,RUN/name)
  save('resolved_manifest.json',[{**r,'internal_role':'TRAIN' if r in train else 'HELD_OUT_COMPATIBLE'} for r in train+dev])
  save('input_integrity_before.json',BEFORE)
  # Arithmetic unit check of fixed denominator; no data/labels changed.
  assert [min(2,375-i) for i in range(0,375,2)][-1]==1
  # Disposable selected-checkpoint smoke: source bytes stay untouched.
  model,opt=newmodel(mean,std);state=torch.load(sourcecp,map_location='cuda:0',weights_only=False)
  assert state['vocabulary']==VOCAB and state['fold_manifest_sha256']==cfg['fold_sha256']
  model.load_state_dict(state['model']);del state
  assert np.array_equal(model.mean.detach().cpu().numpy(),mean) and np.array_equal(model.std.detach().cpu().numpy(),std)
  torch.cuda.reset_peak_memory_stats()
  beforeparam=next(model.parameters()).detach().clone();x,y,l,tl=batch(train[:4],data,VOCAB,'cuda:0');z=model(x)
  loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);loss.backward()
  normval=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(loss) and torch.isfinite(normval)
  opt.step();assert not torch.equal(beforeparam,next(model.parameters()))
  save('gate_A.json',dict(status='PASS',source_checkpoint_strict_load=True,feature_records_checked=len(data),blank_id=0,unknown_id=1,CUDA_forward_backward_optimizer_step=True,loss=float(loss.detach()),gradient_norm=float(normval),peak_allocated_bytes=torch.cuda.max_memory_allocated(),tail_group_count=1,source_checkpoint_unchanged=sha(sourcecp)==cfg['source_checkpoint_sha256'],preflight_seconds=time.perf_counter()-tick))
  del model,opt,beforeparam,x,y,z,loss;torch.cuda.empty_cache()
  # Gate C arithmetic only: no native-cadence landmarks exist or are generated.
  with track(REVIEW/'review_manifest.csv').open(encoding='utf-8-sig',newline='') as f:reviews=list(csv.DictReader(f))
  cadence=[]
  for r in reviews:
   rec=lookup[r['sample_id']];a=int(r['relative_start_frame']);b=int(r['relative_end_frame']);fps=rec['native_fps']
   old=list(range(0,rec['decoded_frames'],3));new=list(range(rec['decoded_frames']))
   cadence.append(dict(review_id=r['review_id'],sample_id=r['sample_id'],group=r['group'],current_inside=[i for i in old if a<=i<=b],native_inside=[i for i in new if a<=i<=b],native_timestamp_ms=[i*1000/fps for i in new if a<=i<=b],actual_new_hand_observability='UNKNOWN_NO_NEW_PERCEPTION',metadata_only=True))
  save('cadence_metadata_assessment.json',dict(status='PARTIAL',events=cadence,current_sampled_fps=30000/1001/3,alternative_fps=30000/1001,features_regenerated=False,recognition_improvement_tested=False,qualified_visual_review='BLOCKED'))
  # Gate B starts fresh and never reads DEV.
  seeds(cfg['seed']);model,opt=newmodel(mean,std);torch.cuda.reset_peak_memory_stats()
  initial=score(model,subset,data,VOCAB,criterion);save('subset_initial_predictions.json',initial)
  emit(dict(stage='SUBSET_STARTED',initial_WER=initial['WER'],initial_CTC=initial['CTC_loss'],samples=16))
  t=time.perf_counter();deadline=t+cfg['subset_seconds'];passed=False;subepoch=0;last=initial;subupdates=0
  for epoch in range(1,cfg['subset_epochs']+1):
   result=fit_epoch(model,opt,subset,data,VOCAB,criterion,deadline)
   if result is None:break
   subupdates+=result['optimizer_updates'];subepoch=epoch
   if epoch%10==0 or epoch==1:
    last=score(model,subset,data,VOCAB,criterion)
    row=dict(epoch=epoch,elapsed_seconds=time.perf_counter()-t,WER=last['WER'],CTC_loss=last['CTC_loss'],blank_fraction=last['blank_fraction'],token_diversity=last['token_diversity'],empty_outputs=last['empty_outputs'],top_token_fraction=last['top_token_fraction'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    append('subset_learning_log.csv',row);save(f'subset_predictions/epoch_{epoch:03d}.json',last)
    passed=last['WER']<=.2 and last['CTC_loss']<=.5*initial['CTC_loss'] and (initial['WER']-last['WER']>=.1 or initial['WER']<=.2 and last['WER']<=initial['WER'])
    emit(dict(stage='SUBSET',**row,gate_pass=passed))
    if passed:break
    if epoch>=100 and last['blank_fraction']>=.995 and last['CTC_loss']>.7*initial['CTC_loss']:break
  subsetcp=cp(RUN/'subset_checkpoint.pt',model,opt,subepoch,dict(stage='subset',optimizer_updates=subupdates))
  save('gate_B.json',dict(status='PASS' if passed else 'FAIL',initial={k:v for k,v in initial.items() if k!='predictions'},final={k:v for k,v in last.items() if k!='predictions'},completed_epochs=subepoch,elapsed_seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint=subsetcp,DEV_used=False))
  del model,opt;torch.cuda.empty_cache()
  if not passed:
   save('decision.json',dict(status='FAIL',reason='FIXED_SUBSET_LEARNING_GATE_FAILED',full_candidate_trained=False,promote=False));return
 else:
  cfg=json.loads((RUN/'config.json').read_text());assert cfg['script_sha256']==sha(__file__)
  assert json.loads((RUN/'gate_B.json').read_text())['status']=='PASS'
  monitorids=json.loads((RUN/'train_monitor_ids.json').read_text())
 # Gate D: independent fresh initialization. Never warm-start from subset/source trained weights.
 seeds(cfg['seed']);model,opt=newmodel(mean,std);monitor=[lookup[i] for i in monitorids]
 start_epoch=1;best=(float('inf'),float('inf'));bestptr=None;credible=1.;stale=0;collapse=0;updates=0;spent=0.
 if args.resume_candidate:
  p=args.resume_candidate.resolve();assert p.parent==RUN/'candidate_checkpoints';s=torch.load(p,map_location='cuda:0',weights_only=False);assert s['config_sha256']==sha(RUN/'config.json')
  model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start_epoch=s['epoch']+1
  random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
  best=tuple(s['best']);bestptr=s['best_pointer'];credible=s['credible'];stale=s['stale'];collapse=s['collapse'];updates=s['updates'];spent=s['elapsed_seconds'];del s
 else:
  init=cp(RUN/'candidate_initialization.pt',model,opt,0,dict(stage='fresh_candidate',optimizer_updates=0))
  save('candidate_initialization.json',dict(**init,fresh=True,trained_weights_reused=False))
  save('candidate_initial_monitor.json',score(model,monitor,data,VOCAB,criterion));save('candidate_initial_primary80.json',score(model,dev,data,VOCAB,criterion))
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter();deadline=started+max(0,cfg['candidate_seconds']-spent);reason='MAX_EPOCHS';completed=start_epoch-1
 emit(dict(stage='CANDIDATE_STARTED',TRAIN=1500,primary=80,from_scratch=not bool(args.resume_candidate),max_epochs=20,max_seconds=180))
 for epoch in range(start_epoch,cfg['candidate_max_epochs']+1):
  epochstart=time.perf_counter();result=fit_epoch(model,opt,train,data,VOCAB,criterion,deadline)
  if result is None:reason='WALL_BUDGET_PARTIAL_EPOCH';break
  completed=epoch;updates+=result['optimizer_updates'];dv=None;sg=None;selected=False
  fulltrain=score(model,train,data,VOCAB,criterion)
  tm=aggregate([p for p in fulltrain['predictions'] if p['sample_id'] in set(monitorids)])
  save(f'candidate_predictions/train1500_epoch_{epoch:03d}.json',fulltrain)
  if epoch in cfg['candidate_eval_epochs']:
   dv=score(model,dev,data,VOCAB,criterion);sg=aggregate([p for p in dv['predictions'] if p['sample_id'] in strong])
   for name,v in [('monitor',tm),('primary80',dv),('strong27',sg)]:save(f'candidate_predictions/{name}_epoch_{epoch:03d}.json',v)
   rank=(dv['WER'],dv['CTC_loss']);selected=rank<best
   if selected:best=rank
   if dv['WER']<=credible-.005+1e-12:credible=dv['WER'];stale=0
   else:stale+=1
   blank=tm['blank_fraction']>=.995 and dv['blank_fraction']>=.995
   token=tm['top_token_fraction']>=.98 and dv['top_token_fraction']>=.98 and tm['token_diversity']<=3 and dv['token_diversity']<=3
   collapse=collapse+1 if blank or token else 0
   if epoch>=6 and collapse>=2:reason='PERSISTENT_COLLAPSE'
   elif epoch>=6 and stale>=cfg['candidate_patience_evaluations']:reason='NO_CREDIBLE_PRIMARY_GAIN'
  path=RUN/'candidate_checkpoints'/f'epoch_{epoch:03d}.pt'
  # Resolve selected path before saving resumable state; checkpoint SHA is recorded after save.
  if selected:bestptr=dict(path=str(path.relative_to(RUN)),epoch=epoch)
  pointer=cp(path,model,opt,epoch,dict(stage='candidate',best=best,best_pointer=bestptr,credible=credible,stale=stale,collapse=collapse,updates=updates,elapsed_seconds=spent+time.perf_counter()-started))
  record=dict(epoch=epoch,elapsed_seconds=spent+time.perf_counter()-started,fit_seconds=result['fit_seconds'],epoch_seconds=time.perf_counter()-epochstart,optimization_CTC_loss=result['optimization_CTC'],gradient_norm_mean=result['gradient_norm_mean'],lr=.001,optimizer_updates=updates,skipped_invalid_batches=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_sha256=pointer['sha256'],selection_reason='PRIMARY80_WER_CTC_MINIMUM' if selected else 'NOT_SELECTED',CUDA_error='')
  for prefix,v in [('TRAIN',fulltrain),('TRAIN_monitor',tm),('heldout80',dv),('strong27',sg)]:
   for k in ['WER','CTC_loss','substitutions','deletions','insertions','reference_tokens','blank_fraction','token_diversity','empty_outputs','top_token_fraction','multi_token_outputs']:record[prefix+'_'+k]=v[k] if v else None
  append('training_metrics.csv',record);emit(dict(stage='EPOCH',epoch=epoch,monitor_WER=tm['WER'] if tm else None,primary80_WER=dv['WER'] if dv else None,elapsed_seconds=record['elapsed_seconds'],stop=reason))
  if epoch==1:save('runtime_estimate_after_epoch1.json',dict(measured_epoch_seconds=record['epoch_seconds'],estimated_20_epoch_seconds=20*record['epoch_seconds'],hard_budget_seconds=180,estimate_not_measurement=True))
  if reason!='MAX_EPOCHS':break
 if bestptr is None:raise RuntimeError('NO_COMPLETED_EVALUATED_CANDIDATE')
 selectedpath=RUN/bestptr['path'];bestptr['sha256']=sha(selectedpath)
 state=torch.load(selectedpath,map_location='cuda:0',weights_only=False);model.load_state_dict(state['model']);del state
 selectedtrain=score(model,train,data,VOCAB,criterion);selecteddev=score(model,dev,data,VOCAB,criterion);selectedstrong=aggregate([p for p in selecteddev['predictions'] if p['sample_id'] in strong])
 for name,v in [('train1500',selectedtrain),('primary80',selecteddev),('strong27',selectedstrong)]:save('selected_'+name+'.json',v)
 priorpred=read(OLD/'selected_primary80.json');assert selecteddev['reference_tokens']==priorpred['reference_tokens']==485
 assert [(p['sample_id'],p['reference']) for p in selecteddev['predictions']]==[(p['sample_id'],p['reference']) for p in priorpred['predictions']]
 trainpass=selectedtrain['WER']<=.5 and selectedtrain['token_diversity']>=10 and selectedtrain['multi_token_outputs']>=.25*len(train)
 gain=status['primary80_WER']-selecteddev['WER']
 result=dict(status='PARTIAL',reason=reason,subset_learning_pass=True,full_candidate_trained=True,completed_epochs=completed,selected_checkpoint=bestptr,TRAIN_WER=selectedtrain['WER'],primary80_WER=selecteddev['WER'],strong27_WER=selectedstrong['WER'],primary_reference_tokens=485,absolute_WER_gain_percentage_points=100*gain,relative_WER_gain_percent=100*gain/status['primary80_WER'],credible_gain=bool(gain>=.005 and trainpass),TRAIN_learning_screen_pass=trainpass,training_loop_seconds=spent+time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),comparison='Equivalent primary IDs/labels/vocabulary/normalization/decoder; only accumulation-tail correction; selection protocol same',promote=False,TEST_used=False)
 save('decision.json',result);emit(result)
def integrity():
 after={};changed=[]
 for p,v in BEFORE.items():
  q=Path(p);s=q.stat();after[p]=dict(sha256=sha(q),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
  if v!=after[p]:changed.append(p)
 save('input_integrity.json',dict(status='PASS' if not changed else 'FAIL',existing_files_checked=len(BEFORE),before=BEFORE,after=after,changed=changed,TEST_used=False,scope='Declared inputs only; no production or entire-workspace hash claim'))
 assert not changed,'INPUT_CHANGED'
if __name__=='__main__':
 try:main()
 except BaseException as e:
  traceback.print_exc()
  if not (RUN/'failure.json').exists():save('failure.json',dict(error=str(e),type=type(e).__name__,preserve_completed_artifacts=True))
  raise
 finally:integrity()

