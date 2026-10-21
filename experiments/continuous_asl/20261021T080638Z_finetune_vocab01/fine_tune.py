"""Isolated, resumable epoch12 fine-tuning; never writes app, old experiments or TEST."""
from pathlib import Path
import argparse,csv,hashlib,json,os,random,shutil,subprocess,sys,threading,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
import numpy as np
import torch
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2]
BASE=ROOT/'experiments/continuous_asl/20261020T143609Z_learning_diagnostic01'
EVID=ROOT/'evidence/20261021T062258Z_continuous_detectionfix01'
SOURCE=BASE/'candidate_checkpoints/epoch_012.pt'
sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
from ctc_experiment import CausalCTC,batch,decode,edit_counts,seeds
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def emit(v):print(json.dumps(v),flush=True)
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def family(r):
 import re
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
 return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def load_rows():
 fold=read(BASE/'fold_manifest.json');lookup={r['sample_id']:r for r in read(BASE/'resolved_manifest.json')}
 train=[lookup[i] for i in fold['train_sample_ids']];dev=[lookup[i] for i in fold['compatible_eval_sample_ids']]
 assert len(train)==1500 and len(dev)==80
 assert not ({family(r) for r in train}&{family(r) for r in dev})
 assert not ({r['sample_id'] for r in train}&set(fold['internal_eval_sample_ids']))
 for key in ['source_group','raw_sha256','decoded_pixel_sha256']:
  assert not ({r[key] for r in train}&{r[key] for r in dev}),key
 assert {r['training_pool'] for r in train+dev}=={'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE'}
 vocab=read(BASE/'vocabulary.json')['tokens']
 assert vocab==read(ROOT/'packages/continuous_asl_research_01/android_assets/continuous_asl/vocabulary.json')
 assert vocab[:2]==['<BLANK>','<UNK>'] and len(vocab)==2289
 data={};times={};inventory=[]
 for r in train+dev:
  p=Path(r['feature_path']);actual=sha(p);assert actual==r['feature_sha256']
  with np.load(p,allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA
   x=z['X'].copy();ts=z['timestamp_ms'].copy();meta=json.loads(str(z['metadata_json'].item()))
  assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
  assert meta['glosses']==r['glosses'] and meta['sample_id']==r['sample_id']
  assert r['glosses'] and all(t in vocab[2:] for t in r['glosses'])
  assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
  data[r['sample_id']]=x;times[r['sample_id']]=ts
  inventory.append(dict(path=str(p),sha256=actual,sample_id=r['sample_id'],role='TRAIN' if r in train else 'DEV',observations=len(x)))
 return train,dev,vocab,data,times,inventory

def load_model(vocab,checkpoint=SOURCE):
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert state['schema']==SCHEMA and state['vocabulary']==vocab
 m=CausalCTC(len(vocab),48,state['model']['mean'].numpy(),state['model']['std'].numpy()).to('cuda:0')
 m.load_state_dict(state['model'],strict=True)
 return m,state

def aggregate(pred):
 from collections import Counter
 counts=Counter();tokens=Counter()
 for p in pred:
  counts.update({k:p[k] for k in ['substitutions','deletions','insertions','reference_tokens']});tokens.update(p['hypothesis'])
 frames=sum(p['output_frames'] for p in pred)
 return dict(WER=sum(counts[k] for k in ['substitutions','deletions','insertions'])/counts['reference_tokens'],CTC_loss=float(np.mean([p['CTC_loss'] for p in pred])),**dict(counts),blank_fraction=sum(p['blank_frames'] for p in pred)/frames,token_diversity=len(tokens),top_token_fraction=(tokens.most_common(1)[0][1]/sum(tokens.values()) if tokens else 0),empty_outputs=sum(not p['hypothesis'] for p in pred),exact_sequences=sum(p['hypothesis']==p['reference'] for p in pred),utterances=len(pred),predictions=pred)

def score(m,rows,data,vocab,cache=None):
 m.eval();pred=[];criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 with torch.inference_mode():
  for r in rows:
   x,y,l,tl=batch([r],data,vocab,'cuda:0');z=m(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
   assert torch.isfinite(loss)
   hyp=decode(z[0],vocab);ids=z[0].argmax(-1)
   p=dict(sample_id=r['sample_id'],reference=r['glosses'],hypothesis=hyp,**edit_counts(r['glosses'],hyp),CTC_loss=float(loss),blank_frames=int((ids==0).sum()),output_frames=len(ids))
   if cache is not None:
    a=z[0].detach().cpu().numpy();cache[r['sample_id']]=a
    probabilities=z[0].softmax(-1);p.update(mean_blank_probability=float(probabilities[:,0].mean()),mean_max_probability=float(probabilities.max(-1).values.mean()))
   pred.append(p)
 return aggregate(pred)

def streaming_score(m,rows,data,times,vocab):
 pred=[];m.eval()
 with torch.inference_mode():
  for r in rows:
   x=data[r['sample_id']];ts=times[r['sample_id']];seen=np.any(x[:,84:126]>0,axis=1);ids=np.zeros(len(x),np.int64)
   k=0
   while k<len(x):
    if not seen[k]:k+=1;continue
    end=k+1
    while end<len(x) and seen[end]:end+=1
    block=x[k:end].copy();block[0,215:378]=0
    ids[k:end]=m(torch.as_tensor(block[None],device='cuda:0'))[0].argmax(-1).cpu().numpy();k=end
   previous=0;emitted=True;since=0.;out=[]
   for token,t,visible in zip(ids,ts,seen):
    if len(out)>=128:break
    if not visible:previous=0;emitted=True;continue
    token=int(token)
    if token!=previous:
     if not emitted:out.append(vocab[previous])
     previous=token;since=float(t);emitted=token==0
    elif not emitted and t-since>=200.:out.append(vocab[previous]);emitted=True
   pred.append(dict(sample_id=r['sample_id'],reference=r['glosses'],hypothesis=out,**edit_counts(r['glosses'],out)))
 from collections import Counter
 c=Counter()
 for p in pred:c.update({k:p[k] for k in ['substitutions','deletions','insertions','reference_tokens']})
 return dict(WER=sum(c[k] for k in ['substitutions','deletions','insertions'])/c['reference_tokens'],**dict(c),utterances=len(pred),exact_sequences=sum(p['reference']==p['hypothesis'] for p in pred),predictions=pred,nohand_publications=0,scope='Cached10Hz current reset/discard/200ms display policy; no oracle end flush; NOT live/unseen-signer accuracy')

def prepare():
 assert not (RUN/'BASELINE_AND_ERROR_ANALYSIS.md').exists(),'PREPARE_ALREADY_COMPLETE'
 train,dev,vocab,data,times,inventory=load_rows();assert torch.cuda.is_available()
 assert sha(SOURCE)=='14d0e4669674265ea2be8a3161fa6699e4f813daf4b97b384863dce1daa038ff'
 seeds(20261020);m,state=load_model(vocab)
 normal=np.load(BASE/'normalization.npz',allow_pickle=False)
 assert np.array_equal(state['model']['mean'].numpy(),normal['mean']) and np.array_equal(state['model']['std'].numpy(),normal['std'])
 protected={str(p):sha(p) for p in [SOURCE,BASE/'vocabulary.json',BASE/'config.json',BASE/'fold_manifest.json',BASE/'normalization.npz',BASE/'resolved_manifest.json',ROOT/'tools/continuous_asl/ctc_experiment.py',ROOT/'tools/continuous_asl/prepare_features.py',ROOT/'android/app/src/main/java/com/example/apphear/continuousasl/ContinuousAslCore.kt',ROOT/'android/app/src/main/java/com/example/apphear/RecognitionActivity.kt',EVID/'AppHear-continuous-refined-20261021.apk']}
 for item in read(EVID/'preservation_before.json'):protected[item['path']]=item['sha256'];assert sha(item['path'])==item['sha256']
 save(RUN/'protected_hashes.json',protected);save(RUN/'input_features.json',inventory)
 for name,rows in [('TRAIN1500',train),('DEV80',dev)]:save(RUN/'manifests'/f'{name}.json',rows)
 save(RUN/'vocabulary.json',{'tokens':vocab,'blank_id':0,'unknown_id':1,'mapping':'unchanged from epoch12'})
 env=dict(torch=torch.__version__,CUDA=torch.version.cuda,available=torch.cuda.is_available(),GPU=torch.cuda.get_device_name(0),VRAM_bytes=torch.cuda.get_device_properties(0).total_memory,free_storage_bytes=shutil.disk_usage(RUN).free)
 save(RUN/'environment.json',env)
 cache={};dv=score(m,dev,data,vocab,cache);old=read(BASE/'selected_primary80.json')
 assert [(p['sample_id'],p['reference'],p['hypothesis']) for p in dv['predictions']]==[(p['sample_id'],p['reference'],p['hypothesis']) for p in old['predictions']],'BASELINE_PREDICTION_DRIFT'
 save(RUN/'baseline_greedy_DEV80.json',dv);np.savez_compressed(RUN/'baseline_logits_DEV80.npz',**cache)
 stream=streaming_score(m,dev,data,times,vocab)
 assert [stream[k] for k in ['substitutions','deletions','insertions','reference_tokens']]==[176,288,0,485]
 save(RUN/'baseline_streaming_DEV80.json',stream)
 save(RUN/'baseline_temporal_metadata.json',[dict(sample_id=r['sample_id'],timestamps_ms=times[r['sample_id']].tolist(),publisher_occurrence_intervals=r.get('individual_sign_times_ms'),status='Published occurrence intervals, not independently reviewed physical sign boundaries') for r in dev])
 monitor_ids=read(BASE/'train_monitor_ids.json');assert len(monitor_ids)==128 and set(monitor_ids)<={r['sample_id'] for r in train}
 save(RUN/'monitor128_ids.json',monitor_ids)
 config=dict(experiment='A_existing_class_finetune',initialization=str(SOURCE),initialization_sha256=sha(SOURCE),schema_version=SCHEMA,seed=20261020,classes=2289,channels=48,blank_id=0,unknown_id=1,feature_width=378,learning_rate=0.0001,parent_learning_rate=0.001,optimizer='AdamW; restore epoch12 moments, change only LR for warm-start fine-tuning',weight_decay=.0001,batch_size=4,gradient_accumulation=2,accumulation_tail='divide by actual number of microbatches in group',gradient_clip=5.,max_epochs=12,wall_seconds=180,evaluate_epochs=[1,2,4,6,8,10,12],early_stopping_stale_evaluations=4,minimum_credible_gain=.005,collapse_after_epoch=4,collapse_two_checks='TRAINmonitor andDEV blank>=.995 OR top_token_fraction>=.98 and diversity<=3',selection='DEV80 greedy tuple(WER,CTC_loss); original27 subgroup descriptive only',normalization='frozen epoch12 mean/std and clamp; no refit',mixed_precision=False,VRAM_cap_bytes=int(2.5*2**30),min_free_VRAM_bytes=256*2**20,promising_for_C='DEV greedy WER >=.02 absolute better than epoch12; no >.02 insertion-rate regression; TRAIN WER no >.05 regression',production_gate='No automatic promotion: independent signer/no-sign/boundary/live-latency gates also required',manifest_sha256={n:sha(RUN/'manifests'/f'{n}.json') for n in ['TRAIN1500','DEV80']},vocabulary_sha256=sha(RUN/'vocabulary.json'),TEST_used=False)
 save(RUN/'A/TRAINING_CONFIGURATION.json',config)
 save(RUN/'PREPARE_PASS.json',dict(status='PASS',source_checkpoint_strict_load=True,old_class_ids_unchanged=True,baseline_predictions_exact=True,baseline_streaming_edits_exact=True,train_features_verified=1500,dev_features_verified=80,CTC_feasible=True,no_blank_or_UNK_targets=True,family_overlap=False,known_signer_development=True))
 report=f'''# Baseline and error analysis\n\nDate:2026-10-21 (Asia/Shanghai). PASS computational baseline; recognition PARTIAL.\n\nThis file was written BEFORE any new optimizer update. Source is frozen epoch12 SHA256 {sha(SOURCE)}. Architecture:378D input,48 channels, four causal blocks with two kernel3 convolutions each, dilations1/2/4/8, receptive field61;2289 logits. CTC blank0/UNK1; collapse consecutive IDs before removing blanks. Frozen normalization is embedded in model forward and not refitted.\n\nAll1500TRAIN and80DEV feature hashes/schema/chronology/embedded literal targets/CTC lengths pass. TRAIN has3publisher signer IDs;80DEV uses two known signers in disjoint recording families. The reserved17TEST set was not accessed. Original manifests are unchanged. Exact new manifest hashes are in A/TRAINING_CONFIGURATION.json.\n\nFull-sequence greedy DEV predictions reproduce the original saved checkpoint predictions exactly:WER{dv['WER']:.9%},S{dv['substitutions']}/D{dv['deletions']}/I{dv['insertions']},485references,{dv['exact_sequences']}/80exact. Current streaming-policy replay reproducesS176/D288/I0,WER95.6701031%,0/80exact. These protocols are distinct even where aggregate WER happens to match. Saved logits can support cheap beam experiments without another model pass. Logit confidence summaries and publisher timing metadata are saved; boundaries are not qualified physical-sign ground truth.\n\nThe existing128TRAIN monitor is retained. Historical fullTRAIN WER88.4005% is reused from verified selected_train1500.json; do not claim a fresh fullTRAIN baseline evaluation. The192memorization control, from a different checkpoint, still has10.5874%TRAIN vs101.0309%developmentWER. No new accuracy claim follows from pipeline validity.\n\nThe current vocabulary has2287literal labels,1182singleton-supported and1808with<5distinctTRAIN utterances. Two label pools remain separate. Local named clips are not admitted by filename; permissions/annotation/split provenance must be verified. New classes will be append-only and need valid labeled evidence.\n\nCUDA is available on{env['GPU']},VRAM{env['VRAM_bytes']/2**30:.3f}GiB;Torch{env['torch']}. No dependency/global installation was changed. A is a newly authorized warm-start fine-tune, not a repeat of rejected framing/history changes. B remains independently gated on valid new-class data. No production/app/asset/checkpoint/TEST modification is authorized.\n'''
 (RUN/'BASELINE_AND_ERROR_ANALYSIS.md').write_text(report,encoding='utf-8')
 (RUN/'FINETUNING_PLAN.md').write_text('# Fine-tuning plan\n\nA: restore epoch12 model and optimizer moments; fixed LR1e-4 (ten times below parent), unchanged1500TRAIN/80DEV, frozen vocabulary/normalization, batch4/accumulation2, FP32, clip5, at most12epochs/180seconds. One disposable configured optimizer-step smoke precedes fitting. Save every completed epoch and RNG state; resume only the exact config/manifest. Early-stop by predeclared DEV patience/collapse rules. Report fullTRAIN and current streaming replay only for selected checkpoint.\n\nB: append verified new classes only after permissions, target/feature/timing and family guards pass. Preserve all2289old IDs and parameters at initialization. Mix genuine old/new records; no duplicates counted as new evidence. Block rather than invent labels/reviewer decisions.\n\nC: only if A or B passes the frozen promising screen, compare one retention/imbalance variation; no simultaneous GPU runs or sweep. Decoder comparisons use same cached logits, no English language prior. No promotion without independent/streaming acceptance.\n',encoding='utf-8')
 emit(dict(stage='BASELINE_READY',report=str(RUN/'BASELINE_AND_ERROR_ANALYSIS.md'),greedy_WER=dv['WER'],streaming_WER=stream['WER'],train=1500,dev=80))

def monitor_gpu(path,stop):
 with path.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['monotonic_seconds','gpu_memory_used_MiB','utilization_percent'])
  while not stop.is_set():
   try:
    value=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3).stdout.strip()
    w.writerow([time.perf_counter(),*value.split(', ')]);f.flush()
   except Exception as e:w.writerow([time.perf_counter(),'UNAVAILABLE',type(e).__name__]);f.flush()
   stop.wait(.5)

def optimizer(m,state,cfg):
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);opt.load_state_dict(state['optimizer'])
 for g in opt.param_groups:g['lr']=cfg['learning_rate']
 return opt

def checkpoint(path,m,opt,epoch,extra):
 assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(schema=SCHEMA,epoch=epoch,source_epoch=12,model=m.state_dict(),optimizer=opt.state_dict(),vocabulary=read(RUN/'vocabulary.json')['tokens'],config_sha256=sha(RUN/'A/TRAINING_CONFIGURATION.json'),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),**extra),path)
 return dict(path=str(path.relative_to(RUN)),sha256=sha(path),epoch=epoch)

def fit_A(resume):
 assert (RUN/'BASELINE_AND_ERROR_ANALYSIS.md').exists() and read(RUN/'PREPARE_PASS.json')['status']=='PASS'
 if (RUN/'A/RESULTS.json').exists():emit(dict(stage='A_ALREADY_COMPLETED_NO_REPEAT'));return
 cfg=read(RUN/'A/TRAINING_CONFIGURATION.json');assert sha(SOURCE)==cfg['initialization_sha256']
 train,dev,vocab,data,times,_=load_rows();ids=set(read(RUN/'monitor128_ids.json'));mon=[r for r in train if r['sample_id'] in ids]
 seeds(cfg['seed']);m,state=load_model(vocab);opt=optimizer(m,state,cfg);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 initial=read(RUN/'baseline_greedy_DEV80.json');best=(initial['WER'],initial['CTC_loss']);selected=None;stale=0;collapse=0;updates=0;start_epoch=1;elapsed_before=0.;records=[]
 if resume:
  pointer=read(RUN/'A/latest.json');p=RUN/pointer['path'];assert sha(p)==pointer['sha256'];s=torch.load(p,map_location='cpu',weights_only=False)
  assert s['config_sha256']==sha(RUN/'A/TRAINING_CONFIGURATION.json');m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start_epoch=s['epoch']+1
  random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']]);best=tuple(s['best']);selected=s['selected'];stale=s['stale'];collapse=s['collapse'];updates=s['updates'];elapsed_before=s['elapsed_seconds'];records=read(RUN/'A/metrics_history.json')
 else:
  # One configured accumulation group, saved as explicit disposable smoke, then reload source.
  before=m.output.weight.detach().clone();opt.zero_grad(set_to_none=True)
  for rr in [train[:4],train[4:8]]:
   x,y,l,tl=batch(rr,data,vocab,'cuda:0');loss=criterion(m(x).log_softmax(-1).transpose(0,1),y,l,tl);assert torch.isfinite(loss);(loss/2).backward()
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(norm);opt.step();assert not torch.equal(before,m.output.weight)
  ptr=checkpoint(RUN/'A/smoke_one_update.pt',m,opt,0,dict(stage='DISPOSABLE_ONE_UPDATE_SMOKE_NOT_SELECTED'))
  save(RUN/'A/SMOKE_RESULT.json',dict(status='PASS',optimizer_updates=1,loss=float(loss.detach()),gradient_norm=float(norm),checkpoint=ptr,restored_source_before_main_fit=True))
  del m,opt,state,before;torch.cuda.empty_cache();seeds(cfg['seed']);m,state=load_model(vocab);opt=optimizer(m,state,cfg)
  assert all(torch.equal(m.state_dict()[k].cpu(),v) for k,v in state['model'].items())
 torch.cuda.reset_peak_memory_stats();stop=threading.Event();thread=threading.Thread(target=monitor_gpu,args=(RUN/'A/gpu_memory.csv',stop),daemon=True);thread.start()
 started=time.perf_counter();deadline=started+max(0,cfg['wall_seconds']-elapsed_before);reason='MAX_EPOCHS';completed=start_epoch-1;discarded_partial_updates=0
 emit(dict(stage='A_TRAINING_STARTED',warm_start_epoch=12,train=1500,dev=80,lr=cfg['learning_rate'],max_epochs=12,max_wall_seconds=180,resume=resume))
 try:
  for epoch in range(start_epoch,13):
   if time.perf_counter()>=deadline:reason='WALL_LIMIT';break
   tick=time.perf_counter();m.train();order=list(train);random.shuffle(order);micro=[order[i:i+4] for i in range(0,len(order),4)];losses=[];epoch_updates=0;partial=False
   for start in range(0,len(micro),2):
    if time.perf_counter()>=deadline:partial=True;reason='WALL_LIMIT_PARTIAL_EPOCH_DISCARDED';break
    free,_=torch.cuda.mem_get_info();assert free>=cfg['min_free_VRAM_bytes'] and torch.cuda.max_memory_allocated()<=cfg['VRAM_cap_bytes'],'VRAM_STOP'
    group=micro[start:start+2];opt.zero_grad(set_to_none=True)
    for rr in group:
     x,y,l,tl=batch(rr,data,vocab,'cuda:0');z=m(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);assert torch.isfinite(loss),'NONFINITE_CTC';(loss/len(group)).backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(norm),'NONFINITE_GRADIENT';opt.step();epoch_updates+=1
   torch.cuda.synchronize();fit_seconds=time.perf_counter()-tick
   if partial:discarded_partial_updates=epoch_updates;break
   completed=epoch;updates+=epoch_updates;tm=score(m,mon,data,vocab);dv=None;improved=False
   if epoch in cfg['evaluate_epochs']:
    dv=score(m,dev,data,vocab);rank=(dv['WER'],dv['CTC_loss']);improved=rank<best
    if improved:best=rank
    stale=0 if dv['WER']<=initial['WER']-cfg['minimum_credible_gain'] else stale+1
    collapsed=all(v['blank_fraction']>=.995 or (v['top_token_fraction']>=.98 and v['token_diversity']<=3) for v in [tm,dv]);collapse=collapse+1 if collapsed else 0
    save(RUN/f'A/predictions/DEV80_epoch_{epoch:03d}.json',dv)
   row=dict(epoch=epoch,optimizer_updates=epoch_updates,optimization_CTC=float(np.mean(losses)),fit_seconds=fit_seconds,epoch_total_seconds=time.perf_counter()-tick,train_monitor_WER=tm['WER'],train_monitor_CTC=tm['CTC_loss'],train_monitor_blank_fraction=tm['blank_fraction'],train_monitor_diversity=tm['token_diversity'],DEV_WER=(dv['WER'] if dv else None),DEV_CTC=(dv['CTC_loss'] if dv else None),DEV_blank_fraction=(dv['blank_fraction'] if dv else None),DEV_diversity=(dv['token_diversity'] if dv else None),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),observations_per_fit_second=sum(len(data[r['sample_id']]) for r in train)/fit_seconds)
   p=RUN/f'A/checkpoints/epoch_{epoch:03d}.pt'
   if improved:selected=dict(path=str(p.relative_to(RUN)),epoch=epoch)
   ptr=checkpoint(p,m,opt,epoch,dict(stage='FINE_TUNING_A',best=best,selected=selected,stale=stale,collapse=collapse,updates=updates,elapsed_seconds=elapsed_before+time.perf_counter()-started))
   # Mutable progress pointers are confined to this new run; checkpoints are exclusive.
   (RUN/'A/latest.json').write_text(json.dumps(ptr,indent=2)+'\n');records.append(row);(RUN/'A/metrics_history.json').write_text(json.dumps(records,indent=2)+'\n')
   emit(dict(stage='A_EPOCH',**row))
   if epoch>=cfg['collapse_after_epoch'] and collapse>=2:reason='COLLAPSE_STOP';break
   if stale>=cfg['early_stopping_stale_evaluations']:reason='NO_CREDIBLE_DEV_GAIN_STOP';break
 finally:
  stop.set();thread.join(timeout=4)
 wall=elapsed_before+time.perf_counter()-started
 if selected is None:
  # Report best actually trained checkpoint separately; never disguise source as fine-tuned.
  evaluated=[r for r in records if r['DEV_WER'] is not None];chosen=min(evaluated,key=lambda r:(r['DEV_WER'],r['DEV_CTC'])) if evaluated else None
  assert chosen,'NO_COMPLETED_EVALUATED_EPOCH';selected=dict(path=f'A/checkpoints/epoch_{chosen["epoch"]:03d}.pt',epoch=chosen['epoch'])
 p=RUN/selected['path'];s=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(s['model']);selected['sha256']=sha(p)
 train_score=score(m,train,data,vocab);cache={};dev_score=score(m,dev,data,vocab,cache);stream=streaming_score(m,dev,data,times,vocab)
 for name,value in [('TRAIN1500',train_score),('DEV80',dev_score),('streaming_DEV80',stream)]:save(RUN/f'A/selected_{name}.json',value)
 np.savez_compressed(RUN/'A/selected_logits_DEV80.npz',**cache);writecsv(RUN/'A/METRICS.csv',records)
 original_train=read(BASE/'selected_train1500.json')
 promising=(initial['WER']-dev_score['WER']>=.02 and dev_score['insertions']/485<=initial['insertions']/485+.02 and train_score['WER']<=original_train['WER']+.05)
 result=dict(status='COMPLETED_BOUNDED_FINE_TUNING',reason=reason,completed_epochs=completed,optimizer_updates=updates,disposable_smoke_updates=1,discarded_partial_epoch_updates=discarded_partial_updates,wall_training_and_epoch_eval_seconds=wall,fit_seconds=sum(r['fit_seconds'] for r in records),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),selected_checkpoint=selected,baseline_greedy_WER=initial['WER'],TRAIN_WER=train_score['WER'],DEV_WER=dev_score['WER'],streaming_DEV_WER=stream['WER'],DEV_substitutions=dev_score['substitutions'],DEV_deletions=dev_score['deletions'],DEV_insertions=dev_score['insertions'],DEV_exact=dev_score['exact_sequences'],promising_for_C=promising,old_classes=2287,new_classes=0,production_promoted=False,TEST_used=False)
 save(RUN/'A/RESULTS.json',result);emit(result)

def verify():
 before=read(RUN/'protected_hashes.json');assert all(sha(p)==s for p,s in before.items())
 inputs=read(RUN/'input_features.json');assert all(sha(r['path'])==r['sha256'] for r in inputs)
 emit(dict(status='PASS',protected_files=len(before),feature_files=len(inputs),TEST_used=False,app_modified=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','A','verify']);p.add_argument('--resume',action='store_true');a=p.parse_args()
 try:
  if a.stage=='prepare':prepare()
  elif a.stage=='A':fit_A(a.resume)
  else:verify()
 except BaseException as e:
  traceback.print_exc();failure=RUN/f'failure_{a.stage}_{int(time.time())}.json';save(failure,dict(stage=a.stage,error=repr(e),checkpoint_artifacts_preserved=True));raise
