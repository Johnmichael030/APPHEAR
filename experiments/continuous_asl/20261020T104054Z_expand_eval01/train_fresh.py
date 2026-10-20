"""Fresh-weight pilot on a frozen, same-pool publisher-signer-disjoint split."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from collections import defaultdict,Counter
import csv,hashlib,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from core import RUN,ROOT,read,write,sha,SCHEMA
sys.path.insert(0,str(ROOT/'tools/continuous_asl'));sys.path.insert(0,str(ROOT))
from ctc_experiment import CausalCTC,batch,evaluate,seeds,edit_counts
OUT=RUN/'pilot_fresh48'

def save(name,obj):write(OUT/name,obj)
def atomic(name,obj):
 path=OUT/name;tmp=path.with_suffix(path.suffix+'.partial')
 with tmp.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
 os.replace(tmp,path)
def stats(model,rows,data,vocab):
 result=evaluate(model,rows,data,vocab,'cuda:0');blank=0;frames=0
 with torch.inference_mode():
  for row in rows:
   z=model(torch.as_tensor(data[row['sample_id']],device='cuda:0')[None])[0];ids=z.argmax(-1)
   blank+=int((ids==0).sum());frames+=len(ids)
 result.update(blank_argmax_fraction=blank/frames,total_timesteps=frames,
               prediction_tokens=sum(len(p['hypothesis']) for p in result['predictions']),empty_hypotheses=sum(not p['hypothesis'] for p in result['predictions']))
 return result

def main():
 if (OUT/'training_status.json').exists():
  status=read(OUT/'training_status.json');p=status['best_checkpoint']
  if sha(OUT/p['path'])!=p['sha256']:raise ValueError('COMPLETED_CHECKPOINT_HASH')
  print('FRESH_PILOT_COMPLETE_NO_REPEAT');return
 if OUT.exists():raise ValueError('EXISTING_INCOMPLETE_PILOT_NO_SILENT_OVERWRITE')
 plan=read(RUN/'dataset_plan.json');manifest=RUN/'data/processed/train_dev/manifest.json'
 if sha(manifest)!=plan['manifest_sha256'] or plan['new_weight_initialization']!='RANDOM_ONLY' or any(plan['overlap'].values()):raise ValueError('FROZEN_SPLIT_GATE')
 if not torch.cuda.is_available():raise ValueError('CUDA_REQUIRED')
 rows=read(manifest);fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];data={}
 for row in rows:
  path=Path(row['feature_path'])
  if sha(path)!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
  with np.load(path,allow_pickle=False) as f:
   if str(f['schema_version'].item())!=SCHEMA:raise ValueError('FEATURE_SCHEMA')
   x=f['X'].copy();ts=f['timestamp_ms']
   if x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
  if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_LENGTH')
  data[row['sample_id']]=x
 if {r['signer_id'] for r in fit}&{r['signer_id'] for r in dev}:raise ValueError('SIGNER_LEAKAGE')
 OUT.mkdir();(OUT/'checkpoints').mkdir();seeds(20261020)
 config={'schema':SCHEMA,'seed':20261020,'channels':48,'batch':4,'gradient_accumulation':2,'lr':.001,'weight_decay':.0001,'gradient_clip':5.,
  'mixed_precision':False,'max_epochs':12,'wall_seconds':180,'minimum_learning_window_epochs':5,'early_stopping_patience':3,
  'initialization':'FRESH_RANDOM_NO_ASL_RECOGNITION_CHECKPOINT_READ','normalization':'NewTRAIN only','vocab':'NewTRAIN only; literal unchanged occurrence labels, blank0/UNK1',
  'manifest_sha256':sha(manifest),'training_script_sha256':sha(__file__),'core_model_decoder_sha256':sha(ROOT/'tools/continuous_asl/ctc_experiment.py'),
  'max_allocated_vram_bytes':2684354560,'minimum_free_vram_bytes':268435456,
  'stop_protocol':'At epoch5 stop if TRAIN monitor WER>=.98 or all monitor predictions empty. Nonfinite loss/gradient or resource failure stops immediately. All-blank TRAIN and DEV with non-improving monitor loss for2epochs stops earlier. After5 continue only if TRAIN monitorWER<=.95 and DEVWER<1.0 with>=5%correct reference tokens; otherwise stop. DEVpatience3 thereafter.',
  'DEV_status':'One publisher held-out signer,413utterances,target convention matched. Historical TRAIN exposure means researchDEV only, not pristineTEST.',
  'production_enabled':False}
 save('config.json',config)
 vocabulary=['<BLANK>','<UNK>']+sorted({t for r in fit for t in r['glosses']});save('vocabulary.json',{'tokens':vocabulary,'fitted_split':'train','schema':SCHEMA})
 values=np.concatenate([data[r['sample_id']] for r in fit]).astype(np.float64)
 mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32);del values
 model=CausalCTC(len(vocabulary),48,mean,std).to('cuda:0');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 groups=defaultdict(list)
 for row in fit:groups[row['collection']].append(row)
 for items in groups.values():items.sort(key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
 monitor=[]
 while len(monitor)<64:
  for key in sorted(groups):
   if groups[key] and len(monitor)<64:monitor.append(groups[key].pop(0))
 save('train_monitor_manifest.json',monitor)
 # Finite forward/backward smoke on four actual TRAIN examples, no optimizer update.
 x,y,l,tl=batch(fit[:4],data,vocabulary,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);loss.backward()
 finite=bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
 if not finite:raise ValueError('CUDA_SMOKE_FINITE_GATE')
 optimizer.zero_grad(set_to_none=True)
 save('cuda_smoke.json',{'status':'PASS','loss':float(loss.detach()),'input_device':str(x.device),'logits_device':str(z.device),'gradient_device':str(next(model.parameters()).grad.device),'optimizer_updates':0,'real_TRAIN_examples':4})
 initial_dev=stats(model,dev,data,vocabulary);initial_monitor=stats(model,monitor,data,vocabulary);save('untrained_dev.json',initial_dev);save('initial_train_monitor.json',initial_monitor)
 save('blank_baseline.json',{'dev_wer':1.,'deletions':sum(len(r['glosses']) for r in dev),'substitutions':0,'insertions':0,'scope':'All-empty literal-token baseline on this new DEV; old9DEV baseline is not comparable.'})
 del x,y,z,loss;torch.cuda.reset_peak_memory_stats();started=time.perf_counter();best=(float('inf'),float('inf'));patience=0;completed=0;status='MAX_EPOCHS';best_pointer=None;previous_monitor=initial_monitor;flat_blank=0
 print(json.dumps({'status':'FRESH_CUDA_PILOT_STARTED','gpu':torch.cuda.get_device_name(0),'train':len(fit),'dev':len(dev),'vocabulary_size':len(vocabulary),'initial_DEV_wer':initial_dev['gloss_wer'],'DEV_OOV':plan['dev_oov_tokens'],'no_recognition_checkpoint_loaded':True}),flush=True)
 try:
  for epoch in range(1,13):
   if time.perf_counter()-started>=180:status='WALL_BUDGET';break
   if shutil.disk_usage(ROOT).free<10737418240:raise RuntimeError('DISK_HEADROOM')
   tick=time.perf_counter();model.train();order=list(fit);random.shuffle(order);optimizer.zero_grad(set_to_none=True);losses=[];batches=list(range(0,len(order),4))
   for j,offset in enumerate(batches):
    if torch.cuda.mem_get_info()[0]<268435456:raise RuntimeError('VRAM_HEADROOM')
    x,y,l,tl=batch(order[offset:offset+4],data,vocabulary,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_CTC_LOSS')
    (loss/2).backward()
    if torch.cuda.max_memory_allocated()>2684354560:raise RuntimeError('VRAM_CAP')
    if (j+1)%2==0 or j+1==len(batches):
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
     if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRADIENT')
     optimizer.step();optimizer.zero_grad(set_to_none=True)
    losses.append(float(loss.detach()))
   score=stats(model,dev,data,vocabulary);training=stats(model,monitor,data,vocabulary);torch.cuda.synchronize()
   rank=(score['gloss_wer'],score['loss']);improved=rank<best
   if improved:best=rank;patience=0
   else:patience+=1
   cp=OUT/'checkpoints'/f'epoch_{epoch:03d}.pt'
   torch.save({'schema':SCHEMA,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'vocabulary':vocabulary,'config':config,
    'manifest_sha256':config['manifest_sha256'],'best_score':best,'patience':patience,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},cp)
   pointer={'path':str(cp.relative_to(OUT)),'sha256':sha(cp),'epoch':epoch};atomic('checkpoints/latest.json',pointer)
   save(f'dev_epoch_{epoch:03d}.json',score);save(f'train_monitor_epoch_{epoch:03d}.json',training)
   if improved:best_pointer=pointer;atomic('checkpoints/best.json',pointer)
   record={'epoch':epoch,'train_optimization_loss':float(np.mean(losses)),'train_monitor_loss':training['loss'],'train_monitor_wer':training['gloss_wer'],
    'train_monitor_blank_fraction':training['blank_argmax_fraction'],'train_monitor_prediction_tokens':training['prediction_tokens'],'dev_loss':score['loss'],'dev_wer':score['gloss_wer'],
    'dev_blank_fraction':score['blank_argmax_fraction'],'dev_prediction_tokens':score['prediction_tokens'],'dev_empty_hypotheses':score['empty_hypotheses'],
    'S':score['substitutions'],'D':score['deletions'],'I':score['insertions'],'reference_tokens':score['reference_tokens'],'epoch_seconds':time.perf_counter()-tick,
    'elapsed_seconds':time.perf_counter()-started,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
    'input_device':str(x.device),'logits_device':str(z.device),'loss_device':str(loss.device),'best':improved}
   with (OUT/'training_log.csv').open('a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(record))
    if f.tell()==0:w.writeheader()
    w.writerow(record)
   completed=epoch;print(json.dumps(record),flush=True)
   both_blank=training['prediction_tokens']==0 and score['prediction_tokens']==0
   flat_blank=flat_blank+1 if both_blank and training['loss']>=previous_monitor['loss']*.99 else 0
   previous_monitor=training
   if flat_blank>=2:status='BLANK_COLLAPSE_WITHOUT_LOSS_PROGRESS';break
   if epoch==5:
    correct=score['reference_tokens']-score['substitutions']-score['deletions']
    if training['gloss_wer']>=.98 or training['prediction_tokens']==0:status='TRAIN_BLANK_OR_NO_LEARNABILITY_GATE';break
    if not (training['gloss_wer']<=.95 and score['gloss_wer']<1. and correct>=.05*score['reference_tokens']):status='NO_CREDIBLE_DEV_GAIN_INITIAL_WINDOW';break
   if epoch>5 and patience>=3:status='DEV_EARLY_STOP';break
  if not completed:raise RuntimeError('NO_COMPLETED_EPOCH')
  loop_seconds=time.perf_counter()-started
  save('last_checkpoint_full_train.json',evaluate(model,fit,data,vocabulary,'cuda:0'))
  loaded=torch.load(OUT/best_pointer['path'],map_location='cuda:0',weights_only=False);model.load_state_dict(loaded['model'])
  best_dev=stats(model,dev,data,vocabulary);best_train=evaluate(model,fit,data,vocabulary,'cuda:0')
  if best_dev['predictions']!=read(OUT/f'dev_epoch_{best_pointer["epoch"]:03d}.json')['predictions']:raise ValueError('CHECKPOINT_RELOAD_PARITY')
  save('best_checkpoint_dev.json',best_dev);save('best_checkpoint_full_train.json',best_train)
  summary={'status':status,'used_cuda':True,'gpu':torch.cuda.get_device_name(0),'completed_epochs':completed,'best_checkpoint':best_pointer,'checkpoint_reload_predictions_match':True,
    'train':len(fit),'dev':len(dev),'unique_train_consumed':len(fit),'utterance_presentations':len(fit)*completed,'best_train_wer':best_train['gloss_wer'],'best_dev_wer':best_dev['gloss_wer'],
    'best_dev_prediction_tokens':best_dev['prediction_tokens'],'best_dev_blank_fraction':best_dev['blank_argmax_fraction'],'epoch_loop_seconds':loop_seconds,
    'fit_reload_full_train_diagnostics_seconds':time.perf_counter()-started,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
    'manifest_sha256':config['manifest_sha256'],'script_sha256':sha(__file__),'normalization_fit':'NewTRAIN only','vocab_fit':'NewTRAIN only','random_initialization':True,
    'old_recognition_weights_read':False,'test_accessed':False,'production_promoted':False,'comparison':'Newsame-pool signer-disjoint DEV; no numerical comparison to frozen old9DEV85.96%.'}
  save('training_status.json',summary);print(json.dumps(summary),flush=True)
 except Exception as error:
  save('failure.json',{'error':str(error),'completed_epochs':completed,'resume_from_last_full_epoch':True});raise

if __name__=='__main__':main()
