"""New bounded continuation, fixed frozen data/model/optimizer/decoder contracts."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from collections import defaultdict,Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
PARENT=ROOT/'experiments/continuous_asl/20261020_continuous_expand08'
DIAG=ROOT/'experiments/continuous_asl/20261020T051631Z_devwer_diagnosis01'
sys.path.insert(0,str(ROOT/'tools/continuous_asl'));sys.path.insert(0,str(ROOT))
from ctc_experiment import CausalCTC,batch,evaluate,seeds
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def write(name,data):
 path=(RUN/name).resolve()
 if RUN not in path.parents:raise ValueError('NEW_PILOT_OUTPUT_SCOPE')
 with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.write('\n')
def atomic(name,data):
 path=RUN/name;temp=path.with_suffix(path.suffix+'.partial')
 with temp.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.write('\n')
 os.replace(temp,path)

def main():
 if (RUN/'training_status.json').exists():
  saved=read(RUN/'training_status.json')
  if sha(RUN/saved['best_checkpoint']['path'])!=saved['best_checkpoint']['sha256']:raise ValueError('COMPLETED_CHECKPOINT_HASH')
  print('VERIFIED_COMPLETE_NO_REFIT',flush=True);return
 if (RUN/'config.json').exists():raise ValueError('INTERRUPTED_RUN_REQUIRES_REVIEW_NO_SILENT_RESTART')
 if not torch.cuda.is_available():raise ValueError('CUDA_REQUIRED')
 if not all(r['passed'] for r in read(DIAG/'checks.json')) or read(DIAG/'supplement_results.json')['status']!='PASS' or read(DIAG/'source_contract_checks.json')['status']!='PASS':raise ValueError('CRITICAL_DIAGNOSTIC_GATE')
 seeds(20261020)
 manifest=PARENT/'data/processed/train_dev/manifest.json';frozen=read(PARENT/'model_selection_freeze.json')
 if sha(manifest)!=frozen['manifest_sha256']:raise ValueError('FROZEN_MANIFEST_HASH')
 pointer=read(PARENT/'models/reference48/checkpoints/latest.json');path=PARENT/'models/reference48'/pointer['path']
 if sha(path)!=pointer['sha256'] or pointer['epoch']!=17:raise ValueError('PARENT_LATEST_CHECKPOINT')
 state=torch.load(path,map_location='cuda:0',weights_only=False);vocab=state['vocabulary'];oldconfig=state['config']
 if state['manifest_sha256']!=frozen['manifest_sha256'] or sha(ROOT/'tools/continuous_asl/ctc_experiment.py')!=read(PARENT/'code_provenance.json')['model_decoder_sha256']:raise ValueError('FROZEN_TRAINING_CONTRACT')
 rows=read(manifest);fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
 if len(fit)!=2119 or len(dev)!=9 or vocab!=['<BLANK>','<UNK>']+sorted({t for r in fit for t in r['glosses']}):raise ValueError('FROZEN_SPLIT_VOCAB')
 if {r['collection'] for r in fit}&{r['collection'] for r in dev}:raise ValueError('COLLECTION_LEAKAGE')
 data={}
 for row in rows:
  if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
  with np.load(row['feature_path'],allow_pickle=False) as f:
   if str(f['schema_version'].item())!=SCHEMA:raise ValueError('FEATURE_SCHEMA')
   x=f['X'].copy()
  if x.dtype!=np.float32 or x.shape[1]!=378 or not np.isfinite(x).all():raise ValueError('FEATURE_VALUES')
  if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_LENGTH')
  data[row['sample_id']]=x
 # Same data/labels/vocab/normalizer/architecture/optimizer. Only new continuation budget and stop rules.
 config={'schema':SCHEMA,'parent_checkpoint':str(path),'parent_checkpoint_sha256':pointer['sha256'],'parent_epoch':17,
  'manifest_path':str(manifest),'manifest_sha256':sha(manifest),'parent_config':oldconfig,'script_sha256':sha(__file__),
  'change':'Continue epoch17 weights and AdamW/RNG state; fixed data/vocab/normalization/decoder. New bounded budget, monitor and stop rules only.',
  'initial_window_epochs':3,'max_additional_epochs':8,'wall_seconds':120,'continuation_gate':'After3additional epochs, continue only if best DEV improves at least2/57tokens versus parent and TRAIN monitor loss<=initial.',
  'collapse_gate':'Stop if TRAIN monitor WER>=.98 and>=90%empty hypotheses. Nonfinite loss/gradients stop immediately.',
  'no_learning_gate':'After2additional epochs stop if monitor loss has not fallen1% and monitor WER has not improved.',
  'evaluation_scope':'Original9DEV cross-corpus research diagnostic; signer independence not established.17reserved TEST untouched.',
  'production_enabled':False,'seed':20261020}
 write('config.json',config);(RUN/'checkpoints').mkdir();write('vocabulary.json',{'tokens':vocab,'fitted_split':'train','schema':SCHEMA})
 groups=defaultdict(list)
 for row in fit:groups[row['collection']].append(row)
 for group in groups.values():group.sort(key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
 monitor=[]
 while len(monitor)<64:
  for name in sorted(groups):
   if groups[name] and len(monitor)<64:monitor.append(groups[name].pop(0))
 write('train_monitor_manifest.json',monitor)
 model=CausalCTC(len(vocab),48,state['model']['mean'].cpu().numpy(),state['model']['std'].cpu().numpy()).to('cuda:0');model.load_state_dict(state['model'])
 optimizer=torch.optim.AdamW(model.parameters(),lr=oldconfig['lr'],weight_decay=oldconfig['weight_decay']);optimizer.load_state_dict(state['optimizer'])
 criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 initial_dev=evaluate(model,dev,data,vocab,'cuda:0');initial_monitor=evaluate(model,monitor,data,vocab,'cuda:0')
 write('initial_dev.json',initial_dev);write('initial_train_monitor.json',initial_monitor)
 if initial_dev['predictions']!=read(DIAG/'latest_epoch_dev_replay.json')['scores']['predictions']:raise ValueError('PARENT_REPLAY_PARITY')
 random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
 del state
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter();best=(float('inf'),float('inf'));status='MAX_ADDITIONAL_EPOCHS';completed=0;best_pointer=None;records=[]
 print(json.dumps({'status':'ACTUAL_CUDA_PILOT_STARTED','gpu':torch.cuda.get_device_name(0),'train':len(fit),'dev':len(dev),'parent_epoch':17,'initial_train_monitor_wer':initial_monitor['gloss_wer'],'initial_dev_wer':initial_dev['gloss_wer']}),flush=True)
 try:
  for local in range(1,9):
   if time.perf_counter()-started>=120:status='WALL_BUDGET';break
   if shutil.disk_usage(RUN).free<10737418240:raise RuntimeError('DISK_HEADROOM')
   tick=time.perf_counter();model.train();order=list(fit);random.shuffle(order);losses=[];optimizer.zero_grad(set_to_none=True)
   batches=list(range(0,len(order),oldconfig['batch']))
   for j,offset in enumerate(batches):
    if torch.cuda.mem_get_info()[0]<oldconfig['minimum_free_vram_bytes']:raise RuntimeError('FREE_VRAM_GUARD')
    x,y,lengths,targets=batch(order[offset:offset+oldconfig['batch']],data,vocab,'cuda:0');logits=model(x)
    loss=criterion(logits.log_softmax(-1).transpose(0,1),y,lengths,targets)
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
    (loss/oldconfig['gradient_accumulation']).backward()
    if torch.cuda.max_memory_allocated()>oldconfig['max_vram_allocated_bytes']:raise RuntimeError('VRAM_GUARD')
    if (j+1)%oldconfig['gradient_accumulation']==0 or j+1==len(batches):
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),oldconfig['gradient_clip'])
     if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRADIENT')
     optimizer.step();optimizer.zero_grad(set_to_none=True)
    losses.append(float(loss.detach()))
   value=evaluate(model,dev,data,vocab,'cuda:0');train_monitor=evaluate(model,monitor,data,vocab,'cuda:0');torch.cuda.synchronize()
   epoch=17+local;score=(value['gloss_wer'],value['loss']);improved=score<best
   if improved:best=score
   cp=RUN/'checkpoints'/f'epoch_{epoch:03d}.pt'
   torch.save({'schema':SCHEMA,'epoch':epoch,'additional_epoch':local,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
    'config':config,'vocabulary':vocab,'manifest_sha256':config['manifest_sha256'],'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'best_score':best},cp)
   cp_pointer={'path':str(cp.relative_to(RUN)),'sha256':sha(cp),'epoch':epoch};atomic('checkpoints/latest.json',cp_pointer)
   write(f'dev_epoch_{epoch:03d}.json',value);write(f'train_monitor_epoch_{epoch:03d}.json',train_monitor)
   if improved:best_pointer=cp_pointer;atomic('checkpoints/best.json',cp_pointer)
   record={'additional_epoch':local,'epoch':epoch,'train_optimization_loss':float(np.mean(losses)),'train_monitor_loss':train_monitor['loss'],'train_monitor_wer':train_monitor['gloss_wer'],
    'dev_loss':value['loss'],'dev_wer':value['gloss_wer'],'substitutions':value['substitutions'],'deletions':value['deletions'],'insertions':value['insertions'],
    'epoch_seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-started,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),
    'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),'input_device':str(x.device),'logits_device':str(logits.device),'loss_device':str(loss.device),'best_in_new_pilot':improved}
   with (RUN/'training_log.csv').open('a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(record))
    if f.tell()==0:w.writeheader()
    w.writerow(record)
   completed=local;records.append(record);print(json.dumps(record),flush=True)
   empty=sum(not p['hypothesis'] for p in train_monitor['predictions'])/len(monitor)
   if train_monitor['gloss_wer']>=.98 and empty>=.9:status='TRAIN_PREDICTION_COLLAPSE';break
   if local>=2 and train_monitor['loss']>=initial_monitor['loss']*.99 and train_monitor['gloss_wer']>=initial_monitor['gloss_wer']:status='NO_TRAIN_LEARNING';break
   if local==3 and not (best[0]<=initial_dev['gloss_wer']-2/57+1e-12 and train_monitor['loss']<=initial_monitor['loss']):status='NO_CREDIBLE_DEV_GAIN_INITIAL_WINDOW';break
  if not completed:raise RuntimeError('NO_COMPLETED_EPOCH')
  loop_seconds=time.perf_counter()-started
  # Save final full TRAIN diagnostics before restoring the best new DEV checkpoint.
  last_train=evaluate(model,fit,data,vocab,'cuda:0');write('last_checkpoint_full_train.json',last_train)
  loaded=torch.load(RUN/best_pointer['path'],map_location='cuda:0',weights_only=False);model.load_state_dict(loaded['model'])
  best_train=evaluate(model,fit,data,vocab,'cuda:0');best_dev=evaluate(model,dev,data,vocab,'cuda:0')
  if best_dev['predictions']!=read(RUN/f'dev_epoch_{best_pointer["epoch"]:03d}.json')['predictions']:raise ValueError('CHECKPOINT_RELOAD_PARITY')
  write('best_checkpoint_full_train.json',best_train);write('best_checkpoint_dev.json',best_dev)
  summary={'status':status,'used_cuda':True,'gpu':torch.cuda.get_device_name(0),'completed_additional_epochs':completed,'parent_epoch':17,'last_epoch':17+completed,
   'best_checkpoint':best_pointer,'best_new_dev_wer':best_dev['gloss_wer'],'best_new_train_wer':best_train['gloss_wer'],'last_train_wer':last_train['gloss_wer'],
   'initial_dev_wer':initial_dev['gloss_wer'],'initial_train_full_wer_from_frozen_evidence':.7195324586031054,
   'epoch_loop_seconds':loop_seconds,'fit_reload_full_train_diagnostics_seconds':time.perf_counter()-started,
   'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
   'checkpoint_reload_predictions_match':True,'manifest_sha256':config['manifest_sha256'],'unique_train_consumed':2119,'utterance_presentations':2119*completed,
   'training_script_sha256':sha(__file__),'production_promoted':False,'test_accessed':False,'frozen_baseline_dev_wer':.8596491228070176,
   'beats_parent_last_dev':best_dev['gloss_wer']<initial_dev['gloss_wer'],'beats_frozen_baseline':best_dev['gloss_wer']<.8596491228070176}
  write('training_status.json',summary);print(json.dumps(summary),flush=True)
 except Exception as e:
  write('failure.json',{'error':str(e),'completed_additional_epochs':completed,'latest_completed_checkpoint':read(RUN/'checkpoints/latest.json') if (RUN/'checkpoints/latest.json').exists() else None})
  raise

if __name__=='__main__':main()
