"""Exactly one bounded fresh research fit on a copied, verified frozen design."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
from collections import defaultdict
import math,random,shutil,sys,time
import numpy as np
import torch
sys.path.insert(0,str(ROOT/'tools/continuous_asl'));sys.path.insert(0,str(ROOT))
from ctc_experiment import CausalCTC,batch,decode,edit_counts,seeds

def main():
 if (RUN/'config.json').exists():raise ValueError('ONE_PILOT_ONLY_NO_REPEAT_OR_SILENT_OVERWRITE')
 source=DESIGN/'folds/A_guarded_pooled_family_holdout.json';expected='87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f'
 assert sha(source)==expected;shutil.copyfile(source,RUN/'fold_manifest.json');assert sha(RUN/'fold_manifest.json')==expected
 fold=read(source);universe=Path(fold['universe_manifest']);assert sha(universe)==fold['universe_manifest_sha256']
 lookup={r['sample_id']:r for r in read(universe) if r['split']=='train'}
 fit=[lookup[i] for i in fold['train_sample_ids']];held=[lookup[i] for i in fold['internal_eval_sample_ids']]
 compatible=[lookup[i] for i in fold['compatible_eval_sample_ids']];strong=[lookup[i] for i in fold['support5_eval_sample_ids']]
 assert (len(fit),len(held),len(compatible),len(strong))==(1500,206,80,27)
 support=Counter(t for r in fit for t in set(r['glosses']));vocab=['<BLANK>','<UNK>']+sorted(support);mapping={t:i for i,t in enumerate(vocab)}
 assert len(vocab)==len(mapping)==2289 and not ({'<BLANK>','<UNK>'}&set(support))
 assert {r['sample_id'] for r in compatible}=={r['sample_id'] for r in held if set(r['glosses'])<=set(support)}
 assert {r['sample_id'] for r in strong}=={r['sample_id'] for r in held if all(support[t]>=5 for t in r['glosses'])}
 overlap={k:sorted({r.get(k) for r in fit if r.get(k)}&{r.get(k) for r in held if r.get(k)}) for k in ['sample_id','source_group','collection','raw_sha256','decoded_pixel_sha256']}
 assert not any(overlap.values());assert not ({family(r) for r in fit}&{family(r) for r in held})
 assert {r['signer_id'] for r in held}<={r['signer_id'] for r in fit}
 assert {r['training_pool'] for r in fit+held}=={'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE'}
 save('resolved_manifest.json',[{**r,'internal_role':'TRAIN' if r['sample_id'] in set(fold['train_sample_ids']) else 'HELD_OUT'} for r in fit+held])
 save('evaluation_views.json',{'primary_compatible80':fold['compatible_eval_sample_ids'],'support5_subgroup27':fold['support5_eval_sample_ids'],'full_internal206_report_only':fold['internal_eval_sample_ids']})
 save('vocabulary.json',{'tokens':vocab,'blank_id':0,'unknown_id':1,'fitted_split':'frozen1500internalTRAIN','literal_labels_unchanged':True})
 data={}
 for r in fit+held:
  assert sha(r['feature_path'])==r['feature_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as f:
   assert str(f['schema_version'].item())==SCHEMA;x=f['X'].copy();ts=f['timestamp_ms']
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
  minimum=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]));assert len(x)>=minimum
  data[r['sample_id']]=x
 values=np.concatenate([data[r['sample_id']] for r in fit]).astype(np.float64);mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32);del values
 np.savez(RUN/'normalization.npz',mean=mean,std=std,schema_version=SCHEMA)
 groups=defaultdict(list)
 for r in fit:groups[r['source_group']].append(r)
 for rr in groups.values():rr.sort(key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
 monitor=[]
 while len(monitor)<128:
  for k in sorted(groups):
   if groups[k] and len(monitor)<128:monitor.append(groups[k].pop(0))
 save('train_monitor_manifest.json',monitor)
 config={'schema':SCHEMA,'seed':20261020,'channels':48,'classes':len(vocab),'batch':4,'gradient_accumulation':2,'lr':.001,'weight_decay':.0001,'gradient_clip':5.,'mixed_precision':False,
  'max_epochs':20,'wall_budget_seconds':180,'conservative_stop_request_seconds':175,'evaluation_epochs':[1,2,4,6,8,10,12,14,16,18,20],
  'init':'FRESH_RANDOM; no previous recognition weights read','normalization':'1500internalTRAIN only','vocabulary':'1500internalTRAIN literal labels; blank0/UNK1; all80/27targets mapped exactly',
  'checkpoint_selection':'Primary80 tuple(WER,CTCloss), all80retained; 27subgroup/full206not used for selection',
  'collapse_gate':'From epoch6: two consecutive evaluations with TRAINmonitor and primary80 both>=99.5%blank, or both>=98%one-token domination with<=3distinct predicted labels.',
  'no_credible_gain_gate':'From epoch6 stop after4evaluation events without>=0.5percentage-point primary80WER improvement relative to all-empty100%baseline; ties do not reset. Max20epochs/175sstop request remains.',
  'success_screen':'Full selectedTRAIN and fixedmonitor WER<=50%,TRAINdiversity>=10,multi>=25%; primary80WER<=90%,diversity>=10,multi>=25%. This is an engineering screen, not unseen-signer accuracy.',
  'VRAM_cap_allocated_bytes':2684354560,'minimum_free_VRAM_bytes':268435456,'disk_headroom_bytes':10737418240,
  'fold_manifest_sha256':expected,'resolved_manifest_sha256':sha(RUN/'resolved_manifest.json'),'vocabulary_sha256':sha(RUN/'vocabulary.json'),'normalization_sha256':sha(RUN/'normalization.npz'),
  'core_model_decoder_sha256':sha(ROOT/'tools/continuous_asl/ctc_experiment.py'),'training_script_sha256':sha(__file__),'common_script_sha256':sha(RUN/'common.py'),
  'TEST_used':False,'production_enabled':False,'user_authorization':'Explicit one bounded source-transfer pilot authorization in this conversation; supersedes design-only no-fit flag without modifying frozen design.'}
 save('config.json',config)
 assert torch.cuda.is_available();seeds(config['seed']);device='cuda:0'
 model=CausalCTC(len(vocab),48,mean,std).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 (RUN/'checkpoints').mkdir()
 def checkpoint(name,epoch,extra=None):
  path=RUN/'checkpoints'/name;assert not path.exists()
  state={'schema':SCHEMA,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'vocabulary':vocab,'configuration':config,'config_sha256':sha(RUN/'config.json'),
   'fold_manifest_sha256':expected,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
  if extra:state.update(extra)
  torch.save(state,path);return {'path':str(path.relative_to(RUN)),'sha256':sha(path),'epoch':epoch}
 initial=checkpoint('initialization.pt',0);save('initialization.json',{**initial,'optimizer_updates':0,'recognition_weights_reused':False})
 save('preflight.json',{'status':'PASS','source_manifest':str(source),'source_manifest_sha256':expected,'counts':{'TRAIN':1500,'primary_eval':80,'strong_subgroup':27,'full_internal_eval':206},
  'overlap':overlap,'capture_family_overlap':False,'train_capture_families':sorted({family(r) for r in fit}),'held_capture_families':sorted({family(r) for r in held}),
  'all80and27labels_in_vocabulary':True,'feature_files_hash_schema_timestamp_CTC_checked':len(data),'GPU':torch.cuda.get_device_name(0),'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory,
  'cuda_version':torch.version.cuda,'torch_version':torch.__version__,'TEST_used':False})
 # Finite CUDA backward smoke, no optimizer step or parameter mutation.
 x,y,l,tl=batch(fit[:4],data,vocab,device);z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);loss.backward()
 assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
 save('cuda_smoke.json',{'status':'PASS','input_device':str(x.device),'logits_device':str(z.device),'loss_device':str(loss.device),'gradient_device':str(next(model.parameters()).grad.device),'optimizer_updates':0,'loss':float(loss.detach())})
 optimizer.zero_grad(set_to_none=True);del x,y,z,loss
 def score(samples):
  model.eval();predictions=[]
  with torch.inference_mode():
   for r in samples:
    x=torch.as_tensor(data[r['sample_id']],device=device)[None];logits=model(x)[0];ids=logits.argmax(-1);hyp=decode(logits,vocab);ctc=None
    if all(t in mapping for t in r['glosses']):
     target=torch.tensor([mapping[t] for t in r['glosses']],device=device,dtype=torch.long)
     ctc=float(criterion(logits.log_softmax(-1)[:,None],target,torch.tensor([len(logits)]),torch.tensor([len(target)])))
     assert math.isfinite(ctc)
    predictions.append({'sample_id':r['sample_id'],'reference':r['glosses'],'hypothesis':hyp,**edit_counts(r['glosses'],hyp),'CTC_loss':ctc,
     'blank_frames':int((ids==0).sum()),'output_frames':len(ids),'signer_id':r['signer_id'],'source_group':r['source_group'],'capture_family':family(r)})
  return summarize(predictions)
 primaryids={r['sample_id'] for r in compatible};strongids={r['sample_id'] for r in strong}
 def views(full):
  return (summarize([p for p in full['predictions'] if p['sample_id'] in primaryids]),summarize([p for p in full['predictions'] if p['sample_id'] in strongids]))
 untrained=score(held);save('untrained_full206.json',untrained);untrained80,untrained27=views(untrained);save('untrained_primary80.json',untrained80);save('untrained_strong27.json',untrained27)
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter();completed=0;updates=0;presentations=0;best=(float('inf'),float('inf'));best_pointer=None;credible=1.;stale=0;collapsed=0;reason='MAX_EPOCHS';metrics=[]
 print(json.dumps({'status':'SOURCE_TRANSFER_CUDA_STARTED','GPU':torch.cuda.get_device_name(0),'TRAIN':1500,'primary_eval':80,'strong_subgroup':27,'classes':len(vocab),'initialization_sha256':initial['sha256'],'config_sha256':sha(RUN/'config.json')}),flush=True)
 try:
  for epoch in range(1,21):
   if time.perf_counter()-started>=175:reason='WALL_BUDGET_STOP';break
   tick=time.perf_counter();model.train();order=list(fit);random.shuffle(order);optimizer.zero_grad(set_to_none=True);losses=[];batchcount=0;partial=False;offset=0
   for offset in range(0,len(order),4):
    if time.perf_counter()-started>=175:partial=True;reason='WALL_BUDGET_STOP';break
    free,_=torch.cuda.mem_get_info()
    if free<268435456 or shutil.disk_usage(ROOT).free<10737418240:raise RuntimeError('RESOURCE_HEADROOM_STOP')
    x,y,l,tl=batch(order[offset:offset+4],data,vocab,device);z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_CTC_LOSS')
    (loss/2).backward();batchcount+=1;presentations+=len(order[offset:offset+4]);losses.append(float(loss.detach()))
    if batchcount%2==0 or offset+4>=len(order):
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
     if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRADIENT')
     optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
    if torch.cuda.max_memory_allocated()>2684354560:raise RuntimeError('VRAM_CAP_STOP')
   fitseconds=time.perf_counter()-tick
   record={'epoch':epoch,'epoch_complete':not partial,'optimization_CTC_loss':float(np.mean(losses)) if losses else None,'fit_epoch_seconds':fitseconds,'elapsed_seconds':time.perf_counter()-started,
    'primary80_WER':None,'strong27_WER':None,'full206_WER':None,'monitor_WER':None,'monitor_CTC':None,'primary80_CTC':None,'strong27_CTC':None,
    'monitor_blank':None,'primary80_blank':None,'strong27_blank':None,'monitor_diversity':None,'primary80_diversity':None,'strong27_diversity':None,'primary80_top_fraction':None,'primary80_multi_outputs':None,
    'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_VRAM_bytes':torch.cuda.max_memory_reserved(),'optimizer_updates':updates,'checkpoint_sha256':''}
   if partial:
    pointer=checkpoint(f'partial_epoch_{epoch:03d}.pt',completed,{'partial_epoch':epoch,'next_batch_offset':offset,'shuffle_order_ids':[r['sample_id'] for r in order],
     'pending_gradient_microbatches':batchcount%2,'pending_gradients':{n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None},'optimizer_updates':updates})
    save('partial_checkpoint.json',pointer)
   else:
    completed=epoch
    if epoch in config['evaluation_epochs']:
     full=score(held);primary,strongscore=views(full);training=score(monitor);torch.cuda.synchronize()
     for name,result in [('monitor',training),('primary80',primary),('strong27',strongscore),('full206',full)]:save(f'predictions/{name}_epoch_{epoch:03d}.json',result)
     rank=(primary['gloss_WER'],primary['CTC_loss']);improved=rank<best
     if improved:best=rank
     pointer=checkpoint(f'epoch_{epoch:03d}.pt',epoch,{'best_score':best,'optimizer_updates':updates})
     if improved:best_pointer=pointer
     if primary['gloss_WER']<=credible-.005+1e-12:credible=primary['gloss_WER'];stale=0
     else:stale+=1
     blankcollapse=training['blank_argmax_fraction']>=.995 and primary['blank_argmax_fraction']>=.995
     frequentcollapse=training['top_predicted_token_fraction']>=.98 and primary['top_predicted_token_fraction']>=.98 and training['unique_predicted_tokens']<=3 and primary['unique_predicted_tokens']<=3
     collapsed=collapsed+1 if blankcollapse or frequentcollapse else 0
     if epoch>=6 and collapsed>=2:reason='PERSISTENT_BLANK_OR_FREQUENT_TOKEN_COLLAPSE'
     elif epoch>=6 and stale>=4:reason='NO_CREDIBLE_PRIMARY_GAIN'
     record.update(primary80_WER=primary['gloss_WER'],strong27_WER=strongscore['gloss_WER'],full206_WER=full['gloss_WER'],monitor_WER=training['gloss_WER'],monitor_CTC=training['CTC_loss'],primary80_CTC=primary['CTC_loss'],strong27_CTC=strongscore['CTC_loss'],
      monitor_blank=training['blank_argmax_fraction'],primary80_blank=primary['blank_argmax_fraction'],strong27_blank=strongscore['blank_argmax_fraction'],monitor_diversity=training['unique_predicted_tokens'],primary80_diversity=primary['unique_predicted_tokens'],strong27_diversity=strongscore['unique_predicted_tokens'],
      primary80_top_fraction=primary['top_predicted_token_fraction'],primary80_multi_outputs=primary['multi_token_hypotheses'],checkpoint_sha256=pointer['sha256'])
     print(json.dumps(record),flush=True)
   record.update(elapsed_seconds=time.perf_counter()-started,epoch_total_seconds=time.perf_counter()-tick)
   metrics.append(record)
   with (RUN/'METRICS.csv').open('a',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(record))
    if f.tell()==0:w.writeheader()
    w.writerow(record)
   if reason!='MAX_EPOCHS':break
  loopseconds=time.perf_counter()-started
  if best_pointer is None:raise RuntimeError('NO_SELECTED_COMPLETED_CHECKPOINT')
  assert sha(RUN/best_pointer['path'])==best_pointer['sha256']
  state=torch.load(RUN/best_pointer['path'],map_location=device,weights_only=False);model.load_state_dict(state['model']);del state
  full=score(held);primary,strongscore=views(full);training=score(fit);mon=score(monitor)
  assert primary['predictions']==read(RUN/f"predictions/primary80_epoch_{best_pointer['epoch']:03d}.json")['predictions']
  for name,result in [('train1500',training),('monitor',mon),('primary80',primary),('strong27',strongscore),('full206',full)]:save('selected_'+name+'.json',result)
  def screen(score):return score['gloss_WER']<=.5 and score['unique_predicted_tokens']>=10 and score['multi_token_hypotheses']>=.25*score['utterances']
  trainpass=screen(training) and screen(mon);evalpass=primary['gloss_WER']<=.9 and primary['unique_predicted_tokens']>=10 and primary['multi_token_hypotheses']>=20
  summary={'status':reason,'completed_epochs':completed,'optimizer_updates':updates,'utterance_presentations':presentations,'selected_checkpoint':best_pointer,'initialization_checkpoint':initial,
   'TRAIN_WER':training['gloss_WER'],'primary80_WER':primary['gloss_WER'],'strong27_WER':strongscore['gloss_WER'],'full206_WER':full['gloss_WER'],
   'TRAIN_learning_screen_pass':trainpass,'primary_transfer_screen_pass':evalpass,'source_transfer_interpretable':trainpass,'further_training_justified':trainpass and evalpass,
   'epoch_loop_seconds':loopseconds,'with_reload_full_TRAIN_validation_seconds':time.perf_counter()-started,'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_VRAM_bytes':torch.cuda.max_memory_reserved(),
   'GPU':torch.cuda.get_device_name(0),'used_cuda':True,'checkpoint_reload_predictions_match':True,'fold_manifest_sha256':expected,'vocabulary_sha256':sha(RUN/'vocabulary.json'),'config_sha256':sha(RUN/'config.json'),
   'normalization_sha256':sha(RUN/'normalization.npz'),'training_script_sha256':sha(__file__),'TEST_used':False,'production_promoted':False,'interpretation':'Known signers; unseen conservative recording families. No causal signer or unseen-signer claim.'}
  save('training_status.json',summary);print(json.dumps(summary),flush=True)
 except Exception as error:
  save('failure.json',{'error':str(error),'completed_epochs':completed,'optimizer_updates':updates,'selected_checkpoint':best_pointer,'preserve_completed_artifacts':True});raise

if __name__=='__main__':main()
