"""One bounded whole-utterance restricted-task pilot; no production outputs."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from support_audit import *
import random,shutil
sys.path.insert(0,str(DIAG))
from audit import measure,torch,CausalCTC,batch,seeds,SCHEMA

def main():
 decision=read(RUN/'decision.json');assert decision['status']=='PASS_SCOPED_PILOT_CANDIDATE'
 out=RUN/'pilot497'
 if out.exists():raise ValueError('NO_OVERWRITE_OR_REPEATED_FIT')
 out.mkdir();(out/'checkpoints').mkdir()
 def put(name,obj):save('pilot497/'+name,obj)
 manifest=RUN/'frozen_manifest.json';assert sha(manifest)==decision['manifest_sha256']
 rows=read(manifest);fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];compatible=read(RUN/'compatible_dev.json')
 vocab=['<BLANK>','<UNK>']+sorted({t for r in fit for t in r['glosses']});assert len(fit)==497 and len(vocab)==425 and len(compatible)==92
 assert all(set(r['glosses'])<=set(vocab) for r in compatible)
 protocol={'manifest_sha256':sha(manifest),'compatible_manifest_sha256':sha(RUN/'compatible_dev.json'),'TRAIN':497,'original_DEV':413,'compatible_DEV':92,
  'vocabulary_classes':425,'vocabulary_fit':'497TRAIN only; literal labels unchanged','normalization_fit':'497TRAIN only','initialization':'fresh random; no prior recognition weights',
  'architecture':'same48channel causal378CTC; RF61; no downsampling','batch':4,'gradient_accumulation':2,'lr':.001,'weight_decay':.0001,'clip':5.,'precision':'FP32','seed':20261020,
  'max_epochs':30,'wall_seconds':90,'local_checkpoint_selection':'compatible92DEV tuple(WER,CTCloss); original413DEV report only',
  'gate10':'Stop if TRAINWER>=.98 and diversity<3, or both evaluations all blank with <1%TRAIN loss progress.',
  'gate20':'Continue only TRAINWER<=.8,TRAINdiversity>=10,TRAINmulti>=25%,compatibleDEVWER<.95 and>=5%correct reference tokens; otherwise stop.',
  'later':'Same gate plus compatibleDEV patience2 evaluated every5epochs. Nonfinite/VRAM>2.5GiB/free<256MiB/disk<10GiB stop.',
  'interpretation':'Restricted task; not comparable to full-vocabulary WER. FullDEV loss usesUNK for unsupported targets; fullDEVWER keeps literal references.',
  'script_sha256':sha(__file__),'TEST_used':False,'production_enabled':False}
 put('protocol.json',protocol);put('vocabulary.json',{'tokens':vocab,'fitted_split':'new497TRAIN'})
 data={}
 for r in rows:
  assert sha(r['feature_path'])==r['feature_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as f:
   assert str(f['schema_version'].item())==SCHEMA;x=f['X'].copy();ts=f['timestamp_ms']
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
  assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
  data[r['sample_id']]=x
 for key in ['signer_id','source_group','raw_sha256','decoded_pixel_sha256']:
  assert not ({r[key] for r in fit}&{r[key] for r in dev})
 values=np.concatenate([data[r['sample_id']] for r in fit]).astype(np.float64);mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32);del values
 seeds(20261020);assert torch.cuda.is_available();model=CausalCTC(len(vocab),48,mean,std).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 put('untrained_compatible_dev.json',measure(model,compatible,data,vocab))
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter();best=(float('inf'),float('inf'));pointer=None;patience=0;completed=0;reason='MAX_EPOCHS';previous=None
 try:
  for epoch in range(1,31):
   if time.perf_counter()-started>=90:reason='WALL_BUDGET';break
   tick=time.perf_counter();order=list(fit);random.shuffle(order);model.train();optimizer.zero_grad(set_to_none=True);losses=[];offsets=list(range(0,len(order),4))
   for j,start in enumerate(offsets):
    if torch.cuda.mem_get_info()[0]<268435456 or shutil.disk_usage(ROOT).free<10737418240:raise RuntimeError('RESOURCE_HEADROOM')
    x,y,l,tl=batch(order[start:start+4],data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
    (loss/2).backward()
    if (j+1)%2==0 or j+1==len(offsets):
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
     if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRAD')
     optimizer.step();optimizer.zero_grad(set_to_none=True)
    if torch.cuda.max_memory_allocated()>2684354560:raise RuntimeError('VRAM_CAP')
    losses.append(float(loss.detach()))
   completed=epoch;record={'epoch':epoch,'optimization_CTC_loss':float(np.mean(losses)),'fit_epoch_seconds':time.perf_counter()-tick}
   if epoch==1 or epoch%5==0:
    tr=measure(model,fit,data,vocab);cd=measure(model,compatible,data,vocab);fd=measure(model,dev,data,vocab);torch.cuda.synchronize()
    for key,result in [('train',tr),('compatible_dev',cd),('original_dev',fd)]:put(f'{key}_epoch_{epoch:03d}.json',result)
    rank=(cd['gloss_wer'],cd['loss']);improved=rank<best
    if improved:best=rank;patience=0
    else:patience+=1
    cp=out/'checkpoints'/f'epoch_{epoch:03d}.pt';torch.save({'schema':SCHEMA,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'vocabulary':vocab,'protocol':protocol,
     'manifest_sha256':sha(manifest),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},cp)
    if improved:pointer={'path':str(cp.relative_to(out)),'sha256':sha(cp),'epoch':epoch}
    multi=sum(len(p['hypothesis'])>=2 for p in tr['predictions'])
    record.update(train_WER=tr['gloss_wer'],compatible_DEV_WER=cd['gloss_wer'],original_DEV_WER=fd['gloss_wer'],train_CTC=tr['loss'],compatible_DEV_CTC=cd['loss'],original_DEV_CTC=fd['loss'],
     train_blank=tr['blank_argmax_fraction'],compatible_DEV_blank=cd['blank_argmax_fraction'],original_DEV_blank=fd['blank_argmax_fraction'],
     train_unique=tr['unique_predicted_tokens'],compatible_DEV_unique=cd['unique_predicted_tokens'],original_DEV_unique=fd['unique_predicted_tokens'],train_multi_outputs=multi,
     epoch_with_eval_seconds=time.perf_counter()-tick,elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    print(json.dumps(record),flush=True)
    if epoch==10 and ((tr['gloss_wer']>=.98 and tr['unique_predicted_tokens']<3) or (tr['prediction_tokens']==cd['prediction_tokens']==0 and previous is not None and tr['loss']>=previous*.99)):reason='COLLAPSE_GATE10'
    if epoch>=20:
     correct=cd['reference_tokens']-cd['substitutions']-cd['deletions']
     if not (tr['gloss_wer']<=.8 and tr['unique_predicted_tokens']>=10 and multi>=.25*len(fit) and cd['gloss_wer']<.95 and correct>=.05*cd['reference_tokens']):reason='NO_MEANINGFUL_LEARNING_GATE20'
     elif patience>=2:reason='COMPATIBLE_DEV_PATIENCE'
    previous=tr['loss']
   with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
   if reason!='MAX_EPOCHS':break
  loop=time.perf_counter()-started;assert pointer and sha(out/pointer['path'])==pointer['sha256']
  state=torch.load(out/pointer['path'],map_location='cuda:0',weights_only=False);model.load_state_dict(state['model'])
  scores={key:measure(model,samples,data,vocab) for key,samples in [('train',fit),('compatible_dev',compatible),('original_dev',dev)]}
  for key,result in scores.items():
   assert result['predictions']==read(out/f'{key}_epoch_{pointer["epoch"]:03d}.json')['predictions'];put('best_'+key+'.json',result)
  put('training_status.json',{'status':reason,'completed_epochs':completed,'optimizer_updates':completed*63,'best_checkpoint':pointer,'manifest_sha256':sha(manifest),
   'TRAIN_WER':scores['train']['gloss_wer'],'compatible_DEV_WER':scores['compatible_dev']['gloss_wer'],'original_DEV_WER':scores['original_dev']['gloss_wer'],
   'fit_loop_seconds':loop,'with_reload_seconds':time.perf_counter()-started,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
   'used_cuda':True,'gpu':torch.cuda.get_device_name(0),'checkpoint_reload_predictions_match':True,'TEST_used':False,'promoted':False})
 except Exception as e:put('failure.json',{'error':str(e),'completed_epochs':completed});raise

if __name__=='__main__':main()
