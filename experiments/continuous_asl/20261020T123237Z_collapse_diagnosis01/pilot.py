"""Small same-pool learnability diagnostic, not a production candidate."""
from audit import *
import random,shutil

def main():
 assert read(RUN/'critical_checks.json')['status']=='PASS_SCOPED_RESEARCH_PILOT'
 out=RUN/'pilot64'
 if out.exists():raise ValueError('NO_REPEAT_OR_OVERWRITE')
 out.mkdir();(out/'checkpoints').mkdir()
 def put(name,value):save('pilot64/'+name,value)
 allfit=read(PARENT/'data/processed/train_dev/manifest.json');lookup={r['sample_id']:r for r in allfit}
 selected=[lookup[r['sample_id']] for r in read(PARENT/'pilot_fresh48_v2/train_monitor_manifest.json')]
 assert len(selected)==64 and all(r['split']=='train' for r in selected)
 dev=[r for r in allfit if r['split']=='dev'];vocab=read(PARENT/'pilot_fresh48_v2/vocabulary.json')['tokens']
 put('manifest.json',selected+dev);manifest_hash=sha(out/'manifest.json')
 protocol={'scope':'64-TRAIN same-pool learnability, full413 publisher signer/source-disjoint DEV; no TEST or product claim',
  'selection':'Reuse preexisting64 source-stratified TRAIN monitor, selected before fitting; no score selection.',
  'variable':'TRAIN subset size/repeated exposure diagnostic. Same full2519vocab,1706TRAIN-only normalization,48channels,features,loss,greedydecoder,lr and initialization seed; not an equal-budget causal A/B.',
  'train':64,'dev':413,'manifest_sha256':manifest_hash,'parent_manifest_sha256':read(PARENT/'dataset_plan.json')['manifest_sha256'],
  'max_epochs':80,'wall_seconds':90,'batch':4,'gradient_accumulation':2,'seed':20261020,'lr':.001,'weight_decay':.0001,'gradient_clip':5.,
  'gate20':'Stop if TRAIN WER>=.98 or unique predicted tokens<2.',
  'gate40':'Continue only TRAINWER<=.8,TRAINunique>=10,TRAIN>=8multi-token outputs AND DEVWER<.9799134734239803 with>=5%reference tokens correct; otherwise stop.',
  'after40':'DEV patience2 at10epoch evaluation intervals; same meaningful sequence gates required.',
  'critical_stop':'Nonfinite loss/grad, allocatedVRAM>2.5GiB, freeVRAM<256MiB, disk<10GiB.',
  'normalization':'Existing1706TRAIN-only buffers, no recognition weights reused','vocab':'Existing1706TRAIN-only2519tokens; no label replacement in WER',
  'random_initialization':True,'test_accessed':False,'production_enabled':False,'script_sha256':sha(__file__)}
 put('protocol.json',protocol);put('vocabulary.json',{'tokens':vocab,'fit':'original1706TRAIN only'})
 data={}
 for r in selected+dev:
  assert sha(r['feature_path'])==r['feature_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as f:
   assert str(f['schema_version'].item())==SCHEMA;data[r['sample_id']]=f['X'].copy()
 for key in ['sample_id','signer_id','source_group','raw_sha256','decoded_pixel_sha256']:
  assert not ({r[key] for r in selected}&{r[key] for r in dev})
 parentstatus=read(PARENT/'pilot_fresh48_v2/training_status.json');cp=PARENT/'pilot_fresh48_v2'/parentstatus['best_checkpoint']['path'];assert sha(cp)==parentstatus['best_checkpoint']['sha256']
 state=torch.load(cp,map_location='cpu',weights_only=False)
 mean=state['model']['mean'].numpy().copy();std=state['model']['std'].numpy().copy();del state
 seeds(20261020);assert torch.cuda.is_available()
 model=CausalCTC(len(vocab),48,mean,std).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 baseline=measure(model,dev,data,vocab);put('untrained_dev.json',baseline)
 best=(float('inf'),float('inf'));pointer=None;patience=0;reason='MAX_EPOCHS';started=time.perf_counter();torch.cuda.reset_peak_memory_stats();completed=0
 train_tokens={t for r in selected for t in r['glosses']}
 put('subset_coverage.json',{'unique_train_target_tokens':len(train_tokens),'DEV_tokens_not_in64TRAIN':sum(t not in train_tokens for r in dev for t in r['glosses']),
  'DEV_reference_tokens':sum(len(r['glosses']) for r in dev),'TRAIN_signers':sorted({r['signer_id'] for r in selected}),'TRAIN_source_groups':len({r['source_group'] for r in selected})})
 try:
  for epoch in range(1,81):
   if time.perf_counter()-started>=90:reason='WALL_BUDGET';break
   tick=time.perf_counter();order=list(selected);random.shuffle(order);model.train();optimizer.zero_grad(set_to_none=True);losses=[]
   for j,start in enumerate(range(0,64,4)):
    if torch.cuda.mem_get_info()[0]<268435456 or shutil.disk_usage(ROOT).free<10737418240:raise RuntimeError('RESOURCE_HEADROOM')
    x,y,l,tl=batch(order[start:start+4],data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
    (loss/2).backward()
    if (j+1)%2==0:
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
     if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRAD')
     optimizer.step();optimizer.zero_grad(set_to_none=True)
    if torch.cuda.max_memory_allocated()>2684354560:raise RuntimeError('VRAM_CAP')
    losses.append(float(loss.detach()))
   completed=epoch;record={'epoch':epoch,'optimization_loss':float(np.mean(losses)),'fit_epoch_seconds':time.perf_counter()-tick}
   if epoch==1 or epoch%10==0:
    tr=measure(model,selected,data,vocab);dv=measure(model,dev,data,vocab);torch.cuda.synchronize()
    put(f'train_epoch_{epoch:03d}.json',tr);put(f'dev_epoch_{epoch:03d}.json',dv)
    score=(dv['gloss_wer'],dv['loss']);improved=score<best
    if improved:best=score;patience=0
    else:patience+=1
    path=out/'checkpoints'/f'epoch_{epoch:03d}.pt'
    torch.save({'schema':SCHEMA,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'vocabulary':vocab,'protocol':protocol,
     'manifest_sha256':manifest_hash,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path)
    if improved:pointer={'path':str(path.relative_to(out)),'sha256':sha(path),'epoch':epoch}
    multi=sum(len(p['hypothesis'])>=2 for p in tr['predictions'])
    record.update(train_WER=tr['gloss_wer'],dev_WER=dv['gloss_wer'],train_CTC=tr['loss'],dev_CTC=dv['loss'],
     train_blank=tr['blank_argmax_fraction'],dev_blank=dv['blank_argmax_fraction'],train_unique=tr['unique_predicted_tokens'],dev_unique=dv['unique_predicted_tokens'],
     train_multi_outputs=multi,dev_emitted_tokens=dv['prediction_tokens'],elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    print(json.dumps(record),flush=True)
    if epoch==20 and (tr['gloss_wer']>=.98 or tr['unique_predicted_tokens']<2):reason='COLLAPSE_GATE20'
    if epoch>=40:
     correct=dv['reference_tokens']-dv['substitutions']-dv['deletions']
     if not (tr['gloss_wer']<=.8 and tr['unique_predicted_tokens']>=10 and multi>=8 and dv['gloss_wer']<.9799134734239803 and correct>=.05*dv['reference_tokens']):reason='NO_MEANINGFUL_DEV_GAIN_GATE40'
     elif patience>=2:reason='DEV_PATIENCE'
   with (out/'training_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
   if reason!='MAX_EPOCHS':break
  loop=time.perf_counter()-started
  assert pointer is not None and sha(out/pointer['path'])==pointer['sha256']
  state=torch.load(out/pointer['path'],map_location='cuda:0',weights_only=False);model.load_state_dict(state['model'])
  tr=measure(model,selected,data,vocab);dv=measure(model,dev,data,vocab)
  assert dv['predictions']==read(out/f'dev_epoch_{pointer["epoch"]:03d}.json')['predictions']
  put('best_train.json',tr);put('best_dev.json',dv)
  put('training_status.json',{'status':reason,'completed_epochs':completed,'optimizer_updates':completed*8,'best_checkpoint':pointer,'manifest_sha256':manifest_hash,
   'TRAIN_WER':tr['gloss_wer'],'DEV_WER':dv['gloss_wer'],'TRAIN_CTC':tr['loss'],'DEV_CTC':dv['loss'],'TRAIN_unique':tr['unique_predicted_tokens'],'DEV_unique':dv['unique_predicted_tokens'],
   'fit_loop_seconds':loop,'with_reload_seconds':time.perf_counter()-started,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
   'used_cuda':True,'gpu':torch.cuda.get_device_name(0),'checkpoint_reload_predictions_match':True,'TEST_used':False,'promoted':False})
 except Exception as e:put('failure.json',{'error':str(e),'completed_epochs':completed});raise

if __name__=='__main__':main()
