"""TRAIN-only framing discriminator; exactly one conditional fixed192 augmentation fit."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import argparse, csv, hashlib, json, os, random, shutil, subprocess, sys, threading, time, traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
OLD=RUN.parent/'20261021T112738_nested192_control01'
PREV=RUN.parent/'20261020T143609Z_learning_diagnostic01'
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'
sys.path[:0]=[str(ROOT),str(ROOT/'tools/continuous_asl')]

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def csvsave(name,rows):
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def family(r):
 import re
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
 return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def snapshot(paths):
 result={}
 for p in sorted(set(map(Path,paths))):
  s=p.stat();result[str(p)]=dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 return result

def translate(x,dx,dy):
 """Constant normalized translation, current valid XY only; never clip/alter deltas or masks."""
 assert x.ndim==2 and x.shape[1]==378 and x.dtype==np.float32 and np.isfinite(x).all()
 y=x.copy();offset=np.asarray([dx,dy],np.float32)
 y[:,:84]= (x[:,:84].reshape(-1,42,2)+x[:,84:126,None]*offset).reshape(-1,84)
 for side in range(2):
  y[:,176+side*3:178+side*3]+=x[:,174+side,None]*offset
 y[:,182:198]=(x[:,182:198].reshape(-1,8,2)+x[:,206:214,None]*offset).reshape(-1,16)
 assert np.isfinite(y).all()
 return y

def synthetic_parity():
 from pipeline.perception_vnext.recognition_schema import RawSequence,RAW_SCHEMA
 from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features,FEATURE_BLOCKS
 rng=np.random.default_rng(940721);t=13
 sizes=np.asarray([[480+(i%2)*160,360+(i%3)*60] for i in range(t)],np.int32)
 valid=rng.random((t,2,21))>.25;observed=rng.random((t,2))>.3
 proposals=rng.random((t,2))>.3;bodyvalid=rng.random((t,8))>.2
 raw=RawSequence(metadata={'schema':RAW_SCHEMA,'image_semantics':'UPRIGHT_UNMIRRORED_SOURCE_RGB','fixture':'SYNTHETIC_NO_PARTICIPANTS'},
  frame_index=np.arange(t,dtype=np.int32)*3,timestamp_ms=np.arange(t)*100.1,source_size=sizes,
  hand_xy=(rng.uniform(.15,.85,(t,2,21,2))*sizes[:,None,None,:]).astype(np.float32),
  hand_proposal_support=rng.uniform(.1,.99,(t,2,21)).astype(np.float32),hand_joint_valid=valid,
  hand_presence=rng.uniform(.3,.99,(t,2)).astype(np.float32),hand_observed=observed,hand_proposal_present=proposals,
  hand_roi_center_side=np.concatenate([(rng.uniform(.2,.8,(t,2,2))*sizes[:,None,:]).astype(np.float32),np.full((t,2,1),72,np.float32)],axis=2),
  hand_handedness_diagnostic=np.zeros((t,2),np.float32),body_xy=(rng.uniform(.1,.9,(t,8,2))*sizes[:,None,:]).astype(np.float32),
  body_support=rng.uniform(.1,.99,(t,8)).astype(np.float32),body_valid=bodyvalid,body_available=bodyvalid.sum(1)>=6)
 x=derive_causal_segmentation_features(raw);original=x.copy();maximum=0.
 position=np.r_[np.arange(84),[176,177,179,180],np.arange(182,198)]
 untouched=np.setdiff1d(np.arange(378),position)
 trials=[(.03,0.),(-.03,0.),(0.,.03),(0.,-.03),(.03,-.03),(0.,0.)]
 for dx,dy in trials:
  offset=np.asarray([dx,dy],np.float32)
  hand=raw.hand_xy+sizes[:,None,None,:]*offset
  body=raw.body_xy+sizes[:,None,:]*offset
  roi=raw.hand_roi_center_side.copy();roi[:,:,:2]+=sizes[:,None,:]*offset
  shifted=replace(raw,hand_xy=hand.astype(np.float32),body_xy=body.astype(np.float32),hand_roi_center_side=roi)
  expected=derive_causal_segmentation_features(shifted);actual=translate(x,dx,dy)
  error=float(np.max(np.abs(expected-actual)));maximum=max(maximum,error)
  assert np.allclose(actual,expected,rtol=0.,atol=3e-7),('RAW_TRANSLATION_PARITY',error)
  assert np.array_equal(actual[:,untouched],x[:,untouched]),'NONPOSITION_CHANNEL_MUTATION'
  assert np.all(actual[:,:84].reshape(t,42,2)[x[:,84:126]==0]==0),'INVALID_HAND_ZERO_MUTATION'
  assert np.all(actual[:,182:198].reshape(t,8,2)[x[:,206:214]==0]==0),'INVALID_BODY_ZERO_MUTATION'
  for side in range(2):assert np.all(actual[x[:,174+side]==0,176+side*3:179+side*3]==0)
 assert np.array_equal(x,original),'INPUT_MUTATION'
 return dict(status='PASS',fixture='SYNTHETIC_ONLY_NOT_MODEL_ACCURACY',frames=t,trials=len(trials),max_abs_error=maximum,absolute_tolerance=3e-7,invalid_zeros_preserved=True,nonposition_channels_exact=True,original_unchanged=True,feature_blocks=[b.__dict__ for b in FEATURE_BLOCKS],limitation='Feature-space translation does not rerun perception or simulate crop/occlusion/physical signer differences.')

def freeze_config():
 assert not (RUN/'config.json').exists(),'CONFIG_ALREADY_FROZEN'
 cfg=dict(schema=SCHEMA,scope='ISOLATED_RESEARCH_TRAIN_ONLY_FRAMING_DIAGNOSTIC_AND_CONDITIONAL_SINGLE_FIT',
  seed=20261020,augmentation_seed=20261021,offsets=[[.03,0.],[-.03,0.],[0.,.03],[0.,-.03]],
  sensitivity_gate=dict(baseline_WER=.10587431693989072,tolerance=1e-6,minimum_absolute_WER_worsening=.10,minimum_offsets_worsened=3),
  intended_changed_variable='Only per-utterance per-epoch constant XY translation, independent RNG, uniform[-.03,.03] each axis; no clipping.',
  model_channels=48,classes=2289,batch_size=4,accumulation=2,accumulation_tail='actual group count',
  lr=.001,weight_decay=.0001,gradient_clip=5.,precision='FP32',max_epochs=30,max_optimizer_steps=720,max_fit_seconds=180,
  evaluations=[0,1,5,10,15,20,25,30],checkpoint_epochs=[0,5,10,20,30],selection='Fixed finalepoch30 only; no development selection.',
  collapse_stop='At epoch10 stop if TRAINWER>=.95 and (dominant_nonblank_share>=.90 or token_diversity<=1).',
  normalization='Exact original1500TRAIN-only mean/std; never refit',decoder='Unchanged literal greedy CTC blank0/UNK1',
  heldout='Historical development-exposed fixed80 and original27subgroup, once after final30 frozen; known signers, new recording families; no unseen-signer claim.',
  subset_sha256=sha(OLD/'subset_manifest.json'),vocabulary_sha256=sha(OLD/'vocabulary.json'),normalization_sha256=sha(OLD/'normalization.npz'),fold_sha256=sha(OLD/'fold_manifest.json'),
  initialization_sha256=sha(OLD/'checkpoints/initialization.pt'),baseline_checkpoint_sha256=sha(OLD/'checkpoints/epoch_030.pt'),script_sha256=sha(__file__),
  model_source_sha256=sha(ROOT/'tools/continuous_asl/ctc_experiment.py'),feature_source_sha256=sha(ROOT/'pipeline/perception_vnext/segmentation/learned_features.py'),
  TEST_used=False,new_data=False,new_labels=False,promote=False)
 save('config.json',cfg)
 for name in ['subset_manifest.json','vocabulary.json','normalization.npz','fold_manifest.json']:shutil.copyfile(OLD/name,RUN/name)
 return cfg

def load_inputs():
 cfg=read(RUN/'config.json');assert sha(__file__)==cfg['script_sha256']
 for name,key in [('subset_manifest.json','subset_sha256'),('vocabulary.json','vocabulary_sha256'),('normalization.npz','normalization_sha256'),('fold_manifest.json','fold_sha256')]:
  assert sha(RUN/name)==sha(OLD/name)==cfg[key]
 assert sha(OLD/'checkpoints/initialization.pt')==cfg['initialization_sha256']
 assert sha(OLD/'checkpoints/epoch_030.pt')==cfg['baseline_checkpoint_sha256']
 assert sha(ROOT/'tools/continuous_asl/ctc_experiment.py')==cfg['model_source_sha256']
 assert sha(ROOT/'pipeline/perception_vnext/segmentation/learned_features.py')==cfg['feature_source_sha256']
 rows=read(RUN/'subset_manifest.json');vocab=read(RUN/'vocabulary.json')['tokens'];fold=read(RUN/'fold_manifest.json')
 assert len(rows)==len({r['sample_id'] for r in rows})==192
 assert {r['sample_id'] for r in rows}<=set(fold['train_sample_ids'])
 assert not {r['sample_id'] for r in rows}&set(fold['internal_eval_sample_ids'])
 assert len(vocab)==2289 and vocab[:2]==['<BLANK>','<UNK>']
 assert {r['training_pool'] for r in rows}=={'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE'}
 data={}
 for r in rows:
  assert sha(r['feature_path'])==r['feature_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA
   x=z['X'].copy();ts=z['timestamp_ms'];meta=json.loads(str(z['metadata_json'].item()))
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
   assert meta['sample_id']==r['sample_id'] and meta['glosses']==r['glosses']
   assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
   assert all(t in vocab[2:] for t in r['glosses'])
   data[r['sample_id']]=x
 with np.load(RUN/'normalization.npz',allow_pickle=False) as z:mean=z['mean'].copy();std=z['std'].copy()
 return cfg,rows,vocab,fold,data,mean,std

def aggregate(pp):
 from ctc_experiment import edit_counts
 counts=Counter();words=Counter()
 for p in pp:counts.update(edit_counts(p['reference'],p['hypothesis']));words.update(p['hypothesis'])
 n=counts['reference_tokens']
 return dict(utterances=len(pp),WER=sum(counts[k] for k in ['substitutions','deletions','insertions'])/n,**dict(counts),CTC_loss=float(np.mean([p['CTC_loss'] for p in pp])),blank_fraction=sum(p['blank_frames'] for p in pp)/sum(p['output_frames'] for p in pp),token_diversity=len(words),dominant_nonblank_share=words.most_common(1)[0][1]/sum(words.values()) if words else 0.,empty_outputs=sum(not p['hypothesis'] for p in pp),hypothesis_tokens=sum(words.values()),exact_sequences=sum(p['reference']==p['hypothesis'] for p in pp))

def evaluate(model,rows,data,vocab,offset=(0.,0.)):
 import torch
 from ctc_experiment import batch,decode
 model.eval();criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);pp=[]
 with torch.inference_mode():
  for r in rows:
   values=data[r['sample_id']]
   if offset!=(0.,0.):values=translate(values,*offset)
   x,y,l,tl=batch([r],{r['sample_id']:values},vocab,'cuda:0');z=model(x)
   loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl);assert torch.isfinite(loss),'NONFINITE_EVAL'
   ids=z[0].argmax(-1);hyp=decode(z[0],vocab)
   pp.append(dict(sample_id=r['sample_id'],reference=r['glosses'],hypothesis=hyp,CTC_loss=float(loss),blank_frames=int((ids==0).sum()),output_frames=len(ids),family=family(r)))
 return dict(**aggregate(pp),predictions=pp)

def diagnose():
 cfg=freeze_config()
 # Freeze precedes model imports/loading/inference. The verifier was executed read-only by the agent.
 paths=list(map(Path,read(OLD/'control_input_hashes_before.json')))
 paths += [OLD/p for p in ['verify192.py','artifact_hashes.sha256','decision.json','RESULTS.json','checkpoints/initialization.pt','checkpoints/epoch_030.pt']]
 paths += [ROOT/'pipeline/perception_vnext/segmentation/learned_features.py',ROOT/'pipeline/perception_vnext/recognition_schema.py']
 save('input_hashes_before.json',snapshot(paths))
 parity=synthetic_parity();save('synthetic_translation_parity.json',parity)
 cfg,rows,vocab,fold,data,mean,std=load_inputs()
 import torch
 from ctc_experiment import CausalCTC,seeds
 seeds(cfg['seed']);assert torch.cuda.is_available(),'CUDA_REQUIRED'
 model=CausalCTC(len(vocab),48,mean,std).to('cuda:0')
 ck=torch.load(OLD/'checkpoints/epoch_030.pt',map_location='cpu',weights_only=False)
 assert ck['vocabulary']==vocab and all(torch.equal(ck['model'][k],torch.as_tensor(v)) for k,v in [('mean',mean),('std',std)])
 model.load_state_dict(ck['model'],strict=True);started=time.perf_counter();scores=[]
 for dx,dy in [[0.,0.]]+cfg['offsets']:
  score=evaluate(model,rows,data,vocab,(dx,dy));name=f'baseline' if dx==dy==0 else f'x{dx:+.2f}_y{dy:+.2f}'
  save(f'sensitivity_predictions/{name}.json',score)
  scores.append(dict(scope=name,dx=dx,dy=dy,**{k:v for k,v in score.items() if k!='predictions'}))
 baseline=scores[0]['WER'];assert abs(baseline-cfg['sensitivity_gate']['baseline_WER'])<=1e-6,'BASELINE_REPRODUCTION_FAILED'
 for r in scores:r['absolute_WER_worsening']=r['WER']-baseline
 count=sum(r['absolute_WER_worsening']>=.10 for r in scores[1:])
 passed=count>=3
 csvsave('SENSITIVITY.csv',scores)
 save('sensitivity_gate.json',dict(status='PASS' if passed else 'FAIL',eligible_for_one_fit=passed,baseline_WER=baseline,offsets_worsened_at_least_10pp=count,required=3,inference_seconds=time.perf_counter()-started,configuration_frozen_before_inference=True,TRAIN_only=True,TEST_used=False))
 save('environment.json',dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(0),device_total_bytes=torch.cuda.get_device_properties(0).total_memory,device_free_bytes=torch.cuda.mem_get_info()[0],nondeterminism='CUDA CTC warn_only; no seed-equivalent causal claim'))
 print(json.dumps(dict(gate=read(RUN/'sensitivity_gate.json'),scores=scores),indent=2),flush=True)
 if not passed:finish_report()

def fit():
 import torch
 from ctc_experiment import CausalCTC,batch,seeds
 assert read(RUN/'sensitivity_gate.json')['eligible_for_one_fit'],'SENSITIVITY_GATE_BLOCKED'
 assert not (RUN/'execution_started.json').exists(),'EXACTLY_ONE_FIT_ONLY'
 cfg,rows,vocab,fold,data,mean,std=load_inputs()
 assert snapshot(read(RUN/'input_hashes_before.json'))==read(RUN/'input_hashes_before.json'),'INPUT_CHANGED'
 seeds(cfg['seed']);model=CausalCTC(len(vocab),48,mean,std).to('cuda:0')
 initial=torch.load(OLD/'checkpoints/initialization.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(initial['model'],strict=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 opt.load_state_dict(initial['optimizer'])
 random.setstate(initial['python_rng']);np.random.set_state(initial['numpy_rng']);torch.set_rng_state(initial['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in initial['cuda_rng']])
 assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in initial['model'].items()),'INITIALIZATION_MISMATCH'
 aug=np.random.default_rng(cfg['augmentation_seed']);criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 started=time.perf_counter();steps=0;complete=0;reason='FIXED_30_EPOCHS';metrics=[];timings=[];checkpoints=[];last=None
 save('execution_started.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),config_sha256=sha(RUN/'config.json'),initial_model_tensors_identical=True,optimizer_state_identical=True,shuffle_RNG_restored=True,augmentation_RNG_separate=True))
 torch.cuda.reset_peak_memory_stats()
 def checkpoint(epoch,partial=None):
  name=f'epoch_{epoch:03d}.pt' if partial is None else f'partial_epoch_{partial[0]:03d}.pt'
  p=RUN/'checkpoints'/name;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
  state=dict(schema=SCHEMA,epoch=epoch,optimizer_steps=steps,model=model.state_dict(),optimizer=opt.state_dict(),vocabulary=vocab,config=cfg,config_sha256=sha(RUN/'config.json'),subset_sha256=cfg['subset_sha256'],python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),augmentation_rng=aug.bit_generator.state,elapsed_seconds=time.perf_counter()-started)
  if partial:state.update(partial_epoch=partial[0],next_microbatch_group_index=partial[1],order_ids=partial[2],pending_gradients=False)
  torch.save(state,p);record=dict(epoch=epoch,path=str(p.relative_to(RUN)),sha256=sha(p),optimizer_steps=steps,partial=partial is not None);checkpoints.append(record);return record
 def score(epoch,train_loss=None):
  nonlocal last
  last=evaluate(model,rows,data,vocab);save(f'train_predictions/epoch_{epoch:03d}.json',last)
  record=dict(epoch=epoch,optimizer_steps=steps,elapsed_seconds=time.perf_counter()-started,training_augmented_CTC_loss=train_loss,**{k:v for k,v in last.items() if k!='predictions'},peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  metrics.append(record);print(json.dumps(record),flush=True)
 score(0);checkpoint(0)
 for epoch in range(1,31):
  tick=time.perf_counter();model.train();order=list(rows);random.shuffle(order)
  # Consume only the independent augmentation RNG; normal shuffle/torch RNG untouched.
  shifted={r['sample_id']:translate(data[r['sample_id']],*aug.uniform(-.03,.03,2)) for r in order}
  micro=[order[i:i+4] for i in range(0,len(order),4)];losses=[];interrupted=False
  for gs in range(0,len(micro),2):
   if time.perf_counter()-started>=180 or steps>=720:
    checkpoint(complete,(epoch,gs,[r['sample_id'] for r in order]));reason='WALL_OR_STEP_BOUND';interrupted=True;break
   assert shutil.disk_usage(RUN).free>=10*2**30,'DISK_HEADROOM_STOP'
   assert torch.cuda.mem_get_info()[0]>=256*2**20,'FREE_VRAM_STOP'
   assert torch.cuda.max_memory_allocated()<=2.5*2**30,'VRAM_CAP_STOP'
   group=micro[gs:gs+2];opt.zero_grad(set_to_none=True)
   for rr in group:
    x,y,l,tl=batch(rr,shifted,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
    assert torch.isfinite(loss),'NONFINITE_TRAIN_CTC';(loss/len(group)).backward();losses.append(float(loss.detach()))
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm),'NONFINITE_GRADIENT';opt.step();steps+=1
  torch.cuda.synchronize();timings.append(dict(epoch=epoch,completed=not interrupted,optimizer_epoch_seconds=time.perf_counter()-tick,optimizer_steps=steps))
  if interrupted:break
  complete=epoch
  if epoch in cfg['evaluations']:score(epoch,float(np.mean(losses)))
  if epoch in cfg['checkpoint_epochs']:checkpoint(epoch)
  if epoch==10 and last['WER']>=.95 and (last['dominant_nonblank_share']>=.9 or last['token_diversity']<=1):reason='COLLAPSE_GATE_EPOCH10';break
 elapsed=time.perf_counter()-started
 csvsave('METRICS.csv',metrics);csvsave('epoch_timing.csv',timings)
 save('training_status.json',dict(status='COMPLETED' if complete==30 else 'STOPPED',reason=reason,complete_epochs=complete,last_scored_epoch=metrics[-1]['epoch'],optimizer_steps=steps,fit_scoring_checkpoint_seconds=elapsed,optimizer_epoch_seconds_sum=sum(r['optimizer_epoch_seconds'] for r in timings),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_TRAIN={k:v for k,v in last.items() if k!='predictions'},checkpoints=checkpoints,selected_checkpoint=checkpoints[-1] if complete==30 else None,promote=False,TEST_used=False))
 if complete==30:
  selected=checkpoints[-1];assert selected['epoch']==30 and sha(RUN/selected['path'])==selected['sha256']
  save('final_checkpoint_frozen.json',dict(**selected,selection='FIXED_EPOCH30_BEFORE_HELDOUT_INFERENCE',heldout_never_selected_checkpoint=True))
  heldout(model,cfg,rows,vocab,fold)

def heldout(model,cfg,train,vocab,fold):
 # Only called after the final checkpoint/hash is frozen; fixed historical development scope.
 universe={r['sample_id']:r for r in read(PREV/'resolved_manifest.json')}
 rows=[universe[i] for i in fold['compatible_eval_sample_ids']];strong=set(fold['support5_eval_sample_ids'])
 assert len(rows)==80 and len(strong)==27 and strong<={r['sample_id'] for r in rows}
 assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in rows}
 assert not {family(r) for r in train}&{family(r) for r in rows}
 data={};hashes={}
 for r in rows:
  p=Path(r['feature_path']);assert sha(p)==r['feature_sha256'];hashes[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA
   x=z['X'].copy();ts=z['timestamp_ms'];meta=json.loads(str(z['metadata_json'].item()))
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
   assert meta['sample_id']==r['sample_id'] and meta['glosses']==r['glosses'] and all(t in vocab[2:] for t in r['glosses'])
   assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
   data[r['sample_id']]=x
 tick=time.perf_counter();results=evaluate(model,rows,data,vocab)
 save('heldout80_predictions.json',results)
 save('heldout27_subgroup.json',aggregate([p for p in results['predictions'] if p['sample_id'] in strong]))
 save('heldout_integrity.json',dict(status='PASS',feature_hashes=hashes,source_family_disjoint=True,same_known_signers=True,TEST_used=False,score_seconds=time.perf_counter()-tick))
 examples=[]
 for scope,name in [('TRAIN192','train_predictions/epoch_030.json'),('HELDOUT80','heldout80_predictions.json')]:
  for p in read(RUN/name)['predictions']:examples.append(dict(scope=scope,sample_id=p['sample_id'],reference=json.dumps(p['reference'],ensure_ascii=False),hypothesis=json.dumps(p['hypothesis'],ensure_ascii=False),CTC_loss=p['CTC_loss']))
 csvsave('PREDICTION_EXAMPLES.csv',examples)

def finish_report():
 before=read(RUN/'input_hashes_before.json');after=snapshot(before);changed=[p for p in before if before[p]!=after[p]]
 assert not changed,('FROZEN_INPUT_CHANGED',changed)
 save('integrity_results.json',dict(status='PASS',original_files=len(before),changed=changed,TEST_used=False))
 gate=read(RUN/'sensitivity_gate.json');fitstatus=read(RUN/'training_status.json') if (RUN/'training_status.json').exists() else None
 lines=['# Fixed192 framing discriminator and conditional augmentation pilot','',
  '**PARTIAL overall recognition.** No production/Android change, vocabulary expansion, acquisition or reserved TEST use.','',
  '## Frozen hypothesis and discriminator',
  'Constant normalized XY translations probe sensitivity to feature-space framing. This is not new signing data, perception/crop augmentation, or proof of a camera/signer cause.',
  f"Frozen192 baseline TRAIN WER: {gate['baseline_WER']*100:.4f}%. At least10pp worsening occurred for{gate['offsets_worsened_at_least_10pp']}/4predeclared offsets; gate {gate['status']}.",
  'Synthetic raw-observation→378feature parity, invalid-zero preservation and exact unchanged nonposition channels are recorded separately. All configuration was written before new inference. Prior verification reused verify192.py (216inputs,80heldout hashes,65output hashes PASS).',
  '']
 if fitstatus:
  lines += ['## Exactly one conditional training fit',f"Status{fitstatus['status']}; reason{fitstatus['reason']}; completed{fitstatus['complete_epochs']}epochs/{fitstatus['optimizer_steps']}updates; measured fit/scoring/checkpoint time{fitstatus['fit_scoring_checkpoint_seconds']:.3f}s.",f"Peak allocated{fitstatus['peak_allocated_bytes']/2**20:.2f}MiB; reserved{fitstatus['peak_reserved_bytes']/2**20:.2f}MiB. TRAIN WER{fitstatus['final_TRAIN']['WER']*100:.3f}%,CTC{fitstatus['final_TRAIN']['CTC_loss']:.6f}.",
   'Original initialization tensors/optimizer/shuffle RNG,192membership/vocabulary/normalization/model/30passes/720updates are unchanged. The independent augmentation RNG samples one constant x/y offset per utterance per epoch,uniform[-.03,.03]. No new images/features are saved as acquired data. CUDA CTC remains nondeterministic in warn-only mode, so this is not a seed-equivalent causal estimate.']
  if (RUN/'heldout80_predictions.json').exists():
   held=read(RUN/'heldout80_predictions.json');sub=read(RUN/'heldout27_subgroup.json');old=read(OLD/'RESULTS.json')
   lines += [f"Fixed final30 heldout80WER{held['WER']*100:.4f}% versus historical same192baseline{old['heldout80']['WER']*100:.4f}%; exact{held['exact_sequences']}/80. Original27subgroupWER{sub['WER']*100:.4f}%. The subgroup does not replace80or select a model.",
    'The80set is development-exposed and known-signer/family-disjoint. Do not claim pristine TEST, unseen-signer or reliable streaming improvement. A favorable result remains exploratory, not promotion evidence.']
 else:lines += ['## No fit executed','The TRAIN-only framing discriminator failed the predeclared gate. Do not perform the conditional augmentation fit; select a different evidence-backed discriminator rather than relax this gate.']
 lines += ['', '## Vocabulary and next action','The model already has2,287literal labels. Increasing labels is unjustified without new authentic examples/support and independent evaluation. Keep ASLLRP3 andNCSLGR2 supervision separate. Qualified review and consented new-signer/no-sign streaming evaluation remain unavailable; do not invent them.',f"Original files unchanged:{len(before)}. SeeSENSITIVITY.csv,METRICS.csv(if fit),PREDICTION_EXAMPLES.csv(if post-fit),config.json,commands_and_results.txt,andartifact_hashes.sha256."]
 (RUN/'PILOT_REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
 commands=[f'Working directory: {ROOT}',f'Python: {ROOT/"experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe"}',f'Executed read-only prior verification: python -B "{OLD/"verify192.py"}"; PASS216inputs/80heldout hashes/65outputs.',f'New discriminator: python -B "{Path(__file__)}" --diagnose',f'Conditional single fit only if gate passes: python -B "{Path(__file__)}" --fit-if-gated',f'Read-only new verification: python -B "{Path(__file__)}" --verify','No previous training or acquisition was repeated. All computed/synthetic scopes are explicitly diagnostic.']
 (RUN/'commands_and_results.txt').write_text('\n'.join(commands)+'\n',encoding='utf-8')
 files=[p for p in RUN.rglob('*') if p.is_file() and p.name not in ['artifact_hashes.sha256','run.log'] and '__pycache__' not in p.parts]
 (RUN/'artifact_hashes.sha256').write_text('\n'.join(sha(p)+'  '+str(p.relative_to(RUN)) for p in sorted(files))+'\n')
 print(json.dumps(dict(gate=gate,fit=fitstatus,originals_unchanged=len(before),overall='PARTIAL',promoted=False)),flush=True)

def verify():
 cfg=read(RUN/'config.json');assert sha(__file__)==cfg['script_sha256']
 before=read(RUN/'input_hashes_before.json');assert snapshot(before)==before
 for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
  digest,rel=line.split('  ',1);assert sha(RUN/rel)==digest,rel
 gate=read(RUN/'sensitivity_gate.json')
 if (RUN/'training_status.json').exists():
  d=read(RUN/'training_status.json');assert gate['eligible_for_one_fit']
  assert d['complete_epochs']<=30 and d['optimizer_steps']<=720 and not d['promote'] and not d['TEST_used']
  for cp in d['checkpoints']:assert sha(RUN/cp['path'])==cp['sha256']
 else:assert not gate['eligible_for_one_fit'] and not (RUN/'execution_started.json').exists()
 print(json.dumps(dict(integrity='PASS',originals_unchanged=len(before),sensitivity_gate=gate['status'],fit_occurred=(RUN/'training_status.json').exists(),TEST_used=False,promoted=False)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnose',action='store_true');p.add_argument('--fit-if-gated',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
 if a.diagnose:diagnose()
 elif a.fit_if_gated:
  stop=threading.Event()
  def monitor():
   with (RUN/'gpu_monitor.csv').open('x',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['UTC','device_used_MiB','utilization_percent'])
    while not stop.is_set():
     r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
     if r.returncode==0:writer.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())]+r.stdout.strip().split(', '));f.flush()
     stop.wait(1.)
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:fit()
  finally:stop.set();thread.join(timeout=6)
  finish_report()
 elif a.verify:verify()
 else:p.error('Choose --diagnose, --fit-if-gated, or --verify')

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if not (RUN/'failure.json').exists():save('failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),preserve_all_artifacts=True))
  raise
