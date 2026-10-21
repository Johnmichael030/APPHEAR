"""Gated future research B path. Missing admission evidence is a hard failure.

This driver has NOT been validated on a real expanded dataset. Only its blocked
preflight and the separate synthetic head/optimizer tests have run.
"""
from pathlib import Path
import argparse,copy,json,random,time
from collections import Counter
import numpy as np
import torch
import fine_tune as ft
from vocabulary_extension import extend,expanded_optimizer_state

def preflight(path):
 if not path.is_file():raise ValueError('B_ADMISSION_BLOCKED: missing '+str(path))
 d=ft.read(path)
 assert d['pool']=='ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE'
 for key in ['permission_review','target_contract_review','temporal_alignment_review']:
  r=d[key];assert r['status']=='PASS' and r['reviewed_by'] and ft.sha(r['evidence_path'])==r['sha256'],key
 assert d['target_contract_review']['qualified_ASL_review'] is True
 assert d['permission_review']['scope_includes_research_training'] is True
 assert d['temporal_alignment_review']['verified_crop_origin_and_cadence'] is True
 oldtrain=ft.read(ft.RUN/'manifests/TRAIN1500.json');olddev=ft.read(ft.RUN/'manifests/DEV80.json')
 oldv=ft.read(ft.RUN/'vocabulary.json')['tokens'];rows=d['records']
 assert rows and len({r['sample_id'] for r in rows})==len(rows)
 assert all(r['role'] in ['TRAIN','VAL'] for r in rows)
 train=[r for r in rows if r['role']=='TRAIN'];val=[r for r in rows if r['role']=='VAL'];assert train and val
 # Existing resolved metadata was already used in A: no reserved TEST is opened.
 resolved=ft.read(ft.BASE/'resolved_manifest.json');trainids={r['sample_id'] for r in oldtrain}
 prohibited=[r for r in resolved if r['sample_id'] not in trainids]
 for key in ['sample_id','raw_sha256','decoded_pixel_sha256','source_group']:
  assert not ({r[key] for r in rows}&{r[key] for r in resolved}),'NEW_RECORD_REUSES_OLD_'+key
  assert not ({r[key] for r in train}&{r[key] for r in val}),'NEW_SPLIT_OVERLAP_'+key
 assert not ({ft.family(r) for r in train+oldtrain}&{ft.family(r) for r in val+prohibited}),'FAMILY_GUARD'
 newlabels=sorted({t for r in train for t in r['glosses']}-set(oldv))
 assert newlabels and {t for r in val for t in r['glosses']}<=set(oldv+newlabels)
 approved=[];data={}
 for t in newlabels:
  nt=sum(t in r['glosses'] for r in train);nv=sum(t in r['glosses'] for r in val)
  assert nt>=3 and nv>=1,'INSUFFICIENT_DISTINCT_UTTERANCE_SUPPORT:'+t
  approved.append(dict(candidate_gloss=t,training_example_count=nt,validation_example_count=nv,permission_status='VERIFIED_RESEARCH_USE',annotation_status='VERIFIED_LITERAL_TARGET_CONTRACT',split_status='PASS_FAMILY_GUARD'))
 for r in rows:
  assert r['permission_status']=='VERIFIED_RESEARCH_USE' and r['annotation_status']=='VERIFIED_LITERAL_TARGET_CONTRACT'
  assert r['provenance']['exclusion_audit_status']=='PASS_NO_PREVIOUSLY_EXCLUDED_SAMPLE'
  assert r['schema_version']==ft.SCHEMA and ft.sha(r['feature_path'])==r['feature_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as z:
   assert str(z['schema_version'].item())==ft.SCHEMA
   x=z['X'].copy();ts=z['timestamp_ms'].copy();meta=json.loads(str(z['metadata_json'].item()))
  assert meta['sample_id']==r['sample_id'] and meta['glosses']==r['glosses']
  assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
  assert r['glosses'] and not set(r['glosses'])&set(oldv[:2])
  assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
  data[r['sample_id']]=x
 return d,train,val,approved,data

def fit(path,checked):
 d,newtrain,newval,approved,newdata=checked
 run=ft.RUN/'B'/time.strftime('admitted_%Y%m%dT%H%M%SZ',time.gmtime());run.mkdir(exist_ok=False)
 oldtrain,olddev,v,data,times,_=ft.load_rows();data.update(newdata)
 ft.seeds(20261021);old,s=ft.load_model(v);model,nv=extend(old,v,approved);model=model.to('cuda:0')
 opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
 opt.load_state_dict(expanded_optimizer_state(s['optimizer'],old,model))
 for group in opt.param_groups:group['lr']=.0001
 del old
 train=oldtrain+newtrain
 for name,value in [('TRAIN',train),('old_DEV80',olddev),('new_VAL',newval)]:ft.save(run/(name+'.json'),value)
 ft.save(run/'vocabulary.json',dict(tokens=nv,blank_id=0,unknown_id=1))
 cfg=dict(seed=20261021,max_epochs=12,max_seconds=180,learning_rate=.0001,batch=4,accumulation=2,gradient_clip=5,source_sha256=ft.sha(ft.SOURCE),admission_sha256=ft.sha(path),manifest_sha256={n:ft.sha(run/(n+'.json')) for n in ['TRAIN','old_DEV80','new_VAL']},vocabulary_sha256=ft.sha(run/'vocabulary.json'),normalization='frozen epoch12',sampling='uniform genuine utterances; old replay plus new TRAIN',selection='new VAL WER, constrained old DEV WER <= baseline + .02',TEST_used=False)
 ft.save(run/'TRAINING_CONFIGURATION.json',cfg)
 criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);start=time.perf_counter();history=[];best=float('inf');stale=0;selected=None;reason='MAX_EPOCHS'
 torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,13):
  if time.perf_counter()-start>=180:reason='WALL_LIMIT';break
  model.train();order=list(train);random.shuffle(order);micro=[order[i:i+4] for i in range(0,len(order),4)];losses=[];partial=False
  for i in range(0,len(micro),2):
   if time.perf_counter()-start>=180:partial=True;reason='WALL_LIMIT_PARTIAL_EPOCH_NOT_SELECTED';break
   assert torch.cuda.mem_get_info()[0]>=256*2**20 and torch.cuda.max_memory_allocated()<2.5*2**30
   group=micro[i:i+2];opt.zero_grad(set_to_none=True)
   for rr in group:
    x,y,l,tl=ft.batch(rr,data,nv,'cuda:0');loss=criterion(model(x).log_softmax(-1).transpose(0,1),y,l,tl);assert torch.isfinite(loss)
    (loss/len(group)).backward();losses.append(float(loss.detach()))
   assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step()
  if partial:break
  a=ft.score(model,olddev,data,nv);b=ft.score(model,newval,data,nv)
  monitor=ft.score(model,oldtrain[:128]+newtrain[:32],data,nv)
  for label,score in [('oldDEV',a),('newVAL',b),('TRAINmonitor',monitor)]:ft.save(run/f'{label}_epoch_{epoch:03d}.json',score)
  checkpoint=run/f'epoch_{epoch:03d}.pt'
  torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,vocabulary=nv,schema=ft.SCHEMA,config_sha256=ft.sha(run/'TRAINING_CONFIGURATION.json'),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
  improvement=a['WER']<=ft.read(ft.RUN/'baseline_greedy_DEV80.json')['WER']+.02 and b['WER']<best-.005
  if improvement:best=b['WER'];stale=0;selected=str(checkpoint)
  else:stale+=1
  row=dict(epoch=epoch,CTC_loss=float(np.mean(losses)),old_DEV_WER=a['WER'],new_VAL_WER=b['WER'],new_VAL_blank_fraction=b['blank_fraction'],new_VAL_diversity=b['token_diversity'],TRAIN_monitor_WER=monitor['WER'],elapsed_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=ft.sha(checkpoint))
  history.append(row);print(json.dumps(row),flush=True)
  collapsed=all(q['blank_fraction']>=.995 or (q['token_diversity']<=3 and q['top_token_fraction']>=.98) for q in [monitor,b])
  if epoch>=4 and collapsed:reason='COLLAPSE_STOP';break
  if stale>=4:reason='NO_CREDIBLE_NEW_VAL_GAIN_STOP';break
 ft.save(run/'RESULTS.json',dict(status='BOUNDED_RESEARCH_ONLY',reason=reason,selected=selected,history=history,actual_new_classes=len(nv)-len(v),production_promoted=False,TEST_used=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ft.RUN/'B/APPROVED_DATA_MANIFEST.json');p.add_argument('--train',action='store_true');a=p.parse_args()
 try:checked=preflight(a.manifest)
 except (AssertionError,ValueError,KeyError,FileNotFoundError) as e:print('B_ADMISSION_BLOCKED:',e);raise SystemExit(2)
 print('B_PREFLIGHT_PASS: manifest, literal targets, append mapping, permissions and recording families verified.')
 if a.train:fit(a.manifest,checked)
