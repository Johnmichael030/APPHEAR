"""File/array/checkpoint forensics only. No model forward, backward, optimizer or GPU calls."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,sys,time
import numpy as np
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];PILOT=RUN.parent/'20261020T133148Z_source_transfer_pilot01'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
 with (RUN/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def snapshot(paths):
 return {str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)} for p in sorted(set(map(Path,paths)))}
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  now=[i]
  for j,y in enumerate(b,1):now.append(min(prev[j]+1,now[j-1]+1,prev[j-1]+(x!=y)))
  prev=now
 return prev[-1]
def summary(values):
 x=np.array(values,dtype=np.float64)
 return {'min':float(x.min()),'median':float(np.median(x)),'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'max':float(x.max())} if len(x) else None
def main():
 start=time.perf_counter();rows=read(PILOT/'resolved_manifest.json');fold=read(PILOT/'fold_manifest.json');cfg=read(PILOT/'config.json');status=read(PILOT/'training_status.json');vocab=read(PILOT/'vocabulary.json')['tokens']
 fit=[r for r in rows if r['internal_role']=='TRAIN'];held=[r for r in rows if r['internal_role']=='HELD_OUT'];assert (len(fit),len(held))==(1500,206)
 primaryids=set(fold['compatible_eval_sample_ids']);strongids=set(fold['support5_eval_sample_ids']);fitids={r['sample_id'] for r in fit};lookup={r['sample_id']:r for r in rows}
 sourcepaths={Path(fold['universe_manifest']),ROOT/'tools/continuous_asl/ctc_experiment.py',ROOT/'pipeline/perception_vnext/segmentation/learned_features.py',ROOT/'pipeline/perception_vnext/recognition_schema.py',ROOT/'tools/continuous_asl/prepare_features.py',
  RUN.parent/'20261020_continuous_expand06/tools/prepare_asllrp.py',RUN.parent/'20261020_continuous_expand06/tools/prepare_asllrp_front.py',RUN.parent/'20261020_continuous_expand06/tools/prepare_asllrp_tail.py',RUN.parent/'20261020_continuous_expand06/tools/asllrp_catalog.py'}
 rawpaths={}
 for r in rows:
  sourcepaths.update([Path(r['feature_path']),Path(r['video_path']),Path(r['annotation_provenance']['csv_path']),Path(r['signer_provenance']['source_page'])])
  raw=Path(r['feature_path'].replace('.features.npz','.raw.npz'));assert raw.exists();rawpaths[r['sample_id']]=raw;sourcepaths.add(raw)
 before=snapshot(sourcepaths);save('input_files_before.json',before)
 sourcecsv=Path(rows[0]['annotation_provenance']['csv_path'])
 with sourcecsv.open(encoding='utf-8-sig',newline='') as f:csvrows={n:r for n,r in enumerate(csv.DictReader(f),2)}
 assert sha(sourcecsv)==rows[0]['annotation_provenance']['csv_sha256']
 original={r['sample_id']:r for r in read(fold['universe_manifest']) if r['split']=='train'};assert set(lookup)<={*original}
 # Imports below are pure cached-landmark array math. No image/model inference is imported.
 sys.path.insert(0,str(ROOT))
 from pipeline.perception_vnext.recognition_schema import load_raw_sequence
 from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
 with np.load(PILOT/'normalization.npz',allow_pickle=False) as z:mean=z['mean'];std=z['std']
 checks=[];features={};content=defaultdict(list);eventcounts=defaultdict(Counter);groupfeatures=defaultdict(list);examples=[]
 for r in rows:
  sid=r['sample_id'];role='TRAIN' if sid in fitids else 'PRIMARY80' if sid in primaryids else 'HELD_OTHER'
  assert r['glosses']==original[sid]['glosses'] and r['feature_sha256']==original[sid]['feature_sha256'] and r['signer_id']==original[sid]['signer_id']
  assert before[r['feature_path']]['sha256']==r['feature_sha256'] and before[r['video_path']]['sha256']==r['raw_sha256']
  assert sha(r['signer_provenance']['source_page'])==r['signer_provenance']['source_page_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as z:
   x=z['X'].copy();ts=z['timestamp_ms'].copy();meta=json.loads(str(z['metadata_json'].item()));schema=str(z['schema_version'].item())
  assert schema==cfg['schema'] and x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
  for key in ['sample_id','collection','utterance_id','source_group','utterance_filename','glosses','raw_sha256']:assert meta[key]==r[key],(sid,'EMBEDDED_METADATA',key)
  assert [e['label'] for e in r['events']]==r['glosses']
  for e in r['events']:
   c=csvrows[e['source_row']]
   assert c['Occurrence label']==e['label'] and c['Video ID number']==e['event_id'] and c['Utterance video filename']==r['utterance_filename'] and c['Master video filename']==r['source_group'] and c['Source collection']==r['collection'] and c['Utterance number']==r['utterance_id']
   assert int(c['Start frame of the sign video'])==e['start_frame_master'] and int(c['End frame of the sign video'])==e['end_frame_master']
  raw=load_raw_sequence(rawpaths[sid]);rederived=derive_causal_segmentation_features(raw)
  assert np.array_equal(x,rederived) and np.array_equal(ts,raw.timestamp_ms)
  assert raw.metadata['sample_id']==sid and raw.metadata['glosses']==r['glosses']
  minimum=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]));assert len(x)>=minimum and len(x)==r['observations']
  assert r['decoded_frames']==r['utterance_end_frame_master']-r['utterance_start_frame_master']+1
  assert abs(r['native_fps']-30000/1001)<.001 and np.all(np.diff(ts)>=100-1e-6)
  hand=np.any(raw.hand_observed,axis=1);both=np.all(raw.hand_observed,axis=1);body=raw.body_available
  normalized=(x-mean)/std;clipfraction=float(np.mean(np.abs(normalized)>10));zero_motion=bool(np.all(x[:,216:300]==0))
  content[hashlib.sha256(x.tobytes()).hexdigest()].append(sid);features[sid]=x
  spans=r['utterance_start_frame_master'];miss=0;nohand=0
  for e in r['events']:
   a=e['start_frame_master']-spans;b=e['end_frame_master']-spans;mask=(raw.frame_index>=a)&(raw.frame_index<=b);n=int(mask.sum());no=n>0 and not bool(hand[mask].any())
   for group in [role]+(['STRONG27'] if sid in strongids else []):
    eventcounts[group]['target_events']+=1;eventcounts[group]['events_no_kept_frame']+=int(n==0);eventcounts[group]['sampled_events_no_observed_hand']+=int(no);eventcounts[group]['hidden_events']+=int(e['hidden'])
   miss+=int(n==0);nohand+=int(no)
   if n==0 and len(examples)<12:examples.append({'sample_id':sid,'role':role,'gloss':e['label'],'source_relative_frame_span':[a,b],'publisher_span_ms':(b-a+1)*1000/r['native_fps'],'kept_frames_in_event':0})
  item={'sample_id':sid,'role':role,'observations':len(x),'target_tokens':len(r['glosses']),'CTC_minimum_steps':minimum,'CTC_slack':len(x)-minimum,'duration_ms':float(ts[-1]-ts[0]),'median_dt_ms':float(np.median(np.diff(ts))),
   'observed_any_hand_fraction':float(hand.mean()),'both_hands_fraction':float(both.mean()),'body_available_fraction':float(body.mean()),'zero_hand_motion_sequence':zero_motion,'normalization_clip_fraction':clipfraction,
   'missing_event_observation_count':miss,'sampled_events_no_hand_count':nohand,'feature_transcript_source_join':'PASS','cached_raw_feature_recomputation':'EXACT'}
  checks.append(item)
  for g in [role]+(['STRONG27'] if sid in strongids else []):groupfeatures[g].append(item)
 trainvalues=np.concatenate([features[r['sample_id']] for r in fit]).astype(np.float64);m=trainvalues.mean(0).astype(np.float32);s=trainvalues.std(0);s=np.where(s<1e-6,1.,s).astype(np.float32)
 assert np.array_equal(m,mean) and np.array_equal(s,std);del trainvalues
 save('feature_and_target_checks.json',checks)
 save('feature_summary.json',{g:{'utterances':len(rr),**{k:summary([r[k] for r in rr]) for k in ['observations','target_tokens','CTC_slack','duration_ms','median_dt_ms','observed_any_hand_fraction','both_hands_fraction','body_available_fraction','normalization_clip_fraction']},
  'zero_hand_motion_sequences':sum(r['zero_hand_motion_sequence'] for r in rr),'events':dict(eventcounts[g])} for g,rr in groupfeatures.items()})
 save('undersampled_event_examples.json',examples)
 save('duplicate_features.json',{'exact_X_duplicates':[v for v in content.values() if len(v)>1],'raw_hash_duplicates_across_train_held':sorted({r['raw_sha256'] for r in fit}&{r['raw_sha256'] for r in held})})
 # Recompute token edit distance independently; do not invoke the model or its evaluator.
 scores={};prediction_tests=0
 for p in sorted(list(PILOT.glob('selected_*.json'))+list((PILOT/'predictions').glob('*.json'))):
  score=read(p);tot=0;refs=0;hyp=Counter()
  for q in score['predictions']:
   assert q['reference']==lookup[q['sample_id']]['glosses'];assert distance(q['reference'],q['hypothesis'])==q['substitutions']+q['deletions']+q['insertions']
   tot+=q['substitutions']+q['deletions']+q['insertions'];refs+=len(q['reference']);hyp.update(q['hypothesis']);prediction_tests+=1
  assert math.isclose(tot/refs,score['gloss_WER'],abs_tol=1e-12)
  assert hyp==Counter(score['hypothesis_token_counts']);scores[p.name]=score
 metrics=list(csv.DictReader((PILOT/'METRICS.csv').open()));evals=[r for r in metrics if r['primary80_WER']];best=min(evals,key=lambda r:(float(r['primary80_WER']),float(r['primary80_CTC'])));assert int(best['epoch'])==12
 credible=1.;stale=0;trace=[]
 for r in evals:
  wer=float(r['primary80_WER']);gain=credible-wer;reset=gain>=.005-1e-12
  if reset:credible=wer;stale=0
  else:stale+=1
  trace.append({'epoch':int(r['epoch']),'WER':wer,'edit_errors':round(wer*485),'gain_from_credible_anchor':gain,'credible_reset':reset,'stale_evaluations':stale})
 assert trace[-1]['epoch']==14 and trace[-1]['stale_evaluations']==4
 save('stopping_reconciliation.json',{'best_epoch':int(best['epoch']),'trace':trace,'primary_reference_tokens':scores['selected_primary80.json']['reference_tokens'],
  'epoch6_to12_improvement_percentage_points':100*(float(next(r for r in evals if r['epoch']=='6')['primary80_WER'])-float(best['primary80_WER'])),
  'threshold_required_edit_improvement':math.ceil(.005*scores['selected_primary80.json']['reference_tokens'])})
 supports=Counter(t for r in fit for t in set(r['glosses']));occ=Counter(t for r in fit for t in r['glosses']);assert vocab==['<BLANK>','<UNK>']+sorted(supports)
 assert all(set(lookup[i]['glosses'])<=set(supports) for i in primaryids)
 assert all(all(supports[t]>=5 for t in lookup[i]['glosses']) for i in strongids)
 save('target_support.json',{'unique_glosses':len(supports),'tokens':sum(occ.values()),'singleton_distinct_utterance_labels':sum(v==1 for v in supports.values()),'support_lt5_labels':sum(v<5 for v in supports.values()),
  'top20_reference_tokens':occ.most_common(20),'special_tokens':['<BLANK>','<UNK>'],'whitespace_or_NUL_labels':[t for t in supports if t!=t.strip() or '\x00' in t]})
 # Trusted own checkpoints only, deserialized onto CPU for metadata/tensor inspection.
 # No CausalCTC instance, training process, optimizer instance, forward/backward or CUDA calls.
 import torch
 cpresults=[];init=None
 for filename in ['initialization.pt','epoch_012.pt','epoch_014.pt']:
  path=PILOT/'checkpoints'/filename;state=torch.load(path,map_location='cpu',weights_only=False);assert all(t.device.type=='cpu' for t in state['model'].values())
  assert state['vocabulary']==vocab and state['config_sha256']==sha(PILOT/'config.json') and state['fold_manifest_sha256']==status['fold_manifest_sha256']
  assert all(bool(torch.isfinite(v).all()) for v in state['model'].values())
  assert np.array_equal(state['model']['mean'].numpy(),mean) and np.array_equal(state['model']['std'].numpy(),std)
  if init is None:init={k:v.clone() for k,v in state['model'].items()}
  delta=math.sqrt(sum(float(((v-init[k]).double()**2).sum()) for k,v in state['model'].items() if k not in ['mean','std']))
  os=state['optimizer'];steps=[int(v['step'].item()) for v in os['state'].values() if 'step' in v]
  cpresults.append({'file':filename,'sha256':sha(path),'epoch':state['epoch'],'model_parameter_count_excluding_norm_buffers':sum(v.numel() for k,v in state['model'].items() if k not in ['mean','std']),
   'state_shapes':{k:list(v.shape) for k,v in state['model'].items()},'parameter_L2_change_from_initialization':delta,'optimizer_step_min':min(steps) if steps else 0,'optimizer_step_max':max(steps) if steps else 0,
   'optimizer_groups':[{k:g[k] for k in ['lr','betas','eps','weight_decay','amsgrad']} for g in os['param_groups']],'all_model_tensors_finite':True})
  del state
 save('checkpoint_metadata_CPU_only.json',cpresults)
 save('audit_summary.json',{'status':'BOUNDED_READ_ONLY_CHECKS_PASS','feature_sequences':len(rows),'TRAIN_feature_transcript_mappings_verified':len(fit),'cached_landmark_feature_recomputations':len(rows),
  'prediction_examples_independently_checked':prediction_tests,'files_with_scores_checked':len(scores),'normalization_TRAIN_recompute_exact':True,
  'input_files_hashed_before':len(before),'input_bytes_hashed_before':sum(v['bytes'] for v in before.values()),'selected_epoch':12,'stop_epoch':14,
  'tail_accumulation_issue':{'microbatches_per_epoch':375,'optimizer_steps_per_epoch':188,'last_microbatch_examples':4,'actual_last_scale':.5,'intended_last_scale_if_averaging_available_microbatches':1.,'updates_with_half_scaled_tail_pct':100/188,'effective_gradient_weight_shortfall_pct':100*2/1500,'primary_cause_proven':False},
  'elapsed_seconds':time.perf_counter()-start,'model_forward_calls':0,'backward_calls':0,'optimizer_steps_executed':0,'GPU_calls':0,'TEST_data_opened':False})
 print(json.dumps({'status':'READ_ONLY_ANALYSIS_COMPLETE','seconds':time.perf_counter()-start,'feature_sequences':len(rows),'prediction_examples_checked':prediction_tests,'inputs_hashed':len(before),'feature_events':{g:dict(c) for g,c in eventcounts.items()}}))

if __name__=='__main__':main()
