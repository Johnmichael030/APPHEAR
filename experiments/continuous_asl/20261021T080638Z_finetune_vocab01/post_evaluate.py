"""Post-fit analysis of saved predictions/logits only. No fitting and no TEST."""
from collections import Counter,defaultdict
from pathlib import Path
import json,time
import numpy as np
import torch
import fine_tune as ft

def align(ref,hyp):
 n,m=len(ref),len(hyp);d=np.zeros((n+1,m+1),dtype=int);d[:,0]=np.arange(n+1);d[0,:]=np.arange(m+1)
 for i in range(1,n+1):
  for j in range(1,m+1):d[i,j]=min(d[i-1,j-1]+(ref[i-1]!=hyp[j-1]),d[i-1,j]+1,d[i,j-1]+1)
 out=[];i,j=n,m
 while i or j:
  if i and j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):out.append(('MATCH' if ref[i-1]==hyp[j-1] else 'SUBSTITUTION',i-1,j-1,ref[i-1],hyp[j-1]));i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:out.append(('DELETION',i-1,None,ref[i-1],''));i-=1
  else:out.append(('INSERTION',None,j-1,'',hyp[j-1]));j-=1
 return list(reversed(out))

def bucket(s):return '1' if s==1 else '2-4' if s<5 else '5-9' if s<10 else '10-19' if s<20 else '20+'

def main():
 torch.set_num_threads(2)
 train=ft.read(ft.RUN/'manifests/TRAIN1500.json');dev=ft.read(ft.RUN/'manifests/DEV80.json');v=ft.read(ft.RUN/'vocabulary.json')['tokens'];meta={r['sample_id']:r for r in dev}
 support=Counter(t for r in train for t in set(r['glosses']));freq=Counter(t for r in train for t in r['glosses'])
 sources={('baseline','greedy'):ft.RUN/'baseline_greedy_DEV80.json',('A','greedy'):ft.RUN/'A/selected_DEV80.json',('baseline','streaming'):ft.RUN/'baseline_streaming_DEV80.json',('A','streaming'):ft.RUN/'A/selected_streaming_DEV80.json'}
 predictions={k:ft.read(p) for k,p in sources.items()};aligned=[];sequences=[];classes=[];bucketrows=[];groups=[];collapse=[]
 for (model,protocol),evaluation in predictions.items():
  per=defaultdict(Counter);bk=defaultdict(Counter);group=defaultdict(Counter);hypcounts=Counter();correct=0
  for p in evaluation['predictions']:
   r=meta[p['sample_id']];hypcounts.update(p['hypothesis']);a=align(p['reference'],p['hypothesis']);counts=Counter(op for op,*_ in a)
   assert [counts[t] for t in ['SUBSTITUTION','DELETION','INSERTION']]==[p[k] for k in ['substitutions','deletions','insertions']]
   for op,i,j,ref,hyp in a:
    aligned.append(dict(model=model,protocol=protocol,sample_id=p['sample_id'],operation=op,reference_index=i,hypothesis_index=j,reference_gloss=ref,hypothesis_gloss=hyp,TRAIN_distinct_utterance_support=support[ref] if ref else ''))
    if ref:
     per[ref]['reference']+=1;bk[bucket(support[ref])]['reference']+=1
    if hyp:per[hyp]['predicted']+=1
    if op=='MATCH':per[ref]['TP']+=1;bk[bucket(support[ref])]['correct']+=1;correct+=1
    elif ref:bk[bucket(support[ref])][op]+=1
   sequences.append(dict(model=model,protocol=protocol,sample_id=p['sample_id'],reference=json.dumps(p['reference'],ensure_ascii=False),hypothesis=json.dumps(p['hypothesis'],ensure_ascii=False),substitutions=p['substitutions'],deletions=p['deletions'],insertions=p['insertions'],reference_tokens=p['reference_tokens'],WER=(p['substitutions']+p['deletions']+p['insertions'])/p['reference_tokens'],signer=r['signer_id'],source=r['source_group'],family=ft.family(r),min_label_support=min(support[t] for t in p['reference']),mean_blank_probability=p.get('mean_blank_probability',''),mean_max_probability=p.get('mean_max_probability','')))
   for kind,key in [('signer',r['signer_id']),('source',r['source_group']),('family',ft.family(r))]:
    c=group[(kind,key)];c.update({k:p[k] for k in ['substitutions','deletions','insertions','reference_tokens']});c['utterances']+=1;c['exact']+=p['reference']==p['hypothesis']
  for token in v[2:]:
   c=per[token];classes.append(dict(model=model,protocol=protocol,gloss=token,TRAIN_distinct_utterance_support=support[token],TRAIN_token_occurrences=freq[token],DEV_reference=c['reference'],DEV_predicted=c['predicted'],aligned_true_positive=c['TP'],precision=(c['TP']/c['predicted'] if c['predicted'] else ''),recall=(c['TP']/c['reference'] if c['reference'] else ''),never_predicted=c['predicted']==0))
  for b,c in bk.items():bucketrows.append(dict(model=model,protocol=protocol,support_bucket=b,reference_tokens=c['reference'],correct=c['correct'],recall=c['correct']/c['reference'],substitutions=c['SUBSTITUTION'],deletions=c['DELETION']))
  for (kind,key),c in group.items():groups.append(dict(model=model,protocol=protocol,group_kind=kind,group=key,**dict(c),WER=(c['substitutions']+c['deletions']+c['insertions'])/c['reference_tokens']))
  if protocol=='greedy':
   cache=ft.RUN/('baseline_logits_DEV80.npz' if model=='baseline' else 'A/selected_logits_DEV80.npz')
   raw=Counter();framecount=0
   with np.load(cache,allow_pickle=False) as z:
    for name in z.files:ids=z[name].argmax(-1);raw.update(map(int,ids));framecount+=len(ids)
   common=hypcounts.most_common(1)[0]
   collapse.append(dict(model=model,protocol=protocol,raw_blank_fraction=raw[0]/framecount,raw_UNK_fraction=raw[1]/framecount,raw_frames=framecount,collapsed_hypothesis_tokens=sum(hypcounts.values()),hypothesis_diversity=len(hypcounts),most_common_hypothesis=common[0],most_common_count=common[1],most_common_fraction=common[1]/sum(hypcounts.values()),empty_sequences=evaluation['empty_outputs'],correct_reference_tokens=correct,CTC_loss=evaluation['CTC_loss']))
 for name,rows in [('ALIGNMENTS.csv',aligned),('PREDICTION_EXAMPLES.csv',sequences),('PER_CLASS_METRICS.csv',classes),('SUPPORT_BUCKET_METRICS.csv',bucketrows),('GROUP_METRICS.csv',groups),('TOKEN_COLLAPSE_AUDIT.csv',collapse)]:ft.writecsv(ft.RUN/name,rows)
 base=predictions[('baseline','greedy')]['predictions'];candidate=predictions[('A','greedy')]['predictions']
 assert [p['sample_id'] for p in base]==[p['sample_id'] for p in candidate]
 rng=np.random.default_rng(20261021);n=np.array([p['reference_tokens'] for p in base]);e=lambda pp:np.array([p['substitutions']+p['deletions']+p['insertions'] for p in pp]);delta=e(candidate)-e(base)
 draws=rng.integers(0,80,size=(5000,80));differences=delta[draws].sum(1)/n[draws].sum(1)
 uncertainty=dict(seed=20261021,paired_utterance_bootstrap_draws=5000,WER_delta_candidate_minus_baseline=float(delta.sum()/n.sum()),descriptive_percentile95_interval=list(map(float,np.quantile(differences,[.025,.975]))),limitation='Development-exposed utterances correlated within only two recording families. Conditional descriptive interval, not a population or unseen-signer confidence claim.',improved_sequences=int((delta<0).sum()),worsened_sequences=int((delta>0).sum()),same_edit_count_sequences=int((delta==0).sum()))
 ft.save(ft.RUN/'PAIRED_UNCERTAINTY.json',uncertainty)
 pointer=ft.read(ft.RUN/'A/RESULTS.json')['selected_checkpoint'];s=torch.load(ft.SOURCE,map_location='cpu',weights_only=False);c=torch.load(ft.RUN/pointer['path'],map_location='cpu',weights_only=False)
 assert s['vocabulary']==c['vocabulary']==v
 changes={k:float((s['model'][k]-c['model'][k]).abs().max()) for k in s['model']}
 assert changes['mean']==changes['std']==0 and changes['output.weight']>0
 ft.save(ft.RUN/'A/PARAMETER_AUDIT.json',dict(status='PASS',vocabulary_unchanged=True,normalization_exact=True,max_absolute_change_by_tensor=changes,source_optimizer_present='optimizer' in s,selected_optimizer_present='optimizer' in c,source_output_dimensions=list(s['model']['output.weight'].shape),selected_output_dimensions=list(c['model']['output.weight'].shape),selected_sha256=ft.sha(ft.RUN/pointer['path'])))
 model=ft.CausalCTC(len(v),48,c['model']['mean'].numpy(),c['model']['std'].numpy());model.load_state_dict(c['model']);model.eval();latency=[]
 with torch.inference_mode():
  for length in [1,17,61]:
   x=torch.randn(1,length,378,generator=torch.Generator().manual_seed(length));durations=[]
   for i in range(35):
    start=time.perf_counter();model(x);elapsed=(time.perf_counter()-start)*1000
    if i>=5:durations.append(elapsed)
   latency.append(dict(length=length,host_CPU_model_only_p50_ms=float(np.median(durations)),host_CPU_model_only_p95_ms=float(np.quantile(durations,.95)),measured_repeats=30,warmups=5,threads=2,input='SYNTHETIC_ENGINEERING_ONLY',scope='Windows CPU model-forward only; not Android, camera-to-text, or MediaPipe latency'))
 ft.writecsv(ft.RUN/'HOST_INFERENCE_LATENCY.csv',latency)
 gpu=list(__import__('csv').DictReader((ft.RUN/'A/gpu_memory.csv').open()));device_peak=max(float(r['gpu_memory_used_MiB']) for r in gpu if r['gpu_memory_used_MiB']!='UNAVAILABLE')
 repeated=[r for r in dev if any(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))]
 stats=dict(status='PASS_COMPUTATIONAL_PARTIAL_RECOGNITION',TRAIN_utterances=len(train),TRAIN_tokens=sum(freq.values()),literal_labels=len(v)-2,singleton_labels=sum(x==1 for x in support.values()),labels_under5=sum(x<5 for x in support.values()),DEV_utterances=len(dev),DEV_reference_tokens=int(n.sum()),DEV_distinct_labels=len({t for r in dev for t in r['glosses']}),TRAIN_signers=len({r['signer_id'] for r in train}),DEV_signers=len({r['signer_id'] for r in dev}),TRAIN_source_keys=len({r['source_group'] for r in train}),DEV_source_keys=len({r['source_group'] for r in dev}),TRAIN_families=len({ft.family(r) for r in train}),DEV_families=len({ft.family(r) for r in dev}),adjacent_repeated_label_DEV_sequences=len(repeated),same_label_anywhere_DEV_sequences=sum(len(set(r['glosses']))<len(r['glosses']) for r in dev),CTC_feasibility='All1580validated; no temporal stride/downsampling',sampled_nvidia_smi_device_peak_MiB=device_peak,device_memory_note='WDDM nvidia-smi sampled device-reported usage differs from allocator accounting; do not substitute it for measured PyTorch peak.',selected_checkpoint_bytes=(ft.RUN/pointer['path']).stat().st_size,model_tensor_bytes=sum(t.numel()*t.element_size() for t in c['model'].values()),APK_asset_impact_bytes=0,new_classes=0,TEST_used=False)
 ft.save(ft.RUN/'POST_EVALUATION.json',stats)
 rows=[]
 for (label,protocol),p in predictions.items():rows.append(dict(experiment=label,protocol=protocol,status='MEASURED',TRAIN_WER=(ft.read(ft.RUN/'A/RESULTS.json')['TRAIN_WER'] if label=='A' else ft.read(ft.BASE/'selected_train1500.json')['gloss_wer'] if 'gloss_wer' in ft.read(ft.BASE/'selected_train1500.json') else ft.read(ft.BASE/'selected_train1500.json').get('WER','')),DEV_WER=p['WER'],substitutions=p['substitutions'],deletions=p['deletions'],insertions=p['insertions'],exact_sequences=p['exact_sequences'],new_classes=0,scope='Known signers; recording-family transfer; development-exposed80'))
 for p in ft.read(ft.RUN/'decoder/RESULTS.json')['results']:
  if p['decoder']=='greedy':continue
  rows.append(dict(experiment=p['model'],protocol='full_sequence_'+p['decoder'],status='MEASURED_CACHED_LOGITS',TRAIN_WER='',DEV_WER=p['WER'],substitutions=p['substitutions'],deletions=p['deletions'],insertions=p['insertions'],exact_sequences=p['exact_sequences'],new_classes=0,scope='CPU prefix beam, noLM; not current streaming decoder'))
 for label,status in [('B','BLOCKED_NEW_CLASS_ADMISSION'),('C','NOT_RUN_PREDECLARED_GATE_FAILED')]:rows.append(dict(experiment=label,protocol='',status=status,TRAIN_WER='',DEV_WER='',substitutions='',deletions='',insertions='',exact_sequences='',new_classes=0,scope='No fit performed'))
 ft.writecsv(ft.RUN/'EXPERIMENT_RESULTS.csv',rows)
 print(json.dumps(dict(stats=stats,uncertainty=uncertainty,latency=latency),indent=2))

if __name__=='__main__':main()
