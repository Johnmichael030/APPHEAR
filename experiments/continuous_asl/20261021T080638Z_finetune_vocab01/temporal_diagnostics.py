"""Cached publisher-time observability and CTC-run diagnostics; no physical timing claim."""
from collections import Counter
import numpy as np
import fine_tune as ft
from post_evaluate import align

v=ft.read(ft.RUN/'vocabulary.json')['tokens']
meta={r['sample_id']:r for r in ft.read(ft.RUN/'baseline_temporal_metadata.json')}
refs={r['sample_id']:r for r in ft.read(ft.RUN/'manifests/DEV80.json')}
observations=[];runs=[];joins=[]
for sid,r in refs.items():
 ts=np.array(meta[sid]['timestamps_ms']);intervals=meta[sid]['publisher_occurrence_intervals'];assert len(intervals)==len(r['glosses'])
 for i,((start,end),token) in enumerate(zip(intervals,r['glosses'])):
  observations.append(dict(sample_id=sid,reference_index=i,gloss=token,publisher_start_ms=start,publisher_end_ms=end,feature_observations_in_inclusive_interval=int(((ts>=start)&(ts<=end)).sum()),duration_ms=end-start,status='PUBLISHER_INTERVAL_UNREVIEWED_PHYSICAL_BOUNDARY'))
for model,path,predpath in [('baseline','baseline_logits_DEV80.npz','baseline_greedy_DEV80.json'),('A','A/selected_logits_DEV80.npz','A/selected_DEV80.json')]:
 pp={p['sample_id']:p for p in ft.read(ft.RUN/predpath)['predictions']}
 with np.load(ft.RUN/path,allow_pickle=False) as cache:
  for sid in cache.files:
   ts=np.array(meta[sid]['timestamps_ms']);z=cache[sid];ids=z.argmax(-1);assert len(ids)==len(ts)
   local=[];k=0
   while k<len(ids):
    e=k+1
    while e<len(ids) and ids[e]==ids[k]:e+=1
    if ids[k]!=0:
     row=dict(model=model,sample_id=sid,hypothesis_index=len(local),gloss=v[int(ids[k])],first_argmax_ms=float(ts[k]),last_argmax_ms=float(ts[e-1]),argmax_frames=e-k,next_observation_ms=float(ts[e]) if e<len(ts) else '',status='MODEL_ARGMAX_RUN_NOT_PHYSICAL_SIGN_BOUNDARY_OR_PUBLICATION_LATENCY')
     runs.append(row);local.append(row)
    k=e
   assert [r['gloss'] for r in local]==pp[sid]['hypothesis']
   for op,i,j,ref,hyp in align(pp[sid]['reference'],pp[sid]['hypothesis']):
    if op!='MATCH':continue
    start,end=meta[sid]['publisher_occurrence_intervals'][i];run=local[j]
    joins.append(dict(model=model,sample_id=sid,reference_index=i,hypothesis_index=j,gloss=ref,publisher_start_ms=start,publisher_end_ms=end,first_argmax_ms=run['first_argmax_ms'],last_argmax_ms=run['last_argmax_ms'],intervals_overlap=run['last_argmax_ms']>=start and run['first_argmax_ms']<=end,status='SEQUENCE_ALIGNMENT_JOIN_ONLY_NOT_REVIEWED_TEMPORAL_ALIGNMENT'))
ft.writecsv(ft.RUN/'PUBLISHER_INTERVAL_OBSERVABILITY.csv',observations)
ft.writecsv(ft.RUN/'CTC_ARGMAX_RUNS.csv',runs)
ft.writecsv(ft.RUN/'MATCHED_TOKEN_TIME_JOINS.csv',joins)
counts=Counter(r['feature_observations_in_inclusive_interval'] for r in observations)
ft.save(ft.RUN/'TEMPORAL_DIAGNOSTICS.json',dict(status='PARTIAL_DIAGNOSTIC_ONLY',DEV_publisher_occurrences=len(observations),zero_sampled_observations=counts[0],one_sampled_observation=counts[1],observation_count_histogram=dict(counts),qualified_physical_boundary_review=False,no_sign_minutes_available=0,adjacent_repeated_label_DEV_sequences=0,live_latency_measured=False,scope='Cached approximately10Hz publisher interval joins. CTC peaks are not forced alignments or authoritative physical sign boundaries. No threshold tuned.'))
print('PASS: saved publisher interval sampling and cached CTC runs; boundary accuracy/no-sign/live latency remain unmeasured.')
