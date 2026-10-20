"""Cached TRAIN-only interval/mask review; no model/perception/media execution."""
from pathlib import Path
from collections import Counter
import csv,ctypes,hashlib,importlib.metadata,json,math,os,re,subprocess,time
import numpy as np
RUN=Path(__file__).resolve().parent;BASE=RUN.parent;ROOT=RUN.parents[2]
AUDIT=BASE/'20261020T135119Z_read_only_failure_audit01';PILOT=BASE/'20261020T133148Z_source_transfer_pilot01'
START=time.perf_counter();READS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def stat(p):
 p=Path(p);s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
def track(p,old=None):
 p=Path(p)
 if str(p) not in READS:
  v=stat(p)
  if old is not None:assert v['sha256']==old['sha256'] and v['bytes']==old['bytes'],str(p)
  READS[str(p)]=v
 return p
def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def save(n,v):
 with (RUN/n).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def csvsave(n,rr):
 with (RUN/n).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def family(r):
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
 return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def stable(e):return hashlib.sha256(('apphear-alignment-review-v1|'+e['event_key']).encode()).hexdigest()
# Explicit prior non-TEST allowlist. No workspace discovery.
for n in ['READ_ONLY_FAILURE_AUDIT.md','commands_and_results.txt']:track(AUDIT/n)
prior=read(AUDIT/'input_files_before.json');assert prior==read(AUDIT/'input_files_after.json')
for rel in ['pipeline/perception_vnext/segmentation/learned_features.py','pipeline/perception_vnext/recognition_schema.py','experiments/continuous_asl/20261020_continuous_expand06/tools/prepare_asllrp.py']:
 p=ROOT/rel;track(p,prior[str(p)])
oldpilot=read(AUDIT/'pilot_files_before.json')
for e in oldpilot:track(e['path'],e)
pilot_paths={str(Path(e['path'])) for e in oldpilot}
audit_paths={str(p) for p in AUDIT.iterdir() if p.is_file()}
for p in audit_paths:track(p)
rows=read(PILOT/'resolved_manifest.json');fold=read(PILOT/'fold_manifest.json')
assert sha(PILOT/'fold_manifest.json')=='87442a6f1241860e369a5c8f0768d583b3a4058f5a9ad03dcc135532c419c72f'
fit=[r for r in rows if r['internal_role']=='TRAIN']
assert len(fit)==1500 and {r['sample_id'] for r in fit}==set(fold['train_sample_ids'])
up=track(fold['universe_manifest'],prior[fold['universe_manifest']]);assert sha(up)==fold['universe_manifest_sha256']
original={r['sample_id']:r for r in read(up) if r['split']=='train'};assert len(original)==1706
assert set(fold['train_sample_ids']).isdisjoint(fold['internal_eval_sample_ids'])
assert set(fold['train_capture_families']).isdisjoint(fold['eval_capture_families'])
assert all(r['split']=='train' and r['glosses']==original[r['sample_id']]['glosses'] for r in fit)
cp=track(fit[0]['annotation_provenance']['csv_path'],prior[fit[0]['annotation_provenance']['csv_path']])
assert sha(cp)==fit[0]['annotation_provenance']['csv_sha256']
needed={e['source_row'] for r in fit for e in r['events']}
with cp.open(encoding='utf-8-sig',newline='') as f:ann={n:r for n,r in enumerate(csv.DictReader(f),2) if n in needed}
support=Counter(t for r in fit for t in set(r['glosses']))
cache={};events=[];errors=[];array_start=time.perf_counter()
for r in sorted(fit,key=lambda r:r['sample_id']):
 sid=r['sample_id'];rp=Path(r['feature_path'].replace('.features.npz','.raw.npz'));track(rp,prior[str(rp)])
 with np.load(rp,allow_pickle=False) as z:
  fi=z['frame_index'].copy();ts=z['timestamp_ms'].copy();ho=z['hand_observed'].copy();ba=z['body_available'].copy()
  bv=z['body_valid'].copy();jv=z['hand_joint_valid'].copy();meta=json.loads(str(z['metadata_json'].item()))
 assert fi.dtype==np.int32 and ho.shape==(len(fi),2) and ho.dtype==bool
 assert np.all(np.diff(fi)>0) and np.all(np.diff(ts)>0)
 assert meta['sample_id']==sid and meta['glosses']==r['glosses']
 assert np.array_equal(fi,np.arange(0,r['decoded_frames'],3,dtype=np.int32))
 assert r['decoded_frames']==r['utterance_end_frame_master']-r['utterance_start_frame_master']+1
 fps=r['native_fps'];assert abs(fps-30000/1001)<1e-8
 err=float(np.max(np.abs(ts-fi*1000/fps)));assert err<1e-5;errors.append(err)
 ah=ho.any(1)
 cache[sid]=dict(fi=fi,ts=ts,ho=ho,ba=ba,bv=bv,jv=jv,rawpath=rp,row=r)
 assert [e['label'] for e in r['events']]==r['glosses']
 for k,e in enumerate(r['events']):
  c=ann[e['source_row']]
  for field,val in [('Occurrence label',e['label']),('Video ID number',e['event_id']),('Utterance video filename',r['utterance_filename']),('Master video filename',r['source_group']),('Source collection',r['collection']),('Utterance number',r['utterance_id'])]:assert c[field]==val,(sid,field)
  for field,val in [('Start frame of the sign video',e['start_frame_master']),('End frame of the sign video',e['end_frame_master']),('Start frame of the containing utterance',r['utterance_start_frame_master']),('End frame of the containing utterance',r['utterance_end_frame_master'])]:assert int(c[field])==val,(sid,field)
  a=e['start_frame_master']-r['utterance_start_frame_master'];b=e['end_frame_master']-r['utterance_start_frame_master'];assert 0<=a<=b<r['decoded_frames']
  ix=np.flatnonzero((fi>=a)&(fi<=b));n=len(ix)
  A=n==0;B=n>0 and not ah[ix].any();C=n>0 and bool(ah[ix].all());assert sum((A,B,C))<=1
  group='A' if A else 'B' if B else 'C' if C else 'MIXED'
  events.append(dict(event_key=f"{sid}:{e['source_row']}:{e['event_id']}",sample_id=sid,event_index=k,group=group,label=e['label'],event_id=e['event_id'],source_row=e['source_row'],hidden=e['hidden'],a=a,b=b,duration_frames=b-a+1,duration_ms=(b-a+1)*1000/fps,support=support[e['label']],source_group=r['source_group'],family=family(r),signer_id=r['signer_id'],T=len(fi),U=len(r['glosses']),ix=ix.tolist(),n=n,hand=float(ah[ix].mean()) if n else None,both=float(ho[ix].all(1).mean()) if n else None,body=float(ba[ix].mean()) if n else None))
counts=Counter(e['group'] for e in events);assert len(events)==12423 and counts['A']==880 and counts['B']==1440
array_seconds=time.perf_counter()-array_start
anchors=[];families=set();sources=set();ids=set()
for _ in range(4):
 e=min((e for e in events if e['group']=='A' and e['sample_id'] not in ids),key=lambda e:(e['family'] in families,e['source_group'] in sources,stable(e)))
 anchors.append(e);families.add(e['family']);sources.add(e['source_group']);ids.add(e['sample_id'])
selected=[];matches=[]
for i,a in enumerate(anchors,1):
 selected.append(dict(a,matched_set=i,anchor=a['event_key'],cost=0.))
 for g in ['B','C']:
  def cost(e):return sum(abs(math.log2(e[k]/a[k])) for k in ['duration_frames','support','T'])
  e=min((e for e in events if e['group']==g and e['sample_id'] not in ids),key=lambda e:(e['family']!=a['family'],cost(e),e['source_group'] in sources,stable(e)))
  selected.append(dict(e,matched_set=i,anchor=a['event_key'],cost=cost(e)));ids.add(e['sample_id']);sources.add(e['source_group'])
  matches.append(dict(anchor=a['event_key'],matched=e['event_key'],group=g,same_family=e['family']==a['family'],cost=cost(e),duration_ratio=e['duration_frames']/a['duration_frames'],support_ratio=e['support']/a['support'],observation_length_ratio=e['T']/a['T']))
assert len(ids)==12 and Counter(e['group'] for e in selected)==dict(A=4,B=4,C=4)
findings=[];review=[]
for order,e in enumerate(selected,1):
 q=cache[e['sample_id']];r=q['row'];fi=q['fi'];ts=q['ts'];ho=q['ho'];ba=q['ba'];a=e['a'];b=e['b']
 ii=[i for i,v in enumerate(fi.tolist()) if a<=v<=b];assert ii==e['ix']
 obs=[bool(ho[i,0] or ho[i,1]) for i in ii];assert (not ii if e['group']=='A' else not any(obs) if e['group']=='B' else all(obs))
 fp=track(r['feature_path'],prior[r['feature_path']]);vp=track(r['video_path'],prior[r['video_path']]);track(r['signer_provenance']['source_page'],prior[r['signer_provenance']['source_page']])
 assert sha(fp)==r['feature_sha256'] and sha(vp)==r['raw_sha256']
 with np.load(fp,allow_pickle=False) as z:
  x=z['X'];fm=json.loads(str(z['metadata_json'].item()))
  assert x.shape==(len(fi),378) and np.array_equal(z['timestamp_ms'],ts)
  assert fm['sample_id']==r['sample_id'] and fm['glosses']==r['glosses'] and fm['utterance_start_frame_master']==r['utterance_start_frame_master']
  assert np.array_equal(x[:,172:174].astype(bool),ho) and np.array_equal(x[:,214].astype(bool),ba)
  assert np.array_equal(x[:,84:126].reshape(-1,2,21).astype(bool),q['jv'] & ho[:,:,None]) and np.array_equal(x[:,206:214].astype(bool),q['bv'])
 pre=np.flatnonzero(fi<a)[-2:].tolist();post=np.flatnonzero(fi>b)[:2].tolist();ctx=pre+post
 def detail(ix):return [dict(frame_index=int(fi[i]),timestamp_ms=float(ts[i]),hand_observed=ho[i].tolist(),raw_hand_valid_joint_counts=q['jv'][i].sum(1).tolist(),effective_hand_valid_joint_counts=(q['jv'][i] & ho[i,:,None]).sum(1).tolist(),body_available=bool(ba[i]),body_valid_joint_count=int(q['bv'][i].sum())) for i in ix]
 p=int(fi[pre[-1]]) if pre else None;n=int(fi[post[0]]) if post else None
 start=a*1000/r['native_fps'];end=b*1000/r['native_fps']
 f=dict(review_id=f"R{order:02d}",matched_set=e['matched_set'],group=e['group'],sample_id=e['sample_id'],event_key=e['event_key'],literal_target=e['label'],publisher_event_id=e['event_id'],publisher_csv_row=e['source_row'],signer_id=e['signer_id'],source_group=e['source_group'],recording_family=e['family'],target_support_distinct_train_utterances=e['support'],sequence_observations=e['T'],sequence_target_count=e['U'],publisher_start_frame=a+r['utterance_start_frame_master'],publisher_end_frame=b+r['utterance_start_frame_master'],utterance_master_origin=r['utterance_start_frame_master'],relative_start_frame=a,relative_end_frame=b,native_fps=r['native_fps'],inclusive_duration_ms=e['duration_ms'],event_start_ms=start,event_last_frame_ms=end,event_end_exclusive_ms=(b+1)*1000/r['native_fps'],median_cadence_ms=float(np.median(np.diff(ts))),duration_to_cadence_ratio=e['duration_ms']/float(np.median(np.diff(ts))),retained_count=e['n'],inside_observations=detail(ii),any_hand_fraction=e['hand'],both_hand_fraction=e['both'],body_available_fraction=e['body'],context_observations=detail(ctx),context_any_hand_fraction=float(ho[ctx].any(1).mean()) if ctx else None,context_body_available_fraction=float(ba[ctx].mean()) if ctx else None,preceding_retained_frame=p,following_retained_frame=n,enclosing_retained_gap_ms=(n-p)*1000/r['native_fps'] if p is not None and n is not None else None,publisher_hidden_flag=e['hidden'],coordinate_check='PASS',literal_join_check='PASS',feature_mask_check='PASS',computational_group_check='PASS',qualified_review_status='BLOCKED_NO_QUALIFIED_REVIEWER',visual_flags=[],semantic_judgment='UNRESOLVED',matching_cost=e['cost'])
 findings.append(f)
 review.append(dict(review_id=f['review_id'],matched_set=e['matched_set'],group=e['group'],sample_id=e['sample_id'],event_key=e['event_key'],literal_publisher_target=e['label'],annotation_type=r['annotation_type'],publisher_csv_path=str(cp),publisher_csv_sha256=READS[str(cp)]['sha256'],publisher_csv_row=e['source_row'],publisher_event_id=e['event_id'],original_video_path=str(vp),original_video_sha256=READS[str(vp)]['sha256'],raw_landmarks_path=str(q['rawpath']),raw_landmarks_sha256=READS[str(q['rawpath'])]['sha256'],feature_path=str(fp),feature_sha256=READS[str(fp)]['sha256'],signer_id=e['signer_id'],source_group=e['source_group'],recording_family=e['family'],publisher_start_frame=f['publisher_start_frame'],publisher_end_frame=f['publisher_end_frame'],utterance_master_origin=f['utterance_master_origin'],relative_start_frame=a,relative_end_frame=b,native_fps=r['native_fps'],start_seconds=start/1000,end_exclusive_seconds=f['event_end_exclusive_ms']/1000,context_start_seconds=max(0,start/1000-1),context_end_seconds=min(r['decoded_frames']/r['native_fps'],f['event_end_exclusive_ms']/1000+1),coordinate_convention='native clip zero-based; publisher master minus inclusive utterance start; endpoints inclusive',source_orientation='UPRIGHT_UNMIRRORED_SOURCE_RGB',review_status='NOT_REVIEWED_NO_QUALIFIED_REVIEWER',reviewer_id='',reviewer_qualification='',reviewed_at='',visual_flags='',visible_articulators='',both_hands_where_required='',neighboring_context='',boundary_rationale='',target_semantics_rationale='',evidence_vs_inference=''))
groups=[]
for g in ['A','B','C']:
 rr=[f for f in findings if f['group']==g];nn=sum(f['retained_count'] for f in rr)
 groups.append(dict(group=g,candidate_events=counts[g],selected_events=4,mean_duration_ms=float(np.mean([f['inclusive_duration_ms'] for f in rr])),min_duration_ms=min(f['inclusive_duration_ms'] for f in rr),max_duration_ms=max(f['inclusive_duration_ms'] for f in rr),mean_target_support=float(np.mean([f['target_support_distinct_train_utterances'] for f in rr])),retained_frames_total=nn,any_hand_frame_fraction=sum(sum(any(o['hand_observed']) for o in f['inside_observations']) for f in rr)/nn if nn else None,body_available_frame_fraction=sum(sum(o['body_available'] for o in f['inside_observations']) for f in rr)/nn if nn else None,mean_context_any_hand_fraction=float(np.mean([f['context_any_hand_fraction'] for f in rr])),mean_context_body_fraction=float(np.mean([f['context_body_available_fraction'] for f in rr])),recording_families=len({f['recording_family'] for f in rr}),source_masters=len({f['source_group'] for f in rr}),computational_discrepancies=0,qualified_reviews=0,semantic_adjudication='BLOCKED'))
csvsave('event_level_findings.csv',findings);csvsave('review_manifest.csv',review);csvsave('group_summary.csv',groups)
save('selection.json',dict(version='apphear-alignment-review-v1',candidate_counts=dict(counts),universe_train_utterances=1500,train_event_count=len(events),full_candidate_event_digest=hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),selected_event_keys=[e['event_key'] for e in selected],matches=matches,group_C_definition='at least one retained frame; every retained frame has at least one observed hand; neither both hands nor linguistic observability guaranteed',algorithm='A: smallest stable SHA after unused family/unused master preference. B,C: same family first, then sum absolute log2 ratios of duration frames, distinct-utterance support and observation length; unused master tie preference then SHA. Hash prefix apphear-alignment-review-v1|. Twelve unique utterances.',candidate_exclusions='Only frozen internal TRAIN; MIXED excluded from A/B/C; no label/hidden-flag filtering; predictions not parsed.',limitations='Hash anchors are not representative. Approximate matching; family preference may worsen balance. C is a cached-mask control.'))
save('input_files_before.json',READS)
hardware=json.loads((RUN/'hardware_capability.json').read_text())
hardware.update(cpu_cached_array_review_seconds=array_seconds,cached_array_ram_bytes=sum(sum(v.nbytes for v in q.values() if isinstance(v,np.ndarray)) for q in cache.values()),independent_cpu_membership_checks_passed=12)
save('hardware_and_timing.json',hardware)
after={p:stat(p) for p in READS};changed=[p for p in READS if READS[p]!=after[p]];assert not changed
assert {str(p) for p in PILOT.rglob('*') if p.is_file()}==pilot_paths
assert {str(p) for p in AUDIT.iterdir() if p.is_file()}==audit_paths
save('input_files_after.json',after)
integ=dict(status='PASS',existing_files_checked=len(READS),existing_bytes_checked=sum(e['bytes'] for e in READS.values()),changed_files=changed,pilot_files_checked=len(pilot_paths),prior_audit_files_checked=len(audit_paths),pilot_additions=[],pilot_removals=[],prior_audit_additions=[],prior_audit_removals=[],train_raw_arrays_checked=1500,selected_feature_video_signer_provenance_files_checked=36,prior_explicit_non_test_hashes_match=True,train_membership='PASS',group_disjointness='PASS',publisher_joins_checked=12423,selected_independent_scalar_checks_passed=12,max_timestamp_coordinate_error_ms=max(errors),no_model_created=True,no_model_inference=True,no_training=True,no_perception_execution=True,no_video_decoding=True,reserved_test_access=False,reserved_test_enumeration=False,new_writes_directory=str(RUN),production_asset_writes=False,limitations='Explicit scoped before/after bytes/mtime/hash and pilot/audit membership checks. Production untouched by this program but not independently hashed; whole workspace not enumerated. Shared train_dev/annotation metadata read for TRAIN membership/rows; no reserved TEST paths/payloads enumerated. Cannot certify historical actions outside process.',elapsed_analysis_and_hardware_seconds=time.perf_counter()-START)
save('integrity_results.json',integ)
print(json.dumps(dict(output_directory=str(RUN),counts=dict(counts),groups=groups,hardware=hardware,integrity=integ),indent=2))

