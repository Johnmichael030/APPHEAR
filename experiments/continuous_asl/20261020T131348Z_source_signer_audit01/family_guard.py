"""Conservative provenance-family guard; never infer a person or split undated series."""
from audit import *
import re

def main():
 allrows=read(PARENT/'data/processed/train_dev/manifest.json');rows=[r for r in allrows if r['split']=='train'];lookup={r['sample_id']:r for r in rows}
 def family(r):
  m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
  return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
 groups=defaultdict(list)
 for r in rows:groups[family(r)].append(r)
 save('family_guard_protocol.json',{'reason':'Publisher master filenames are source IDs, not proof of independent capture lineage. Numbered Ben introduction-to-conclusion series may share a parent; no authoritative independence map is present.',
  'guard':'Keep all masters with same known signer and date-shaped master prefix together; keep all undated masters for a signer together. This is conservative overgrouping, not a claim of acquisition dates or shared physical recording.',
  'assignment':'Same source-ID SHA256 rule at family level; only signers with>=2families and family>=20utterances eligible.',
  'rejected_shortcut':'Do not split adjacent/unknown-parent series merely because filenames differ. Preliminary filename folds retained as evidence, not fit-ready.',
  'no_WER_used_for_assignment':True,'training_authorized':False})
 table('SOURCE_FAMILY_COVERAGE.csv',[{'family_id':k,'signer_id':rr[0]['signer_id'],'utterances':len(rr),'published_master_sources':len({r['source_group'] for r in rr}),
  'master_sources':json.dumps(sorted({r['source_group'] for r in rr})),'status':'CONSERVATIVE_DATE_PREFIX_GROUP' if 'UNRESOLVED' not in k else 'UNRESOLVED_PARENT_SERIES_KEEP_TOGETHER'} for k,rr in sorted(groups.items())])
 csvpath=RUN/'CANDIDATE_FOLDS.csv';old=list(csv.DictReader(csvpath.open(encoding='utf-8-sig')));table('PRELIMINARY_FILENAME_FOLDS.csv',old)
 for r in old:
  if r['kind']=='blocked':r.update(capture_family_overlap='UNKNOWN_NO_ASSIGNMENT',eval_capture_families='',TRAIN_capture_families='',compatible_eval_token_coverage='');continue
  f=read(RUN/'folds'/f"{r['fold']}.json");tr=[lookup[i] for i in f['train_sample_ids']];ev=[lookup[i] for i in f['internal_eval_sample_ids']]
  overlap=sorted({family(a) for a in tr}&{family(a) for a in ev});r.update(capture_family_overlap=bool(overlap),eval_capture_families=len({family(a) for a in ev}),TRAIN_capture_families=len({family(a) for a in tr}),compatible_eval_token_coverage='')
  if overlap:r['status']='PARTIAL_FILENAME_ONLY_GUARD_FAILED';r['causal_scope']+=' Conservative capture-family guard fails; not fit-ready.'
 held={}
 for signer in sorted({r['signer_id'] for r in rows}):
  keys=[k for k,rr in groups.items() if rr[0]['signer_id']==signer and len(rr)>=20]
  if len(keys)>=2:held[signer]=sorted(keys,key=lambda k:hashlib.sha256(k.encode()).hexdigest())[0]
 added=[];coverage=[];viewexclusions=[];details={}
 def fold(name,kind,tr,ev):
  trainfamilies={family(r) for r in tr};evalfamilies={family(r) for r in ev};assert not trainfamilies&evalfamilies
  for key in ['sample_id','source_group','collection','raw_sha256','decoded_pixel_sha256']:assert not ({r.get(key) for r in tr if r.get(key)}&{r.get(key) for r in ev if r.get(key)})
  assert {r['signer_id'] for r in ev}<={r['signer_id'] for r in tr}
  ts=support(tr);ec=tok(ev);tc=tok(tr);es=support(ev);compatible=[r for r in ev if set(r['glosses'])<=set(ts)];strong=[r for r in ev if all(ts[t]>=5 for t in r['glosses'])]
  role={'fold':name,'kind':kind,'universe_manifest':str(PARENT/'data/processed/train_dev/manifest.json'),'universe_manifest_sha256':sha(PARENT/'data/processed/train_dev/manifest.json'),
   'train_sample_ids':sorted(r['sample_id'] for r in tr),'internal_eval_sample_ids':sorted(r['sample_id'] for r in ev),'compatible_eval_sample_ids':sorted(r['sample_id'] for r in compatible),'support5_eval_sample_ids':sorted(r['sample_id'] for r in strong),
   'train_capture_families':sorted(trainfamilies),'eval_capture_families':sorted(evalfamilies),'family_rule':'family_guard_protocol.json','training_authorized':False}
  save('folds/'+name+'.json',role)
  cp_tokens=sum(len(r['glosses']) for r in compatible)
  d={'fold':name,'kind':kind,'status':'PASS_CONSERVATIVE_GROUP_GUARD; PARTIAL_EXPLORATORY_SUPPORT','causal_scope':'Same known signers; held-out conservative recording families. Capture/date/topic/label composition still bundled; no isolated signer or dataset effect.',
   'TRAIN_utterances':len(tr),'internal_eval_utterances':len(ev),'TRAIN_signers':len({r['signer_id'] for r in tr}),'eval_signers':len({r['signer_id'] for r in ev}),'shared_signer_IDs':len({r['signer_id'] for r in tr}&{r['signer_id'] for r in ev}),
   'TRAIN_sources':len({r['source_group'] for r in tr}),'eval_sources':len({r['source_group'] for r in ev}),'TRAIN_labels':len(ts),'eval_labels':len(es),'shared_literal_labels':len(set(ts)&set(es)),
   'eval_reference_tokens':sum(ec.values()),'eval_token_coverage':sum(c for t,c in ec.items() if t in ts)/sum(ec.values()),'compatible_eval_utterances':len(compatible),'support5_eval_utterances':len(strong),
   'minimum_CTC_slack':min(r['observations']-len(r['glosses'])-sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:])) for r in tr+ev),'known_recording_content_leakage':False,
   'manifest_sha256':sha(RUN/'folds'/f'{name}.json'),'capture_family_overlap':False,'eval_capture_families':len(evalfamilies),'TRAIN_capture_families':len(trainfamilies),'compatible_eval_token_coverage':cp_tokens/sum(ec.values())}
  added.append(d);details[name]={'summary':d,'eval_count_by_signer':dict(Counter(r['signer_id'] for r in ev)),'compatible_eval_count_by_signer':dict(Counter(r['signer_id'] for r in compatible)),
   'support5_eval_count_by_signer':dict(Counter(r['signer_id'] for r in strong)),'eval_families':sorted(evalfamilies),'train_families':sorted(trainfamilies),
   'compatible_eval_tokens':cp_tokens,'full_eval_length_summary':{'min':min(len(r['glosses']) for r in ev),'max':max(len(r['glosses']) for r in ev)}}
  for b in ['0','1','2-4','5-9','10-19','20+']:
   labels={t for t in set(ts)|set(es) if bucket(ts[t])==b};coverage.append({'fold':name,'fold_TRAIN_distinct_support_bucket':b,'TRAIN_labels':sum(t in ts for t in labels),'TRAIN_tokens':sum(tc[t] for t in labels),
    'eval_labels':sum(t in es for t in labels),'eval_tokens':sum(ec[t] for t in labels),'eval_token_fraction':sum(ec[t] for t in labels)/sum(ec.values()),'control':'literal labels; support counted by distinct foldTRAIN utterances'})
  for view,subset in [('compatible',compatible),('support5',strong)]:
   good={r['sample_id'] for r in subset}
   for r in ev:
    if r['sample_id'] not in good:viewexclusions.append({'fold':name,'view':view,'sample_id':r['sample_id'],'reason':'WHOLE_SEQUENCE_OUTSIDE_VIEW; NOT_REMOVED_FROM_FULL_EVAL','labels':json.dumps(sorted(t for t in set(r['glosses']) if ts[t]<(1 if view=='compatible' else 5)),ensure_ascii=False)})
  return d
 heldids={r['sample_id'] for k in held.values() for r in groups[k]}
 pooled=fold('A_guarded_pooled_family_holdout','source_family_pooled',[r for r in rows if r['sample_id'] not in heldids],[r for r in rows if r['sample_id'] in heldids])
 for signer,key in sorted(held.items()):
  sub=[r for r in rows if r['signer_id']==signer]
  fold('A_guarded_within_signer_'+signer.rsplit('_',1)[1],'source_family_within_signer',[r for r in sub if family(r)!=key],[r for r in sub if family(r)==key])
 old.extend(added)
 old.append({k:('A_guarded_within_signer_5' if k=='fold' else 'blocked' if k=='kind' else 'BLOCKED_SINGLE_UNRESOLVED_PARENT_FAMILY' if k=='status' else 'All335Ben records kept together; no safe same-person capture-family holdout.' if k=='causal_scope' else '') for k in old[0]})
 with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)
 for filename,newrows in [('LABEL_COVERAGE_BY_FOLD.csv',coverage),('EVAL_VIEW_EXCLUSIONS.csv',viewexclusions)]:
  with (RUN/filename).open('a',encoding='utf-8-sig',newline='') as f:
   csv.DictWriter(f,fieldnames=list(newrows[0])).writerows(newrows)
 save('guarded_fold_details.json',details)
 save('guarded_decision.json',{'recommendation':'SOURCE_TRANSFER_PILOT_CONSERVATIVE_FAMILY_LEVEL','candidate':pooled,'training_authorized':False,
  'supersedes':'decision.json recommendation and filename-only A folds; their completed metadata artifacts are retained.',
  'reason':'Keep uncertain related masters together. Two eval family clusters across two known seen signers; exploratory only. Cory compatible support is six samples, so no stable Cory-specific accuracy estimate.',
  'held_family_by_signer':held,'independent_source_signer_effects':'BLOCKED; no crossed cells. This pilot only asks about new recording-family transfer among known signers.',
  'signer5_limitation':'No independence evidence for ten undated numbered masters; keep whole335-record family together.',
  'source_identity_limit':'Date-shaped publisher master prefixes are conservative grouping proxies, not proven capture dates. Unknown cross-date re-encoding remains a metadata limitation.'})
 # Fill a recoverable global TRAIN blank statistic from an already completed replay, no inference.
 recovered=read(DIAG/'replayed_train.json')
 path=RUN/'SAVED_MODEL_PARTITION_SUMMARY.csv';summary=list(csv.DictReader(path.open(encoding='utf-8-sig')))
 for r in summary:
  if r['model']=='prior1706' and r['split']=='TRAIN':r['blank_argmax_fraction']=recovered['blank_argmax_fraction']
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 save('supplement_input_hashes.json',INPUTS)
 print(json.dumps({'status':'CONSERVATIVE_GUARD_COMPLETE_NO_TRAINING','recommended':pooled,'held_families':held}),flush=True)

if __name__=='__main__':main()
