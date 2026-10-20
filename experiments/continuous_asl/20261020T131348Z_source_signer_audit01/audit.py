"""TRAIN-only source/signer design. No fitting, model execution, or acquisition."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,time
import numpy as np
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];TREE=RUN.parent
PARENT=TREE/'20261020T104054Z_expand_eval01';PREV=TREE/'20261020T124813Z_label_support01';DIAG=TREE/'20261020T123237Z_collapse_diagnosis01'
INPUTS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):
 p=Path(p);INPUTS[str(p)]=sha(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,x):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.write('\n')
def table(name,rows):
 assert rows
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def support(rows):return Counter(t for r in rows for t in set(r['glosses']))
def tok(rows):return Counter(t for r in rows for t in r['glosses'])
def bucket(n):return '0' if n==0 else '1' if n==1 else '2-4' if n<5 else '5-9' if n<10 else '10-19' if n<20 else '20+'
def edit_summary(preds):
 c=Counter()
 for p in preds:
  for k in ['substitutions','deletions','insertions','reference_tokens']:c[k]+=p[k]
 return {'WER':(c['substitutions']+c['deletions']+c['insertions'])/c['reference_tokens'] if c['reference_tokens'] else None,**dict(c)}

def main():
 started=time.perf_counter()
 save('protocol.json',{'scope':'TRAIN-only identity/fold design. Existing frozen DEV predictions may be summarized afterward, never used for fold assignment. No TEST features, labels or predictions.',
  'source_levels':['publisher dataset','parent recording','utterance media file'],'crossing_rule':'Need observed shared signers across sources AND multi-signer sources within a compatible label pool; no inference from names/images.',
  'source_holdout_rule':'Per known ASLLRP TRAIN signer, one complete recording with>=20utterances, smallest SHA256(recording ID). No score or label-coverage optimization.',
  'folds':['pooled same-signers source holdout','within-signer source holdout3','leave-one-signer-out3 (joint source+signer effect)','crossed only if actual incidence supports it'],
  'label_controls':'Full internal evaluation plus whole utterances with all labels seen in foldTRAIN; additional >=5foldTRAIN-utterance-support view. Report every dropped ID/reason. Never remove target tokens.',
  'recommendation_screen':'Pooled source pilot only if>=1000TRAIN,>=100internalEval,all3signers retained,>=3held-outrecordings,>=30compatibleEval and>=80%fullEvaltoken coverage; exploratory, not power analysis.',
  'training_authorized':False})
 all_asl=read(PARENT/'data/processed/train_dev/manifest.json');asl=[r for r in all_asl if r['split']=='train'];assert len(asl)==1706
 catalog=read(PARENT/'eligible_corpus/manifest.json');ncatalog=[r for r in catalog if r['training_pool'].startswith('NCSLGR')]
 oldpath=TREE/'20261020_continuous_expand09/data/processed/train_dev/manifest.json';old=[r for r in read(oldpath) if r['split']=='train']
 newpath=PARENT/'timed_preparation/manifest.json';new=[r for r in read(newpath) if r['split']=='train']
 membership={r['sample_id']:str(oldpath) for r in old};membership.update({r['sample_id']:str(newpath) for r in new})
 assert len(old)==906 and len(new)==96 and len(membership)==1002
 assert set(membership)=={r['sample_id'] for r in ncatalog}
 canonical={r['sample_id']:r for r in catalog};metadata=read(PARENT/'signer_metadata.json')['joined_eligible_records']
 for r in asl:
  assert r['signer_id']==metadata[r['sample_id']]['signer_id'] and r['source_group'] and r['signer_provenance']['method'].startswith('Official public')
 rows=asl+ncatalog;assert len({r['sample_id'] for r in rows})==2708
 priorchecks=read(DIAG/'critical_checks.json');assert priorchecks['feature_records_checked']==3121 and priorchecks['CTC_length_violations']==0
 inventory=[];rawgroups=defaultdict(set);pixelgroups=defaultdict(list);sourceurls=defaultdict(set);sourcecollections=defaultdict(set);sourcenames=defaultdict(set)
 for r in rows:
  pool=r['training_pool'];dataset=r['dataset'];group=r.get('source_group') or '';key=pool+'::'+group;signer=r.get('signer_id') or ''
  minimum=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]));assert r['observations']>=minimum
  assert Path(r['feature_path']).is_file() and r['feature_sha256']
  if pool.startswith('ASLLRP'):
   origin=str(PARENT/'data/processed/train_dev/manifest.json');split='train';source_status='VERIFIED_PUBLISHER_PARENT_RECORDING';identity='PUBLISHER_PARTICIPANT_QUERY_JOIN'
  else:
   origin=membership[r['sample_id']];split='train';source_status='EXACT_EAF_FULL_SOURCE_MEDIA' if (r.get('annotation_provenance') or {}).get('member','').endswith('.eaf') else 'INDEX_MEDIA_ID; UNDERLYING_RECORDING_GRANULARITY_UNCERTAIN'
   identity='EXPLICIT_EXISTING_SIGNER_ID' if signer else 'MISSING; NO_IDENTITY_INFERENCE'
  sourceurls[key].add(r.get('source_url') or r.get('source_group') or '');sourcecollections[key].add(r.get('collection') or '');sourcenames[group].add(pool)
  if r.get('raw_sha256'):rawgroups[r['raw_sha256']].add(key)
  if r.get('decoded_pixel_sha256'):pixelgroups[r['decoded_pixel_sha256']].append(r['sample_id'])
  inventory.append({'sample_id':r['sample_id'],'source_dataset':dataset,'source_recording_id':group,'recording_id_scope':pool,'source_identity_status':source_status,
   'signer_id':signer,'signer_identity_status':identity,'label_pool':pool,'label_schema':'literal dominant realized occurrence' if pool.startswith('ASLLRP') else 'literal main gloss',
   'ordered_glosses':json.dumps(r['glosses'],ensure_ascii=False),'gloss_tokens':len(r['glosses']),'observations':r['observations'],'minimum_CTC_steps':minimum,'CTC_slack':r['observations']-minimum,
   'feature_path':r['feature_path'],'feature_sha256':r['feature_sha256'],'feature_validation':'REUSED_PRIOR3121HASH_SCHEMA_FINITE_TIMESTAMP_CTC_PASS; EXISTS_NOW',
   'applicable_frozen_manifest':origin,'applicable_split':split,'catalog_split_annotation':r.get('split',''),'membership_provenance':'Explicit TRAIN membership in cited manifest; catalog membership alone not used',
   'collection':r.get('collection',''),'raw_sha256':r.get('raw_sha256',''),'decoded_pixel_sha256':r.get('decoded_pixel_sha256',''),
   'annotation_provenance':json.dumps(r.get('annotation_provenance',{}),ensure_ascii=False),'signer_provenance':json.dumps(r.get('signer_provenance',{}),ensure_ascii=False)})
 table('SOURCE_SIGNER_COVERAGE.csv',inventory)
 signers=sorted({r['signer_id'] for r in rows if r.get('signer_id')});cross=defaultdict(Counter);datasetcross=defaultdict(Counter);signerpool=Counter();sourcepool=Counter()
 for r in rows:
  s=r.get('signer_id') or 'UNKNOWN_NOT_A_PERSON';cross[(r['dataset'],r['training_pool'],r['source_group'])][s]+=1;datasetcross[r['dataset']][s]+=1;signerpool[(s,r['training_pool'])]+=1;sourcepool[(r['source_group'],r['training_pool'])]+=1
 crossrows=[]
 for (dataset,pool,group),c in sorted(cross.items()):crossrows.append({'source_dataset':dataset,'label_pool':pool,'source_recording_id':group,**{s:c[s] for s in signers},'unknown_signer_records':c['UNKNOWN_NOT_A_PERSON'],'known_signer_IDs_in_source':sum(c[s]>0 for s in signers),'utterances':sum(c.values())})
 table('SOURCE_SIGNER_CROSS_TAB.csv',crossrows)
 table('DATASET_SIGNER_CROSS_TAB.csv',[{'source_dataset':d,**{s:c[s] for s in signers},'unknown_signer_records':c['UNKNOWN_NOT_A_PERSON']} for d,c in sorted(datasetcross.items())])
 table('SIGNER_BY_LABEL_POOL.csv',[{'signer_id':s,'label_pool':p,'utterances':c,'identity_scope':'UNKNOWN bucket is missing metadata, not a shared person' if s=='UNKNOWN_NOT_A_PERSON' else 'Corpus-scoped ID; no cross-corpus identity equivalence established'} for (s,p),c in sorted(signerpool.items())])
 table('SOURCE_BY_LABEL_POOL.csv',[{'source_recording_id':s,'label_pool':p,'utterances':c} for (s,p),c in sorted(sourcepool.items())])
 # Exact source aliases/content joins are used only as conservative leakage guards.
 aliases=[{'kind':'SAME_RAW_HASH_MULTIPLE_SOURCE_KEYS','identity':h,'members':sorted(v)} for h,v in rawgroups.items() if len(v)>1]
 aliases += [{'kind':'SAME_PIXEL_HASH_MULTIPLE_UTTERANCES','identity':h,'members':v} for h,v in pixelgroups.items() if len(v)>1]
 asources=sorted({r['source_group'] for r in asl});asigners=sorted({r['signer_id'] for r in asl});cells=Counter((r['source_group'],r['signer_id']) for r in asl)
 known_per_source={g:sorted({s for (source,s),n in cells.items() if source==g}) for g in asources};assert all(len(v)==1 for v in known_per_source.values())
 X=np.array([[1]+[int(r['source_group']==g) for g in asources[1:]] for r in asl],dtype=float)
 Z=np.array([[int(r['signer_id']==s) for s in asigners[1:]] for r in asl],dtype=float)
 rank_source=int(np.linalg.matrix_rank(X));rank_joint=int(np.linalg.matrix_rank(np.column_stack([X,Z])))
 matrix=np.array([[cells[g,s] for s in asigners] for g in asources],dtype=float);expected=matrix.sum(1,keepdims=True)@matrix.sum(0,keepdims=True)/matrix.sum();chi=float(((matrix-expected)**2/expected).sum());cramer=math.sqrt(chi/(matrix.sum()*min(matrix.shape[0]-1,matrix.shape[1]-1)))
 incidence={'ASLLRP_TRAIN_utterances':len(asl),'source_recordings':len(asources),'publisher_signer_IDs':len(asigners),'sources_with_multiple_signers':sum(len(v)>1 for v in known_per_source.values()),
  'sources_per_signer':{s:len({r['source_group'] for r in asl if r['signer_id']==s}) for s in asigners},'Cramers_V_source_vs_signer':cramer,
  'source_indicator_rank':rank_source,'source_plus_signer_indicator_rank':rank_joint,'signer_additional_rank':rank_joint-rank_source,
  'conclusion':'Signer indicators perfectly nested/aliased with recording indicators. No fully crossed or independent additive recording+signer effect identification.',
  'cross_dataset_signer_bridge':'UNKNOWN; no verified same-person link across the two pool namespaces','pool_target_equivalence':'NOT_PROVEN; keep separate'}
 save('identifiability.json',incidence)
 save('identity_issues.json',{'missing_signer_records':sum(not r.get('signer_id') for r in rows),'missing_source_group_records':sum(not r.get('source_group') for r in rows),
  'multi_source_raw_or_pixel_aliases':aliases,'source_key_multiple_urls':[{'source':k,'values':sorted(v)} for k,v in sourceurls.items() if len(v)>1],
  'source_key_multiple_collections':[{'source':k,'values':sorted(v)} for k,v in sourcecollections.items() if len(v)>1],
  'unscoped_group_name_collisions':[{'name':k,'pools':sorted(v)} for k,v in sourcenames.items() if len(v)>1],
  'source_status_counts':dict(Counter(r['source_identity_status'] for r in inventory)),
  'repeated_views':'No identity inferred from camera filenames or faces. Exact hashes and parent source/collection groups checked; unresolved NCS parent/view links prohibit NCS crossed folds.',
  'limits':'One raw hash reused for nonoverlapping utterances in the same full recording is expected, not a duplicate example. Exact hashes cannot detect every re-encoding or unpublished alternate view.'})
 # Freeze fold assignments before inspecting group WER. Assignment uses source IDs only.
 held={}
 for signer in asigners:
  eligible=[g for g in asources if known_per_source[g]==[signer] and sum(cells[g,s] for s in asigners)>=20]
  held[signer]=sorted(eligible,key=lambda g:hashlib.sha256(g.encode()).hexdigest())[0]
 candidates=[];coverage=[];exclusions=[];fold_details={}
 def fold(name,kind,tr,ev,causal_status):
  ts=support(tr);es=support(ev);tc=tok(tr);ec=tok(ev);vocab=set(ts);compatible=[r for r in ev if set(r['glosses'])<=vocab];strong=[r for r in ev if all(ts[t]>=5 for t in r['glosses'])]
  overlaps={k:sorted({r.get(k) for r in tr if r.get(k)}&{r.get(k) for r in ev if r.get(k)}) for k in ['sample_id','source_group','collection','raw_sha256','decoded_pixel_sha256']}
  assert not any(overlaps.values()),('FOLD_LEAKAGE',name,overlaps)
  same_signers=sorted({r['signer_id'] for r in tr}&{r['signer_id'] for r in ev})
  if kind.startswith('source'):assert {r['signer_id'] for r in ev}<={r['signer_id'] for r in tr}
  if kind=='signer_joint_holdout':assert not same_signers
  for key in ['raw_sha256','decoded_pixel_sha256']:
   assert not ({r.get(key) for r in tr if r.get(key)}&{r.get(key) for r in ev if r.get(key)})
  compatible_ids={r['sample_id'] for r in compatible};strong_ids={r['sample_id'] for r in strong}
  for r in ev:
   if r['sample_id'] not in compatible_ids:exclusions.append({'fold':name,'view':'compatible','sample_id':r['sample_id'],'reason':'WHOLE_REFERENCE_HAS_FOLD_TRAIN_UNSEEN_LABEL','labels':json.dumps(sorted(set(r['glosses'])-vocab),ensure_ascii=False)})
   if r['sample_id'] not in strong_ids:exclusions.append({'fold':name,'view':'support5','sample_id':r['sample_id'],'reason':'WHOLE_REFERENCE_HAS_LABEL_WITH_FOLD_TRAIN_SUPPORT_LT5','labels':json.dumps(sorted(t for t in set(r['glosses']) if ts[t]<5),ensure_ascii=False)})
  role={'fold':name,'kind':kind,'universe_manifest':str(PARENT/'data/processed/train_dev/manifest.json'),'universe_manifest_sha256':INPUTS[str(PARENT/'data/processed/train_dev/manifest.json')],
   'train_sample_ids':sorted(r['sample_id'] for r in tr),'internal_eval_sample_ids':sorted(r['sample_id'] for r in ev),'compatible_eval_sample_ids':sorted(compatible_ids),'support5_eval_sample_ids':sorted(strong_ids),
   'evaluation_policy':'Full internalEval is primary coverage report; compatible/support5 views auxiliary, no target trimming; frozen externalDEV/TEST not used.', 'causal_scope':causal_status,'training_authorized':False}
  save('folds/'+name+'.json',role)
  details={'fold':name,'kind':kind,'status':'PASS_INTERNAL_GROUP_ASSIGNMENT','causal_scope':causal_status,'TRAIN_utterances':len(tr),'internal_eval_utterances':len(ev),
   'TRAIN_signers':len({r['signer_id'] for r in tr}),'eval_signers':len({r['signer_id'] for r in ev}),'shared_signer_IDs':len(same_signers),'TRAIN_sources':len({r['source_group'] for r in tr}),'eval_sources':len({r['source_group'] for r in ev}),
   'TRAIN_labels':len(ts),'eval_labels':len(es),'shared_literal_labels':len(set(ts)&set(es)),'eval_reference_tokens':sum(ec.values()),'eval_token_coverage':sum(c for t,c in ec.items() if t in ts)/sum(ec.values()),
   'compatible_eval_utterances':len(compatible),'support5_eval_utterances':len(strong),'minimum_CTC_slack':min(r['observations']-len(r['glosses'])-sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:])) for r in tr+ev),
   'known_recording_content_leakage':False,'manifest_sha256':sha(RUN/'folds'/f'{name}.json')}
  candidates.append(details);fold_details[name]={'train_signers':sorted({r['signer_id'] for r in tr}),'eval_signers':sorted({r['signer_id'] for r in ev}),'train_sources':sorted({r['source_group'] for r in tr}),'eval_sources':sorted({r['source_group'] for r in ev}),
   'overlap':overlaps,'compatible_eval_by_signer':dict(Counter(r['signer_id'] for r in compatible)),'support5_eval_by_signer':dict(Counter(r['signer_id'] for r in strong))}
  for b in ['0','1','2-4','5-9','10-19','20+']:
   labels={t for t in set(ts)|set(es) if bucket(ts[t])==b}
   coverage.append({'fold':name,'fold_TRAIN_distinct_support_bucket':b,'TRAIN_labels':sum(t in ts for t in labels),'TRAIN_tokens':sum(tc[t] for t in labels),'eval_labels':sum(t in es for t in labels),'eval_tokens':sum(ec[t] for t in labels),'eval_token_fraction':sum(ec[t] for t in labels)/sum(ec.values()),'control':'literal labels; support counted by distinct foldTRAIN utterances'})
  return details
 heldset=set(held.values());pooled=fold('A_pooled_recording_holdout','source_pooled',[r for r in asl if r['source_group'] not in heldset],[r for r in asl if r['source_group'] in heldset],'Same three known signers in fit and evaluation; unseen recordings. Recording/topic/session and label-composition effects remain bundled; not a dataset effect.')
 for signer in asigners:
  short=signer.rsplit('_',1)[1];sub=[r for r in asl if r['signer_id']==signer]
  fold('A_within_signer_'+short,'source_within_one_signer',[r for r in sub if r['source_group']!=held[signer]],[r for r in sub if r['source_group']==held[signer]],'Signer fixed; one held-out recording. Smaller label support; one eval cluster prevents broad inference.')
  fold('B_signer_'+short,'signer_joint_holdout',[r for r in asl if r['signer_id']!=signer],[r for r in asl if r['signer_id']==signer],'Signer and all its recordings jointly held out. Not an isolated signer effect because no recording is shared across signers.')
 for name,why in [('C_crossed','BLOCKED: every known ASLLRP recording contains only one signer; no crossed cells.'),('NCS_crossed','BLOCKED:988missing signer IDs, mixed recording granularity, no verified cross-pool signer mapping or target equivalence.')]:
  candidates.append({k:(name if k=='fold' else 'blocked' if k=='kind' else 'BLOCKED_INSUFFICIENT_IDENTIFIABILITY' if k=='status' else why if k=='causal_scope' else '') for k in pooled})
 table('CANDIDATE_FOLDS.csv',candidates);table('LABEL_COVERAGE_BY_FOLD.csv',coverage);table('EVAL_VIEW_EXCLUSIONS.csv',exclusions);save('fold_details.json',fold_details)
 valid=pooled['TRAIN_utterances']>=1000 and pooled['internal_eval_utterances']>=100 and pooled['TRAIN_signers']==pooled['eval_signers']==3 and pooled['eval_sources']>=3 and pooled['compatible_eval_utterances']>=30 and pooled['eval_token_coverage']>=.8
 save('decision.json',{'recommendation':'SOURCE_TRANSFER_PILOT_RECORDING_LEVEL_ONLY' if valid else 'BLOCKED_INSUFFICIENT_SUPPORT','training_authorized':False,'candidate':pooled,
  'held_recording_by_signer':held,'identifiability':'PARTIAL: source-recording transfer with signer identities held constant is testable. Independent source-dataset vs signer causal effects and crossed design are BLOCKED.',
  'must_initialize':'Fresh recognition model/optimizer: all records were TRAIN in older recognizers; existing recognition checkpoints leak into internal heldout.',
  'elapsed_seconds_before_group_summary':time.perf_counter()-started})
 # Performance summary is descriptive only and occurs AFTER metadata-only fold freeze.
 groups=[];partitions=[]
 models=[('prior1706',PARENT/'pilot_fresh48_v2/best_checkpoint_full_train.json',PARENT/'pilot_fresh48_v2/best_checkpoint_dev.json',asl),
  ('prior64',DIAG/'pilot64/best_train.json',DIAG/'pilot64/best_dev.json',[r for r in read(DIAG/'pilot64/manifest.json') if r['split']=='train']),
  ('prior497',PREV/'pilot497/best_train.json',PREV/'pilot497/best_original_dev.json',[r for r in read(PREV/'frozen_manifest.json') if r['split']=='train'])]
 # FrozenDEV metadata is used only here to join previously saved DEV predictions, never for folds.
 devlookup={r['sample_id']:r for r in all_asl if r['split']=='dev'}
 for model,tp,dp,fitrows in models:
  ts=support(fitrows);fitlookup={r['sample_id']:r for r in fitrows}
  for split,path,lookup in [('TRAIN',tp,fitlookup),('FROZEN_DEV_REPORT_ONLY',dp,devlookup)]:
   score=read(path);preds=score['predictions'];assert {p['sample_id'] for p in preds}==set(lookup)
   partitions.append({'model':model,'split':split,'utterances':len(preds),'WER':score['gloss_wer'],'CTC_loss':score['loss'],'blank_argmax_fraction':score.get('blank_argmax_fraction','UNKNOWN_NOT_SAVED'),
    'empty_outputs':sum(not p['hypothesis'] for p in preds),'predicted_token_diversity':len({t for p in preds for t in p['hypothesis']}),'interpretation':'TRAIN resubstitution is not transfer; frozenDEV descriptive only'})
   for axis in ['signer_id','source_group']:
    by=defaultdict(list)
    for p in preds:by[lookup[p['sample_id']][axis]].append(p)
    for key,ps in sorted(by.items()):
     hyp=Counter(t for p in ps for t in p['hypothesis']);ref=Counter(t for p in ps for t in p['reference']);top,n=hyp.most_common(1)[0] if hyp else ('',0)
     groups.append({'model':model,'split':split,'group_axis':axis,'group_id':key,'utterances':len(ps),**edit_summary(ps),'empty_outputs':sum(not p['hypothesis'] for p in ps),
      'prediction_tokens':sum(hyp.values()),'predicted_token_diversity':len(hyp),'top_prediction':top,'top_prediction_fraction':n/sum(hyp.values()) if hyp else None,
      'reference_tokens_unseen_in_actual_fit':sum(c for t,c in ref.items() if ts[t]==0),'reference_tokens_support_lt5':sum(c for t,c in ref.items() if ts[t]<5),
      'blank_fraction':'UNKNOWN_NO_SAVED_PER_GROUP_LOGITS','interpretation':'Descriptive conditional error, not causal or new transfer evaluation'})
 table('SAVED_MODEL_GROUP_ERRORS.csv',groups);table('SAVED_MODEL_PARTITION_SUMMARY.csv',partitions)
 save('input_hashes.json',INPUTS)
 print(json.dumps({'status':'AUDIT_COMPLETE_NO_TRAINING','TRAIN_inventory':len(rows),'identifiability':incidence,'recommended_fold':pooled,'elapsed_seconds':time.perf_counter()-started}),flush=True)

if __name__=='__main__':main()
