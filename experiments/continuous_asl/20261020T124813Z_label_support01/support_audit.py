"""Distinct-utterance support and frozen, whole-sequence candidate audit."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,sys,time
import numpy as np
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];TREE=RUN.parent
PARENT=TREE/'20261020T104054Z_expand_eval01';DIAG=TREE/'20261020T123237Z_collapse_diagnosis01'
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
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bucket(n):return '0_UNSEEN' if n==0 else '1' if n==1 else '2-4' if n<5 else '5-9' if n<10 else '10-19' if n<20 else '20+'
def support(rows):return Counter(t for r in rows for t in set(r['glosses']))
def tokens(rows):return Counter(t for r in rows for t in r['glosses'])
def distribution(values):
 a=np.asarray(values);return {'min':float(a.min()),'p25':float(np.quantile(a,.25)),'median':float(np.median(a)),'p75':float(np.quantile(a,.75)),'p95':float(np.quantile(a,.95)),'max':float(a.max()),'mean':float(a.mean())}
def align(ref,hyp):
 n,m=len(ref),len(hyp);d=np.zeros((n+1,m+1),dtype=np.int32);d[:,0]=range(n+1);d[0,:]=range(m+1)
 for i in range(1,n+1):
  for j in range(1,m+1):d[i,j]=min(d[i-1,j-1]+(ref[i-1]!=hyp[j-1]),d[i-1,j]+1,d[i,j-1]+1)
 out=[];i=n;j=m
 while i or j:
  if i and j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):out.append((ref[i-1],'M' if ref[i-1]==hyp[j-1] else 'S'));i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:out.append((ref[i-1],'D'));i-=1
  else:out.append((hyp[j-1],'I'));j-=1
 return out[::-1]

def main():
 start=time.perf_counter()
 save('audit_protocol.json',{'thresholds':[5,10],'support_unit':'distinct TRAIN utterances, full original TRAIN universe',
  'retention':'One pass: retain complete utterance iff every literal gloss meets original TRAIN support threshold; no target deletion or remapping.',
  'additional_check':'Recompute retained-set support; report recursive fixed-point failure, not silently claim threshold holds after filtering.',
  'candidate_rule':'Prefer5 over10 by predeclared coverage principle if >=200TRAIN, all3TRAINsigners,>=20sourcegroups,>=50compatibleDEV and>=4DEVsourcegroups. No WER used to choose threshold.',
  'evaluation':'Unchanged original413DEV plus all whole sequences compatible with actual retained TRAIN vocabulary. RestrictedDEV selects local diagnostic checkpoint; originalDEV reported separately.',
  'test_accessed':False,'fit_authorization':'Exactly one bounded research pilot only if candidate gates pass.'})
 catalog=read(PARENT/'eligible_corpus/manifest.json');asl=read(PARENT/'data/processed/train_dev/manifest.json');at=[r for r in asl if r['split']=='train'];ad=[r for r in asl if r['split']=='dev']
 ncs=[r for r in catalog if r['training_pool'].startswith('NCSLGR')];oldncs=read(TREE/'20261020_continuous_expand09/data/processed/train_dev/manifest.json');nd=[r for r in oldncs if r['split'] in ['dev','val']];ntold=[r for r in oldncs if r['split']=='train']
 assert len(at)==1706 and len(ad)==413 and len(ncs)==1002 and len(nd)==9
 recentdev=read(DIAG/'pilot64/best_dev.json');recenttrain=read(DIAG/'pilot64/best_train.json');effective=read(DIAG/'pilot64/manifest.json');et=[r for r in effective if r['split']=='train'];assert len(et)==64
 ncsdev=read(TREE/'20261020_continuous_expand09/models/reference48/dev_best.json')
 bigdev=read(PARENT/'pilot_fresh48_v2/best_checkpoint_dev.json')
 poolstats=[];tokenrows=[];pooldetails={};errors=[];candidates=[];candidate_details={};excluded=[]
 for name,tr,dv,scored,historical in [('ASLLRP3',at,ad,recentdev,at),('NCSLGR2',ncs,nd,ncsdev,ntold)]:
  s=support(tr);tc=tokens(tr);dc=tokens(dv);hs=support(historical);pred=Counter(t for p in scored['predictions'] for t in p['hypothesis']);aligncounts=defaultdict(Counter)
  for p in scored['predictions']:
   for t,op in align(p['reference'],p['hypothesis']):aligncounts[t][op]+=1
  for t in sorted(set(tc)|set(dc)):
   tokenrows.append({'pool':name,'label':t,'TRAIN_distinct_utterances':s[t],'TRAIN_token_occurrences':tc[t],'support_bucket':bucket(s[t]),
    'DEV_reference_occurrences':dc[t],'DEV_predicted_occurrences':pred[t],'DEV_only':s[t]==0,'DEV_never_predicted':dc[t]>0 and pred[t]==0,
    'DEV_reference_matches':aligncounts[t]['M'],'DEV_substitutions':aligncounts[t]['S'],'DEV_deletions':aligncounts[t]['D'],
    'effective_model_TRAIN_support':support(et)[t] if name=='ASLLRP3' else hs[t],
    'historical_fit_TRAIN_support':hs[t],'prediction_model':'64TRAIN_epoch40' if name=='ASLLRP3' else '906TRAIN_reference48_selected'})
  bins=Counter(bucket(v) for v in s.values());known={r['signer_id'] for r in tr if r.get('signer_id')};overlap={k:sorted({r.get(k) for r in tr if r.get(k)}&{r.get(k) for r in dv if r.get(k)}) for k in ['sample_id','signer_id','source_group','raw_sha256','decoded_pixel_sha256']}
  total=sum(dc.values());covered=sum(c for t,c in dc.items() if s[t]);whole=sum(set(r['glosses'])<=set(s) for r in dv)
  row={'pool':name,'TRAIN_scope':'frozen1706TRAIN' if name=='ASLLRP3' else '1002TRAIN-eligible catalog;906 used in historical fit',
   'unique_TRAIN_utterances':len(tr),'TRAIN_gloss_tokens':sum(tc.values()),'unique_TRAIN_labels':len(s),'DEV_utterances':len(dv),
   'DEV_token_coverage_pct':100*covered/total,'DEV_whole_utterance_coverage_pct':100*whole/len(dv),'DEV_only_label_count':len(set(dc)-set(s)),
   'DEV_never_predicted_reference_labels':len(set(dc)-set(pred)),'known_TRAIN_signer_ids':len(known),'unknown_TRAIN_signer_records':sum(not r.get('signer_id') for r in tr),
   'source_group_keys':len({r['source_group'] for r in tr}),'source_granularity':'publisher recording' if name=='ASLLRP3' else 'mixed full recording and utterance clip;not independent recording count',
   'split_status':'PASS_single_heldout_publisher_signer' if name=='ASLLRP3' else 'PARTIAL_9DEV_signer_metadata_incomplete'}
  for b in ['1','2-4','5-9','10-19','20+']:row['labels_support_'+b]=bins[b];row['labels_support_'+b+'_pct']=100*bins[b]/len(s)
  poolstats.append(row)
  slack=[r['observations']-len(r['glosses'])-sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:])) for r in tr]
  assert min(slack)>=0
  pooldetails[name]={'top30_tokens':tc.most_common(30),'label_sequence_length':distribution([len(r['glosses']) for r in tr]),'feature_sequence_length':distribution([r['observations'] for r in tr]),
   'CTC_slack':distribution(slack),'CTC_violations':0,'train_dev_known_overlap':overlap,'DEV_only_labels':sorted(set(dc)-set(s)),
   'schema':'literal realized dominant occurrence' if name=='ASLLRP3' else 'literal main gloss; mixed legacy/exact-EAF time strata',
   'DEV_signers':sorted({r['signer_id'] for r in dv if r.get('signer_id')}),'DEV_sources':len({r['source_group'] for r in dv})}
  for threshold in [5,10]:
   allowed={t for t,c in s.items() if c>=threshold};kept=[r for r in tr if set(r['glosses'])<=allowed];actual=support(kept);comp=[r for r in dv if set(r['glosses'])<=set(actual)]
   cover=sum(c for t,c in dc.items() if t in actual);fixed=tr;rounds=[]
   while True:
    fs=support(fixed);nl={t for t,c in fs.items() if c>=threshold};new=[r for r in fixed if set(r['glosses'])<=nl];rounds.append(len(new))
    if len(new)==len(fixed):break
    fixed=new
   key=f'{name}_support{threshold}'
   details={'threshold_in_parent_TRAIN':threshold,'parent_labels_allowed':len(allowed),'retained_TRAIN_utterances':len(kept),'actual_TRAIN_vocabulary_labels':len(actual),
    'retained_tokens':sum(tokens(kept).values()),'excluded_TRAIN_utterances':len(tr)-len(kept),'original_DEV_tokens_covered_pct':100*cover/total,
    'compatible_DEV_utterances':len(comp),'original_DEV_whole_utterance_coverage_pct':100*len(comp)/len(dv),
    'retained_support_min':min(actual.values()) if actual else 0,'retained_labels_below_parent_threshold':sum(c<threshold for c in actual.values()),
    'retained_support_bins':dict(Counter(bucket(c) for c in actual.values())),'TRAIN_signers':sorted({r['signer_id'] for r in kept if r.get('signer_id')}),
    'TRAIN_sources':len({r['source_group'] for r in kept}),'compatible_DEV_sources':len({r['source_group'] for r in comp}),'recursive_counts':rounds,'recursive_final_TRAIN':len(fixed)}
   candidate_details[key]=details
   candidates.append({'candidate':key,**{k:v for k,v in details.items() if not isinstance(v,(list,dict))},'status':'AUDITED_NOT_PROMOTED'})
   save('candidates/'+key+'_train.json',kept);save('candidates/'+key+'_compatible_dev.json',comp)
   for r in tr:
    if r['sample_id'] not in {v['sample_id'] for v in kept}:excluded.append({'candidate':key,'sample_id':r['sample_id'],'reason':'WHOLE_UTTERANCE_HAS_LABEL_BELOW_PARENT_TRAIN_SUPPORT','unsupported_labels':json.dumps(sorted(set(r['glosses'])-allowed),ensure_ascii=False)})
 # Attribute each saved DEV reference edit by support using identical deterministic alignment.
 for model,score,train_universe in [('latest64',recentdev,et),('full1706',bigdev,at),('NCS906',ncsdev,ntold)]:
  sup=support(train_universe);strata=defaultdict(Counter)
  for p in score['predictions']:
   for t,op in align(p['reference'],p['hypothesis']):
    if op!='I':strata[bucket(sup[t])][op]+=1
  for b,c in strata.items():errors.append({'model':model,'effective_TRAIN_support_bucket':b,'reference_tokens':sum(c.values()),'correct':c['M'],'substitutions':c['S'],'deletions':c['D'],'reference_error_fraction':(c['S']+c['D'])/sum(c.values()),'insertions':'excluded from reference-token strata; preserved in original WER'})
 candidate='ASLLRP3_support5';d=candidate_details[candidate]
 valid=d['retained_TRAIN_utterances']>=200 and len(d['TRAIN_signers'])==3 and d['TRAIN_sources']>=20 and d['compatible_DEV_utterances']>=50 and d['compatible_DEV_sources']>=4
 selected=read(RUN/'candidates'/f'{candidate}_train.json');compatible=read(RUN/'candidates'/f'{candidate}_compatible_dev.json')
 assert not ({r['sample_id'] for r in selected}&{r['sample_id'] for r in ad})
 for k in ['signer_id','source_group','raw_sha256','decoded_pixel_sha256']:assert not ({r[k] for r in selected}&{r[k] for r in ad})
 save('frozen_manifest.json',selected+ad);save('compatible_dev.json',compatible)
 with (RUN/'SUBSET_MANIFEST.sha256').open('x') as f:f.write(sha(RUN/'frozen_manifest.json')+'  frozen_manifest.json\n'+sha(RUN/'compatible_dev.json')+'  compatible_dev.json\n')
 table('TOKEN_SUPPORT.csv',tokenrows);table('POOL_COMPARISON.csv',poolstats);table('INTERMEDIATE_SUBSET_COUNTS.csv',candidates);table('EXCLUDED_UTTERANCES.csv',excluded);table('RARE_LABEL_ERRORS.csv',errors)
 save('pool_details.json',pooldetails);save('candidate_details.json',candidate_details)
 save('decision.json',{'status':'PASS_SCOPED_PILOT_CANDIDATE' if valid else 'BLOCKED','selected':candidate,'rule':'ParentTRAINsupport>=5 whole-sequence one-pass; chosen by predeclared coverage gates, no score-based threshold choice.',
  'critical_limitation':'Retained-set support may fall below5; not a guaranteed5-per-label closed core. Recursive5/10 cores empty. No target edits.',
  'details':d,'manifest_sha256':sha(RUN/'frozen_manifest.json'),'compatible_dev_sha256':sha(RUN/'compatible_dev.json'),'original_DEV_unchanged':True,
  'trial_budget':'One fresh425-class48channelCUDA model,30epoch/90s bound,greedyCTC. NewTRAIN-onlymean/std/vocab; restrictedDEVlocal selection,original413DEVreport only.',
  'elapsed_seconds':time.perf_counter()-start,'TEST_used':False})
 save('input_hashes.json',INPUTS)
 print(json.dumps({'status':'CANDIDATE_PASS' if valid else 'BLOCKED','pools':poolstats,'candidates':candidates,'selected':d}),flush=True)

if __name__=='__main__':main()
