"""Bounded read-only audits; outputs confined to this new research directory."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,sys,time,unicodedata
import numpy as np
import torch
RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[2]; PARENT=RUN.parent/'20261020T104054Z_expand_eval01'
sys.path.insert(0,str(ROOT/'tools/continuous_asl'));sys.path.insert(0,str(ROOT))
from ctc_experiment import CausalCTC,batch,evaluate,seeds,decode
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,x):
 with (RUN/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.write('\n')
def table(name,rows):
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def measure(model,rows,data,vocab):
 score=evaluate(model,rows,data,vocab,'cuda:0');frames=Counter();prob=0.;steps=0
 with torch.inference_mode():
  for r in rows:
   z=model(torch.as_tensor(data[r['sample_id']],device='cuda:0')[None])[0]
   frames.update(z.argmax(-1).cpu().tolist());prob+=float(z.softmax(-1)[:,0].sum());steps+=len(z)
 tokens=Counter(t for p in score['predictions'] for t in p['hypothesis']);refs=Counter(t for r in rows for t in r['glosses'])
 lengths=Counter(len(p['hypothesis']) for p in score['predictions'])
 score.update(blank_argmax_fraction=frames[0]/steps,blank_mean_probability=prob/steps,total_timesteps=steps,
  unique_predicted_tokens=len(tokens),prediction_tokens=sum(tokens.values()),empty_hypotheses=lengths[0],
  hypothesis_length_histogram=dict(lengths),hypothesis_token_counts=dict(tokens),reference_token_counts=dict(refs),
  frame_argmax_counts={vocab[k]:v for k,v in frames.items()},scope='RESEARCH_ONLY; publisher-signers verified; no streaming/product claim')
 return score

def main():
 started=time.perf_counter();seeds(20261020)
 manifest=PARENT/'data/processed/train_dev/manifest.json';rows=read(manifest);plan=read(PARENT/'dataset_plan.json');allrows=read(PARENT/'eligible_corpus/manifest.json')
 assert sha(manifest)==plan['manifest_sha256'];assert len(allrows)==3121
 vocab=read(PARENT/'pilot_fresh48_v2/vocabulary.json')['tokens'];mapping={t:i for i,t in enumerate(vocab)}
 fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
 assert len(vocab)==len(mapping)==2519 and vocab[:2]==['<BLANK>','<UNK>']
 assert vocab[2:]==sorted({t for r in fit for t in r['glosses']})
 assert not ({'<BLANK>','<UNK>'}&{t for r in allrows for t in r['glosses']})
 poolrows=[];freq=[];poolsets={};feature_data={};lengths=[];pool_details={}
 for pool in sorted({r['training_pool'] for r in allrows}):
  group=[r for r in allrows if r['training_pool']==pool];counts=Counter(t for r in group for t in r['glosses']);poolsets[pool]=set(counts)
  whitespace=[t for t in counts if t!=t.strip()];norm=defaultdict(list)
  for t in counts:norm[unicodedata.normalize('NFC',t)].append(t)
  collisions=[v for v in norm.values() if len(v)>1]
  unknown_signers=sum(not r.get('signer_id') for r in group)
  signer_ids={r['signer_id'] for r in group if r.get('signer_id')}
  if pool.startswith('ASLLRP'):signer_ids={r['signer_id'] for r in rows};unknown_signers=0
  pool_details[pool]={'token_counts':dict(counts),'literal_whitespace_labels':whitespace,'unicode_NFC_collisions':collisions,
   'timing_status_counts':dict(Counter(r.get('timing_status',r.get('time_units','UNKNOWN')) for r in group))}
  poolrows.append({'status':'PASS_SCOPED_CONTRACT' if pool.startswith('ASLLRP') else 'PARTIAL_TEMPORAL_AND_DEV_LIMITS','pool':pool,'eligible_utterances':len(group),
   'unique_glosses':len(counts),'reference_tokens':sum(counts.values()),'top_gloss':counts.most_common(1)[0][0],'top_gloss_count':counts.most_common(1)[0][1],
   'singleton_glosses':sum(v==1 for v in counts.values()),'source_recordings':len({r['source_group'] for r in group}),'known_signer_ids':len(signer_ids),
   'unknown_signer_records':unknown_signers,'feature_schema':SCHEMA,'label_contract':'literal realized dominant occurrence' if pool.startswith('ASLLRP') else 'literal main gloss; legacy and exact EAF timing strata',
   'DEV_contract':'413 same-pool held-out publisher signer' if pool.startswith('ASLLRP') else 'historical9 DEV; no verified robust same-pool signer split here','merge_status':'BLOCKED_UNPROVEN_ONTOLOGY_EQUIVALENCE'})
  for t,c in counts.most_common():freq.append({'pool':pool,'token':t,'count':c})
 for r in allrows:
  assert r['glosses'] and all(isinstance(t,str) and t for t in r['glosses'])
  assert sha(r['feature_path'])==r['feature_sha256']
  with np.load(r['feature_path'],allow_pickle=False) as f:
   assert str(f['schema_version'].item())==SCHEMA
   x=f['X'].copy();ts=f['timestamp_ms'].copy()
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
  minimum=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
  assert len(x)>=minimum
  if r['training_pool'].startswith('ASLLRP'):
   assert all(r['utterance_start_frame_master']<=e['start_frame_master']<=e['end_frame_master']<=r['utterance_end_frame_master'] for e in r['events'])
   feature_data[r['sample_id']]=x
  lengths.append({'sample_id':r['sample_id'],'pool':r['training_pool'],'observations':len(x),'target_length':len(r['glosses']),'minimum_CTC_steps':minimum,'slack':len(x)-minimum,'status':'PASS_NO_DOWNSAMPLING'})
 status=read(PARENT/'pilot_fresh48_v2/training_status.json');cp=PARENT/'pilot_fresh48_v2'/status['best_checkpoint']['path'];assert sha(cp)==status['best_checkpoint']['sha256']
 state=torch.load(cp,map_location='cpu',weights_only=False);assert state['vocabulary']==vocab
 values=np.concatenate([feature_data[r['sample_id']] for r in fit]).astype(np.float64)
 mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32);del values
 assert np.array_equal(mean,state['model']['mean'].numpy()) and np.array_equal(std,state['model']['std'].numpy())
 model=CausalCTC(len(vocab),48,mean,std).cuda();model.load_state_dict(state['model']);del state
 # Decoder identity: blank-separated duplicates must survive as two literal tokens.
 ids=[0,2,2,0,2,3,0];z=torch.full((len(ids),len(vocab)),-10.);z[range(len(ids)),ids]=10.
 assert decode(z,vocab)==[vocab[2],vocab[2],vocab[3]]
 audits=[];examples=[]
 for split,selected in [('train',fit),('dev',dev)]:
  result=measure(model,selected,feature_data,vocab)
  saved=read(PARENT/('pilot_fresh48_v2/best_checkpoint_full_train.json' if split=='train' else 'pilot_fresh48_v2/best_checkpoint_dev.json'))
  assert result['predictions']==saved['predictions']
  save('replayed_'+split+'.json',result)
  tc=Counter(result['hypothesis_token_counts']);rc=Counter(result['reference_token_counts']);top,count=tc.most_common(1)[0]
  audits.append({'status':'PASS_MEASURED_COLLAPSE','split':split,'utterances':len(selected),'WER':result['gloss_wer'],'CTC_loss':result['loss'],
   'blank_argmax_fraction':result['blank_argmax_fraction'],'mean_blank_probability':result['blank_mean_probability'],'empty_outputs':result['empty_hypotheses'],
   'unique_predicted_tokens':len(tc),'prediction_tokens':sum(tc.values()),'top_predicted_token':top,'top_token_count':count,'top_token_fraction':count/sum(tc.values()),
   'unique_reference_tokens':len(rc),'reference_tokens':sum(rc.values()),'top_reference_token':rc.most_common(1)[0][0],'top_reference_count':rc.most_common(1)[0][1],
   'hypothesis_length_histogram':json.dumps(result['hypothesis_length_histogram'])})
  for p in result['predictions']:examples.append({'split':split,'sample_id':p['sample_id'],'reference':json.dumps(p['reference'],ensure_ascii=False),'hypothesis':json.dumps(p['hypothesis'],ensure_ascii=False),'S':p['substitutions'],'D':p['deletions'],'I':p['insertions'],'status':'MEASURED_LITERAL_LABELS'})
 intersections={k:sorted({r[k] for r in fit}&{r[k] for r in dev}) for k in ['sample_id','signer_id','source_group','collection','raw_sha256','decoded_pixel_sha256']};assert not any(intersections.values())
 keys=list(poolsets);shared=poolsets[keys[0]]&poolsets[keys[1]]
 table('POOL_COMPARISON.csv',poolrows);table('TOKEN_COLLAPSE_AUDIT.csv',audits);table('PREDICTION_EXAMPLES.csv',examples);table('pool_token_frequencies.csv',freq);table('ctc_lengths.csv',lengths)
 save('pool_details.json',pool_details)
 save('critical_checks.json',{'status':'PASS_SCOPED_RESEARCH_PILOT','feature_records_checked':len(allrows),'selected_checkpoint_replay_exact':True,
  'manifest_sha256':sha(manifest),'selected_checkpoint_sha256':sha(cp),'blank_id':0,'unknown_id':1,'training_unknown_targets':0,
  'vocabulary_exact_checkpoint_and_train_set_match':True,'normalization_exact_TRAIN_recomputation_match':True,'CTC_length_violations':0,
  'decoder_duplicate_blank_roundtrip':True,'pool_shared_literal_spellings':len(shared),'pool_union_glosses':len(poolsets[keys[0]]|poolsets[keys[1]]),
  'semantic_pool_equivalence':'UNKNOWN; shared spelling is insufficient','train_dev_overlap':intersections,'DEV_OOV_tokens':plan['dev_oov_tokens'],
  'alignment_limits':'Publisher frame/event and feature contracts verified; prior joins reused. No independent linguistic/visual boundary ground truth created. NCSLGR legacy alignment remains limited.',
  'elapsed_seconds':time.perf_counter()-started,'TEST_used':False,'bug_found':False})
 print(json.dumps({'status':'AUDIT_PASS','seconds':time.perf_counter()-started,'collapse':audits,'pool_summary':poolrows}),flush=True)

if __name__=='__main__':main()
