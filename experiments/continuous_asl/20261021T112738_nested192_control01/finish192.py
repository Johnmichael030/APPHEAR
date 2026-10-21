"""Freeze/reload the completed192 control; one post-fit80 evaluation, no fitting."""
from pathlib import Path
from collections import Counter
import csv, hashlib, importlib.util, json, sys, time
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
BASE=RUN.parent
OLD=BASE/'20261020T144622Z_compact_ctc_control01'
PREV=BASE/'20261020T143609Z_learning_diagnostic01'
sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
from ctc_experiment import CausalCTC,batch,decode,edit_counts,seeds

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 with (RUN/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
def csvsave(name,rows):
 with (RUN/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def family(r):
 import re
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group'])
 return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
def aggregate(predictions):
 c=Counter()
 for p in predictions:c.update(edit_counts(p['reference'],p['hypothesis']))
 n=c['reference_tokens'];words=Counter(t for p in predictions for t in p['hypothesis'])
 return dict(utterances=len(predictions),WER=sum(c[k] for k in ['substitutions','deletions','insertions'])/n,**dict(c),exact_sequences=sum(p['reference']==p['hypothesis'] for p in predictions),empty_outputs=sum(not p['hypothesis'] for p in predictions),token_diversity=len(words),hypothesis_tokens=sum(words.values()),CTC_loss=float(np.mean([p['CTC_loss'] for p in predictions])),blank_fraction=sum(p['blank_frames'] for p in predictions)/sum(p['output_frames'] for p in predictions))

def main():
 assert not (RUN/'heldout80_predictions.json').exists(),'POSTFIT_EVALUATION_ALREADY_COMPLETE'
 d=read(RUN/'decision.json');cfg=read(RUN/'control_config.json');subset=read(RUN/'subset_manifest.json')
 assert d['complete_epochs']==30 and d['optimizer_steps']==720
 integrity=read(RUN/'control_input_integrity.json');assert integrity['status']=='PASS' and not integrity['changed']
 before=read(RUN/'control_input_hashes_before.json')
 for p,e in before.items():
  q=Path(p);s=q.stat();assert sha(q)==e['sha256'] and s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'],p
 checkpoint=RUN/d['checkpoint']['path'];assert sha(checkpoint)==d['checkpoint']['sha256']
 frozen=read(RUN/'fold_manifest.json');assert sha(RUN/'fold_manifest.json')==cfg['frozen_fold_sha256']
 vocab=read(RUN/'vocabulary.json')['tokens'];assert sha(RUN/'vocabulary.json')==cfg['vocabulary_sha256']
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert ck['vocabulary']==vocab and ck['config_sha256']==sha(RUN/'control_config.json')
 initial=torch.load(RUN/'checkpoints/initialization.pt',map_location='cpu',weights_only=False)
 oldinit=torch.load(OLD/'checkpoints/initialization.pt',map_location='cpu',weights_only=False)
 assert initial['model'].keys()==oldinit['model'].keys()
 assert all(torch.equal(initial['model'][k],oldinit['model'][k]) for k in initial['model'])
 tensor_hash=hashlib.sha256()
 for k in sorted(initial['model']):tensor_hash.update(k.encode()+initial['model'][k].numpy().tobytes())
 universe=read(PREV/'resolved_manifest.json');fulltrain=[r for r in universe if r['internal_role']=='TRAIN']
 lookup={r['sample_id']:r for r in universe};held=[lookup[i] for i in frozen['compatible_eval_sample_ids']]
 strongids=set(frozen['support5_eval_sample_ids'])
 assert len(held)==80 and len(strongids)==27 and strongids<={r['sample_id'] for r in held}
 ids={r['sample_id'] for r in subset};originalids={r['sample_id'] for r in read(OLD/'subset_manifest.json')}
 assert len(ids)==192 and originalids<ids and ids<=set(frozen['train_sample_ids'])
 assert not ids&set(frozen['internal_eval_sample_ids'])
 assert not ({family(r) for r in subset}&{family(r) for r in held})
 assert {r['training_pool'] for r in subset+held}=={'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE'}
 assert {r['signer_id'] for r in held}<={r['signer_id'] for r in subset}
 support=Counter(t for r in subset for t in set(r['glosses']))
 fullsupport=Counter(t for r in fulltrain for t in set(r['glosses']))
 data={};feature_hashes={}
 for r in held:
  p=Path(r['feature_path']);assert sha(p)==r['feature_sha256'];feature_hashes[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as z:
   assert str(z['schema_version'].item())==cfg['schema']
   x=z['X'].copy();ts=z['timestamp_ms'];meta=json.loads(str(z['metadata_json'].item()))
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
   assert meta['sample_id']==r['sample_id'] and meta['glosses']==r['glosses']
   assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
   assert all(t in vocab[2:] for t in r['glosses'])
   data[r['sample_id']]=x
 seeds(cfg['seed']);assert torch.cuda.is_available()
 model=CausalCTC(len(vocab),48,ck['model']['mean'].numpy(),ck['model']['std'].numpy()).to('cuda:0')
 model.load_state_dict(ck['model'],strict=True);model.eval();criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
 started=time.perf_counter();predictions=[]
 with torch.inference_mode():
  for r in held:
   x,y,l,tl=batch([r],data,vocab,'cuda:0');z=model(x);loss=criterion(z.log_softmax(-1).transpose(0,1),y,l,tl)
   assert torch.isfinite(loss);h=decode(z[0],vocab);arg=z[0].argmax(-1)
   predictions.append(dict(sample_id=r['sample_id'],reference=r['glosses'],hypothesis=h,**edit_counts(r['glosses'],h),CTC_loss=float(loss),blank_frames=int((arg==0).sum()),output_frames=len(arg),family=family(r),labels_not_fitted_in192=sum(t not in support for t in r['glosses'])))
 torch.cuda.synchronize();eval_seconds=time.perf_counter()-started
 scores=aggregate(predictions);strong=aggregate([p for p in predictions if p['sample_id'] in strongids])
 save('heldout80_predictions.json',dict(**scores,predictions=predictions,scope='POSTFIT_FIXED_CHECKPOINT_KNOWN_SIGNER_FAMILY_HOLDOUT_NO_SELECTION'))
 save('heldout27_subgroup.json',dict(**strong,scope='ORIGINAL_SUPPORT5_SUBGROUP_FULL1500_SUPPORT_NOT_REDEFINED_FOR192'))
 totalrefs=sum(len(r['glosses']) for r in held)
 coverage=dict(heldout_utterances=80,reference_tokens=totalrefs,model_vocabulary_covered_reference_tokens=totalrefs,actual192_target_covered_reference_tokens=sum(t in support for r in held for t in r['glosses']),actual192_target_fully_covered_utterances=sum(all(t in support for t in r['glosses']) for r in held),new192_unique_target_labels=len(support),old96_unique_target_labels=read(OLD/'subset_coverage.json')['unique_labels'],same_signers=True,recording_family_disjoint=True,heldout_families=len({family(r) for r in held}),no_selection=True,unseen_signer_generalization=False)
 save('heldout_coverage.json',coverage)
 final=read(RUN/'final_predictions.json');assert aggregate(final['predictions'])['WER']==final['WER']
 old=read(OLD/'final_predictions.json')
 metrics=[]
 for name,pp in [('Frozen96_TRAIN',old['predictions']),('New192_TRAIN',final['predictions']),('Original96_inside_new192', [p for p in final['predictions'] if p['sample_id'] in originalids]),('Added96_inside_new192',[p for p in final['predictions'] if p['sample_id'] not in originalids]),('New192_fixed_heldout80',predictions),('Original_support5_subgroup27',[p for p in predictions if p['sample_id'] in strongids])]:
  metrics.append(dict(scope=name,**aggregate(pp)))
 csvsave('LEARNING_COMPARISON.csv',metrics)
 csvsave('PREDICTION_EXAMPLES.csv',[dict(scope=scope,sample_id=p['sample_id'],reference=json.dumps(p['reference'],ensure_ascii=False),hypothesis=json.dumps(p['hypothesis'],ensure_ascii=False),substitutions=p['substitutions'],deletions=p['deletions'],insertions=p['insertions'],reference_tokens=p['reference_tokens']) for scope,pp in [('TRAIN192',final['predictions']),('HELDOUT80',predictions)] for p in pp])
 priorities=[]
 for token,n in sorted(fullsupport.items(),key=lambda x:(x[1],x[0])):
  if n>=5:continue
  matching=[r for r in fulltrain if token in r['glosses']]
  priorities.append(dict(literal_gloss=token,training_pool='ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE',distinct_TRAIN_utterances=n,known_signers=len({r['signer_id'] for r in matching}),recording_families=len({family(r) for r in matching}),planning_additional_utterances_to_reach5=5-n,annotation_review_required=True,collection_permission_status='NOT_ESTABLISHED_FOR_NEW_RECORDINGS'))
 csvsave('LABEL_COLLECTION_PRIORITIES.csv',priorities)
 epoch=list(csv.DictReader((RUN/'epoch_timing.csv').open()))
 gpu=list(csv.DictReader((RUN/'gpu_monitor.csv').open()))
 used=[float(r['device_used_MiB']) for r in gpu if r['device_used_MiB'].replace('.','',1).isdigit()]
 result=dict(status='PASS_LEARNING_CONTROL_PARTIAL_GENERALIZATION',decision=d,heldout80=scores,strong27=strong,coverage=coverage,initial_model_tensors_identical_to96=True,initial_model_tensor_sha256=tensor_hash.hexdigest(),heldout_evaluation_seconds=eval_seconds,optimizer_epoch_seconds_mean=float(np.mean([float(r['optimizer_epoch_seconds']) for r in epoch])),optimizer_epoch_seconds_total=sum(float(r['optimizer_epoch_seconds']) for r in epoch),devicewide_peak_used_MiB=max(used) if used else None,learning_comparison=metrics,singleton_labels=sum(n==1 for n in fullsupport.values()),below5_labels=len(priorities),qualified_reviewer='UNAVAILABLE_PER_USER',new_participant_permissions='NOT_ESTABLISHED',TEST_used=False,promoted=False)
 save('RESULTS.json',result)
 assert sha(checkpoint)==d['checkpoint']['sha256']
 for p,digest in feature_hashes.items():assert sha(p)==digest
 save('postfit_evaluation_integrity.json',dict(status='PASS',checkpoint_sha256=d['checkpoint']['sha256'],heldout_feature_hashes=feature_hashes,original_control_inputs_unchanged=len(before),initial_tensors_identical=True,no_reserved_TEST_access=True))
 print(json.dumps({k:v for k,v in result.items() if k not in ['learning_comparison','decision']},indent=2))

if __name__=='__main__':main()
