"""Frozen TRAIN-only cached-cadence replay; no fitting, TEST, camera or label changes."""
import csv, hashlib, json, pathlib, sys, time
from collections import Counter
import numpy as np
import onnxruntime as ort
ROOT=pathlib.Path.cwd(); OUT=pathlib.Path('evidence/current_continuous_detection_task.txt').read_text().strip(); OUT=pathlib.Path(OUT)
sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
from ctc_experiment import edit_counts

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
manifest=ROOT/'experiments/continuous_asl/20261021T112738_nested192_control01/subset_manifest.json'
package=ROOT/'packages/continuous_asl_research_01/android_assets/continuous_asl'
vocab=json.loads((package/'vocabulary.json').read_text()); mp=json.loads((package/'manifest.json').read_text())
assert sha(package/'causal_gloss_ctc_fp32.onnx')==mp['model_sha256']
assert sha(package/'vocabulary.json')==mp['vocabulary_sha256']
config={'schema':'apphear.continuous_asl.cached_training_replay.v1','rows':'frozen192TRAINonly','manifest':str(manifest),'manifest_sha256':sha(manifest),'model_sha256':mp['model_sha256'],'vocabulary_sha256':mp['vocabulary_sha256'],'policies':['continuous_training_features_greedy','current_nohand_flush','nohand_discard_fix'],'changed_runtime_variable':'No-hand pending-token flush versus discard only. Both retain existing no-hand model/feature reset.','stable_token_ms':200.0,'sampling':'Existing cached10Hz genuine observations only; this does NOT simulate missing intervening camera frames or measure live accuracy.','maximum_tokens':128,'reserved_TEST_accessed':False,'model_selection':False}
config_path=OUT/'runtime_replay_config.json'
if config_path.exists():raise RuntimeError('REPLAY_ALREADY_CONFIGURED')
save('runtime_replay_config.json',config)
options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
session=ort.InferenceSession(str(package/'causal_gloss_ctc_fp32.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
rows=json.loads(manifest.read_text()); assert len(rows)==192 and all(r['split']=='train' for r in rows)
aggregate={p:Counter(substitutions=0,deletions=0,insertions=0,reference_tokens=0,exact=0,tokens=0) for p in config['policies']}
examples=[]; events=Counter(); input_paths=[]; start=time.perf_counter()

def greedy(ids):
    previous=-1; result=[]
    for token in ids:
        if token!=previous and token!=0:result.append(vocab[token])
        previous=token
    return result

def stable(ids,timestamps,visible,flush):
    previous=0; emitted=True; since=0.; result=[]; nohand_emissions=0
    for token,t,seen in zip(ids,timestamps,visible):
        if len(result)>=128:break
        if not seen:
            if flush and not emitted:
                result.append(vocab[previous]); nohand_emissions+=1
            previous=0; emitted=True; continue
        token=int(token)
        if token!=previous:
            if not emitted:result.append(vocab[previous])
            previous=token; since=float(t); emitted=token==0
        elif not emitted and t-since>=200.:
            result.append(vocab[previous]); emitted=True
    # Streaming screen has no utterance oracle/end-of-record flush: do not fabricate one.
    return result,nohand_emissions

for row in rows:
    p=pathlib.Path(row['feature_path']); assert sha(p)==row['feature_sha256']; input_paths.append((str(p),row['feature_sha256']))
    with np.load(p,allow_pickle=False) as z:
        assert str(z['schema_version'].item())=='apphear.continuous_asl.candidate01_causal378.v1'
        x=z['X'].copy(); ts=z['timestamp_ms'].copy(); metadata=json.loads(str(z['metadata_json'].item()))
    assert metadata['glosses']==row['glosses'] and metadata['sample_id']==row['sample_id']
    assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
    # Effective joint-valid flags already include the handObserved gate, matching Core.visible.
    visible=np.any(x[:,84:126]>0,axis=1)
    events.update(utterances=1,frames=len(x),nohand_frames=int((~visible).sum()),hand_reentries=int(np.sum(visible[1:]&~visible[:-1])))
    full_ids=session.run(['logits'],{'features':x[None]})[0][0].argmax(-1)
    current_ids=np.zeros(len(x),dtype=np.int64)
    k=0
    while k<len(x):
        if not visible[k]:k+=1;continue
        end=k+1
        while end<len(x) and visible[end] and ts[end]-ts[end-1]<=1000.:end+=1
        block=x[k:end].copy(); block[0,215:378]=0.
        current_ids[k:end]=session.run(['logits'],{'features':block[None]})[0][0].argmax(-1)
        events['visible_runs']+=1; k=end
    outputs={'continuous_training_features_greedy':greedy(full_ids)}
    outputs['current_nohand_flush'],triggered=stable(current_ids,ts,visible,True)
    outputs['nohand_discard_fix'],fixed=stable(current_ids,ts,visible,False)
    assert fixed==0
    events['pending_tokens_published_at_nohand_before']+=triggered
    events['pending_tokens_published_at_nohand_after']+=fixed
    events['visible_argmax_differences_from_continuous_context']+=int(np.sum(current_ids[visible]!=full_ids[visible]))
    for policy,hyp in outputs.items():
        edits=edit_counts(row['glosses'],hyp); aggregate[policy].update(edits)
        aggregate[policy].update(exact=int(hyp==row['glosses']),tokens=len(hyp))
        examples.append({'sample_id':row['sample_id'],'policy':policy,'reference':' '.join(row['glosses']),'hypothesis':' '.join(hyp),'nohand_frames':int((~visible).sum()),**edits})
summary={}
for policy,counts in aggregate.items():
    summary[policy]=dict(counts); summary[policy]['WER']=(counts['substitutions']+counts['deletions']+counts['insertions'])/counts['reference_tokens']
    summary[policy]['exact_sequence_fraction']=counts['exact']/len(rows)
result={'status':'PASS computational TRAIN-only replay; NOT live or unseen-signer accuracy','runtime_seconds':time.perf_counter()-start,'events':dict(events),'metrics':summary,'interpretation':'A no-hand flush can create display emissions without observed valid hands. Removing it enforces abstention but can increase deletions. Training-continuous versus runtime-reset context is a separate observed serving-policy mismatch, not a proven generalization cause. No thresholds/model were tuned.'}
save('RUNTIME_REPLAY_RESULTS.json',result)
with (OUT/'RUNTIME_REPLAY_EXAMPLES.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
save('runtime_replay_inputs.json',[{'path':p,'sha256':s} for p,s in input_paths])
assert all(sha(p)==s for p,s in input_paths)
print(json.dumps(result,indent=2))
