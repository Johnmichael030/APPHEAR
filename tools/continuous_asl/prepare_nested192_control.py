"""Create one nested TRAIN-size control by reusing the frozen compact trainer."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/continuous_asl'
OLD = BASE/'20261020T144622Z_compact_ctc_control01'
code = (OLD/'control.py').read_text(encoding='utf-8')
assert hashlib.sha256((OLD/'control.py').read_bytes()).hexdigest() == '4521c8a5948c9e3a512c0c5f01fcba02ef1bb48fc8955427e1d636803a9e377f'
new_prepare = '''def prepare():
 assert not (RUN/'control_config.json').exists(),'PREPARATION_ALREADY_FROZEN'
 old=BASE/'20261020T144622Z_compact_ctc_control01'
 frozen=read(old/'frozen_control_inputs.json')
 assert sha(old/'subset_manifest.json')==frozen['subset_manifest_sha256']=='c7b78f1aef43b5501fce5fafbf69e91753e5d9d85cb594a3e18fc693ecfb5198'
 original=read(old/'subset_manifest.json');assert len(original)==96
 for line in (old/'artifact_hashes.sha256').read_text().splitlines():
  digest,rel=line.split('  ',1);p=(old/rel).resolve();assert old in p.parents and sha(p)==digest
 rows=read(PREV/'resolved_manifest.json');train=[r for r in rows if r['internal_role']=='TRAIN']
 assert len(train)==1500 and len({r['sample_id'] for r in train})==1500
 fold=read(PREV/'fold_manifest.json');assert {r['sample_id'] for r in train}==set(fold['train_sample_ids'])
 assert all(r['split']=='train' and r['training_pool']=='ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE' for r in train)
 meta={r['sample_id']:r for r in read(AUDIT/'feature_and_target_checks.json') if r['role']=='TRAIN'}
 assert set(meta)=={r['sample_id'] for r in train}
 support=Counter(t for r in train for t in set(r['glosses']));occ=Counter(t for r in train for t in r['glosses'])
 oldcoverage=read(old/'subset_coverage.json');tq=oldcoverage['input_quartile_cuts'];uq=oldcoverage['target_quartile_cuts']
 def fields(r):
  m=meta[r['sample_id']]
  return dict(family=family(r),signer=r['signer_id'],master=r['source_group'],input_quartile=int(np.searchsorted(tq,m['observations'],side='left')),target_quartile=int(np.searchsorted(uq,len(r['glosses']),side='left')),rare_bearing=any(support[t]<5 for t in r['glosses']),common_bearing=any(support[t]>=20 for t in r['glosses']),observation_category='NO_FRAME' if m['missing_event_observation_count'] else 'NO_HAND' if m['sampled_events_no_hand_count'] else 'OTHER',missing_frame_events=m['missing_event_observation_count'],no_hand_events=m['sampled_events_no_hand_count'])
 desc={r['sample_id']:fields(r) for r in train};lookup={r['sample_id']:r for r in train}
 selected=[lookup[r['sample_id']] for r in original];used={r['sample_id'] for r in selected}
 for r in original:
  assert r['compact_selection']==desc[r['sample_id']]
  assert all(r[k]==lookup[r['sample_id']][k] for k in ['feature_path','feature_sha256','glosses','source_group','signer_id'])
 families=sorted({family(r) for r in train});assert len(families)==6
 assert all(sum(family(r)==f for r in train)>=32 for f in families)
 strata=Counter();globalcounts=Counter();masters=Counter()
 def update(r):
  d=desc[r['sample_id']];fam=d['family']
  strata[(fam,d['input_quartile'],d['target_quartile'],d['observation_category'],d['rare_bearing'])]+=1;masters[d['master']]+=1
  for k in ['input_quartile','target_quartile','observation_category','rare_bearing','common_bearing']:globalcounts[(k,d[k])]+=1
 for r in selected:update(r)
 for roundnum in range(16):
  for fam in families:
   choices=[r for r in train if family(r)==fam and r['sample_id'] not in used]
   def key(r):
    d=desc[r['sample_id']];sk=(fam,d['input_quartile'],d['target_quartile'],d['observation_category'],d['rare_bearing'])
    balance=sum(globalcounts[(k,d[k])] for k in ['input_quartile','target_quartile','observation_category','rare_bearing','common_bearing'])
    return strata[sk],masters[d['master']],balance,hashlib.sha256(('compact96-seed20261020|'+r['sample_id']).encode()).hexdigest()
   r=min(choices,key=key);selected.append(r);used.add(r['sample_id']);update(r)
 assert len(selected)==192 and len(used)==192 and all(sum(family(r)==f for r in selected)==32 for f in families)
 assert used<=set(fold['train_sample_ids']) and not used&set(fold['internal_eval_sample_ids'])
 held=[r for r in rows if r['sample_id'] in set(fold['compatible_eval_sample_ids'])]
 assert len(held)==80 and not ({family(r) for r in selected}&{family(r) for r in held})
 vocab=read(PREV/'vocabulary.json')['tokens'];assert vocab==['<BLANK>','<UNK>']+sorted(support)
 assert all(t in vocab[2:] for r in held for t in r['glosses'])
 paths=[Path(__file__),old/'control.py',old/'subset_manifest.json',old/'subset_coverage.json',old/'control_config.json',old/'decision.json',old/'final_predictions.json',old/'checkpoints/initialization.pt',old/'checkpoints/epoch_030.pt',old/'artifact_hashes.sha256',PREV/'resolved_manifest.json',PREV/'fold_manifest.json',PREV/'vocabulary.json',PREV/'normalization.npz',PREV/'config.json',AUDIT/'feature_and_target_checks.json',ROOT/'tools/continuous_asl/ctc_experiment.py',ROOT/'tools/continuous_asl/prepare_features.py']
 assert sha(ROOT/'tools/continuous_asl/ctc_experiment.py')==read(PREV/'config.json')['source_model_decoder_sha256']
 for r in selected:
  p=Path(r['feature_path']);assert sha(p)==r['feature_sha256'];paths.append(p)
  with np.load(p,allow_pickle=False) as z:
   assert str(z['schema_version'].item())==SCHEMA
   x=z['X'];ts=z['timestamp_ms'];m=json.loads(str(z['metadata_json'].item()))
   assert x.dtype==np.float32 and x.shape==(len(ts),378) and np.isfinite(x).all() and np.all(np.diff(ts)>0)
   assert m['sample_id']==r['sample_id'] and m['glosses']==r['glosses']
   assert len(x)>=len(r['glosses'])+sum(a==b for a,b in zip(r['glosses'],r['glosses'][1:]))
 selectedsupport=Counter(t for r in selected for t in set(r['glosses']))
 coverage=dict(utterances=192,retained_original_96=True,added_utterances=96,families=dict(Counter(family(r) for r in selected)),signers=dict(Counter(r['signer_id'] for r in selected)),masters=len(masters),unique_labels=len(selectedsupport),token_occurrences=sum(len(r['glosses']) for r in selected),selected_labels_by_fullTRAIN_support={name:sum(fn(support[t]) for t in selectedsupport) for name,fn in [('1',lambda n:n==1),('2_4',lambda n:2<=n<=4),('5_9',lambda n:5<=n<=9),('10_19',lambda n:10<=n<=19),('20plus',lambda n:n>=20)]},input_quartile_cuts=tq,target_quartile_cuts=uq,input_quartiles=dict(Counter(desc[r['sample_id']]['input_quartile'] for r in selected)),target_quartiles=dict(Counter(desc[r['sample_id']]['target_quartile'] for r in selected)),utterances_no_retained_frame_events=sum(desc[r['sample_id']]['missing_frame_events']>0 for r in selected),utterances_sampled_no_hand_events=sum(desc[r['sample_id']]['no_hand_events']>0 for r in selected),limitation='Nested TRAIN learnability; equal family quotas. Same known signers; no independent signer claim.')
 cfg=read(old/'control_config.json');cfg.update(max_epochs=30,max_optimizer_steps=720,selection='Frozen original96 plus16 unused TRAIN utterances/family; same coverage rule/tie seed, counters initialized with96',control_script_sha256=sha(__file__),previous_subset_sha256=sha(old/'subset_manifest.json'),intended_changed_variable='TRAIN set size96to192',training_passes=30,optimizer_steps_consequence='24per_epoch/720total versus12/360; NOT matched compute',evaluation_policy='TRAIN learning only during fit; fixed primary80 and subgroup27 evaluated once after epoch30 is frozen; no selection/tuning on them')
 save(RUN/'subset_manifest.json',[dict(**r,compact_selection=desc[r['sample_id']]) for r in selected])
 save(RUN/'subset_coverage.json',coverage);save(RUN/'control_config.json',cfg)
 for name in ['vocabulary.json','normalization.npz','fold_manifest.json']:shutil.copyfile(PREV/name,RUN/name)
 paths += [RUN/n for n in ['subset_manifest.json','subset_coverage.json','control_config.json','vocabulary.json','normalization.npz','fold_manifest.json']]
 save(RUN/'control_input_hashes_before.json',snap(paths))
 save(RUN/'diagnosis_summary.json',dict(pipeline_valid_for_compact_control=True,source=str(old/'diagnosis_summary.json'),new_feature_checks=192,nested_membership=True,heldout_recording_family_guard=True,authentic_labels=True))
 (RUN/'PHASE1_FINDINGS.md').write_text('PASS: reuse original numerical diagnosis; verify192 schema/feature hashes/chronology/labels/CTC feasibility. Frozen original96 retained; no TEST access.\\n')
 save(RUN/'preflight.json',dict(status='PASS',subset_features_verified=192,vocabulary_size=len(vocab),nested_original_96=True,numerical_integrity=True,fold_membership=True,source_family_guard=True))
 print(json.dumps(dict(status='PREPARED',coverage=coverage,predeclared_bounds=dict(epochs=30,steps=720,seconds=90)),indent=2),flush=True)
'''
start = code.index('def prepare():')
end = code.index('\ndef fit():', start)
code = code[:start] + new_prepare + code[end:]
code = code.replace('assert len(rows)==96 and all(', 'assert len(rows)==192 and all(')
code = code.replace("if passed:reason='PREDECLARED_LEARNING_GATE_PASS';break", "if passed and epoch==cfg['max_epochs']:reason='PREDECLARED_LEARNING_GATE_PASS';break")
code = code.replace("current_epoch=epoch\n   if epoch==1", """current_epoch=epoch
   torch.cuda.synchronize()
   timing=dict(epoch=epoch,optimizer_steps=steps,optimizer_epoch_seconds=time.perf_counter()-tick,utterances=len(rows),utterance_presentations=epoch*len(rows))
   with (RUN/'epoch_timing.csv').open('a',encoding='utf-8',newline='') as tf:
    tw=csv.DictWriter(tf,fieldnames=list(timing))
    if tf.tell()==0:tw.writeheader()
    tw.writerow(timing)
   if epoch==1""")
# Instrumentation only: monitor device-wide VRAM separately from PyTorch allocation.
code = code.replace("elif a.train:fit()", """elif a.train:
  import subprocess,threading
  stop=threading.Event()
  def monitor():
   with (RUN/'gpu_monitor.csv').open('x',encoding='utf-8',newline='') as mf:
    mw=csv.writer(mf);mw.writerow(['utc','device_used_MiB','utilization_percent'])
    while not stop.is_set():
     try:
      result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
      if result.returncode==0:mw.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())]+result.stdout.strip().split(', '));mf.flush()
     except Exception as e:mw.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'MONITOR_ERROR',type(e).__name__]);mf.flush()
     stop.wait(1.)
  watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
  try:fit()
  finally:stop.set();watcher.join(timeout=6)
""")
assert 'assert len(rows)==192' in code
compile(code, 'control192.py', 'exec')
stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
run = BASE/(stamp+'_nested192_control01')
run.mkdir(exist_ok=False)
(run/'control192.py').write_text(code, encoding='utf-8')
diff = {'source_script': str(OLD/'control.py'), 'source_sha256': hashlib.sha256((OLD/'control.py').read_bytes()).hexdigest(),
        'training_code_reused': True, 'train_size': [96,192], 'fixed_complete_epochs': 30,
        'max_seconds': 90, 'max_optimizer_steps': 720,
        'unchanged': ['architecture','features','literal_labels','vocabulary','normalization','seed','AdamW','learning_rate','weight_decay','batch4','accumulation2','gradient_clip','greedy_CTC','TRAIN_gate_thresholds'],
        'instrumentation_changes': ['nested_prepare','epoch_timer','devicewide_GPU_monitor'],
        'stop_policy': 'Complete30passes even if diagnostic gate passed sooner; same collapse rule at epoch30/resource/numerical guard; no further fit'}
(run/'PREDECLARED_DESIGN.json').write_text(json.dumps(diff, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'run': str(run), 'script': str(run/'control192.py')}, indent=2))
