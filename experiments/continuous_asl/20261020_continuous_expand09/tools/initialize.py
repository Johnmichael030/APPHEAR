"""Reuse frozen-manifest verification code; port unchanged cold-fit loop."""
from collections import Counter,defaultdict
import ast,json
from pathlib import Path
import numpy as np
from core import RUN,ROOT,SOURCE,FIVE,SIX,SEVEN,SCHEMA,read,write,sha

def main():
 if (RUN/'dataset_plan.json').exists():
  if sha(RUN/'data/processed/train_dev/manifest.json')!=read(RUN/'dataset_plan.json')['manifest_sha256']:raise ValueError('FROZEN_MANIFEST_CHANGED')
  print('PILOT_SNAPSHOT_ALREADY_FROZEN');return
 gate=read(SEVEN/'overfit_diagnostic_v2/result.json')
 if gate['status']!='PASS_LEARNABILITY_GATE' or gate['best_train']['gloss_wer']>.05 or gate['loss_ratio']>.25 or not gate['checkpoint_reload_predictions_match']:raise ValueError('CLEAN_OVERFIT_GATE')
 if sha(gate['checkpoint_path'])!=gate['checkpoint_sha256']:raise ValueError('CLEAN_OVERFIT_CHECKPOINT_HASH')
 blocked=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
 if blocked&{r['sample_id'] for r in read(SEVEN/'overfit_diagnostic_v2/manifest.json')}:raise ValueError('CLEAN_OVERFIT_ELIGIBILITY')
 # Reuse exactly the already reviewed timed-manifest verification prelude.
 path=SEVEN/'tools/initialize.py';text=path.read_text(encoding='utf-8');tree=ast.parse(text)
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 body=ast.get_source_segment(text,function)
 prefix=body[:body.index("    plan=read(FIVE/'training_plan.v2.json')")]
 namespace=dict(globals());exec(compile(prefix+'\n',str(path),'exec'),namespace);namespace['main']()
 rows=read(RUN/'data/processed/train_dev/manifest.json');fit=[r for r in rows if r['split']=='train']
 groups=defaultdict(list)
 for row in fit:groups[row['collection']].append(row)
 for group in groups.values():group.sort(key=lambda r:sha_string(r['sample_id']))
 monitor=[]
 while len(monitor)<min(64,len(fit)):
  for key in sorted(groups):
   if groups[key] and len(monitor)<64:monitor.append(groups[key].pop(0)['sample_id'])
 plan=read(FIVE/'training_plan.v2.json')
 plan.update(models={'smoke':{'channels':48,'max_epochs':1,'wall_seconds':60},'reference48':{'channels':48,'max_epochs':30,'wall_seconds':120}},
             task='Literal authentic NCSLGR2 main-gloss events on906 verified TRAIN utterances',
             selection='Exact original9DEV WER then loss; max30epochs/120seconds/patience10',
             normalization='This frozen906TRAIN only',train_monitor_ids=monitor,
             overfit_gate_sha256=sha(SEVEN/'overfit_diagnostic_v2/result.json'),protocol_sha256=sha(RUN/'research_protocol.md'))
 write(RUN/'training_plan.v2.json',plan)
 write(RUN/'dataset_plan.json',{'fit_counts':dict(Counter(r['split'] for r in rows)),'manifest_sha256':sha(RUN/'data/processed/train_dev/manifest.json'),
       'reused_train':455,'new_timed_train':451,'reserved_test':17,'test_accessed':False,'signer_independent':False})
 source_path=FIVE/'tools/train_expanded.py';source=source_path.read_text(encoding='utf-8')
 changes=[('SOURCE, ACQUIRED, OLD, SCHEMA','SOURCE, ACQUIRED, FIVE, SIX, OLD, SCHEMA'),
   ("quality['source_media_hash_duplicates']","quality['invalid_source_overlap']"),
   ('or ACQUIRED in path.parents or OLD in path.parents','or ACQUIRED in path.parents or FIVE in path.parents or SIX in path.parents or OLD in path.parents'),
   ("if name=='compact24' and not (RUN/'models/reference48/training_status.json').is_file(): raise ValueError('REFERENCE_MUST_BE_FROZEN_FIRST')",''),
   ("choices=('smoke','reference48','compact24')","choices=('smoke','reference48')"),
   ("    start=time.perf_counter(); status='MAX_EPOCHS'; torch.cuda.reset_peak_memory_stats()", """    monitor=[r for r in fit if r['sample_id'] in config['train_monitor_ids']]
    if not (out/'initial_train_monitor.json').exists():write(out/'initial_train_monitor.json',evaluate(model,monitor,data,vocabulary,'cuda:0'))
    start=time.perf_counter(); status='MAX_EPOCHS'; torch.cuda.reset_peak_memory_stats()"""),
   ("            result=evaluate(model,dev,data,vocabulary,'cuda:0');torch.cuda.synchronize()", "            result=evaluate(model,dev,data,vocabulary,'cuda:0');train_monitor=evaluate(model,monitor,data,vocabulary,'cuda:0');torch.cuda.synchronize()"),
   ("record={'epoch':epoch,'train_ctc_loss':float(np.mean(losses))", "record={'epoch':epoch,'train_monitor_wer':train_monitor['gloss_wer'],'train_monitor_ctc_loss':train_monitor['loss'],'train_ctc_loss':float(np.mean(losses))"),
   ("        # Reload the new checkpoint", "        write(out/'last_epoch_train.json',evaluate(model,fit,data,vocabulary,'cuda:0'))\n        write(out/'last_epoch_dev.json',evaluate(model,dev,data,vocabulary,'cuda:0'))\n        # Reload the new checkpoint"),
   ("        write(out/'training_status.json'", "        write(out/'selected_checkpoint_train.json',evaluate(model,fit,data,vocabulary,'cuda:0'))\n        write(out/'training_status.json'"),
   ("'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__))", "'unique_train_utterances_consumed':len(fit),'train_utterance_presentations':len(fit)*completed,'train_observations_per_complete_epoch':sum(len(data[r['sample_id']]) for r in fit),'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__))")]
 for before,after in changes:
  if before not in source:raise ValueError('PILOT_PORT_ANCHOR:'+before)
  source=source.replace(before,after)
 with (RUN/'tools/train_pilot.py').open('x',encoding='utf-8') as f:f.write(source)
 write(RUN/'code_provenance.json',{'source_sha256':sha(source_path),'verification_prelude_sha256':sha(path),
       'trainer_sha256':sha(RUN/'tools/train_pilot.py'),'changes':'New906 snapshot/paths; unchanged model and optimizer math; stratifiedTRAIN monitoring and last/selected fullTRAIN diagnostics added.'})
 print(json.dumps({'status':'PASS_PILOT_FROZEN','train':len(fit),'dev':9,'monitor':len(monitor)}),flush=True)

def sha_string(s):
 import hashlib
 return hashlib.sha256(('20261020:'+s).encode()).hexdigest()

if __name__=='__main__':main()
