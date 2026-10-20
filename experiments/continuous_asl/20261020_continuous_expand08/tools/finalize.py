"""Finalize measured results and inventories only after the main run completes."""
from collections import Counter
from datetime import datetime,timezone
import ast,csv,json
from pathlib import Path
import shutil
from core import RUN,ROOT,SOURCE,ACQUIRED,FIVE,SIX,SEVEN,read,sha

NINE=ROOT/'experiments/continuous_asl/20261020_continuous_expand09'
NEW_ROOTS=(ACQUIRED,FIVE,SIX,SEVEN,RUN,NINE)

def save(root,name,value,text=False):
 path=(root/name).resolve()
 if root not in path.parents or root not in NEW_ROOTS:raise ValueError('REPORT_OUTPUT_SCOPE')
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as stream:
  if text:stream.write(value)
  else:json.dump(value,stream,indent=2);stream.write('\n')

def preservation():
 path=FIVE/'tools/finish_mission.py';source=path.read_text(encoding='utf-8');tree=ast.parse(source)
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preservation')
 body=ast.get_source_segment(source,node).replace('def preservation():','def historical_preservation():').replace('for root in (ACQUIRED, RUN):','for root in (ACQUIRED, FIVE):')
 scope=dict(globals());exec(compile(body,str(path),'exec'),scope);result=scope['historical_preservation']()
 baseline=read(SIX/'preservation_baseline.json')
 for row in baseline['fresh_frozen_artifact_hashes']:
  p=Path(row['path'])
  if sha(p)!=row['sha256'] or p.stat().st_size!=row['size'] or p.stat().st_mtime_ns!=row['mtime_ns']:raise ValueError('FRESH_FROZEN_SOURCE_CHANGED:'+str(p))
 result['fresh_additional_frozen_hashes_and_mtimes']=len(baseline['fresh_frozen_artifact_hashes'])
 for root,name in ((RUN,'reference48'),(NINE,'reference48')):
  freeze=read(root/'model_selection_freeze.json');model=freeze['models'][name];base=root/'models'/name
  if sha(root/'data/processed/train_dev/manifest.json')!=freeze['manifest_sha256']:raise ValueError('NEW_MANIFEST_CHANGED')
  for relative,digest in [('training_status.json',model['status_sha256']),('config.json',model['config_sha256']),(model['pointer']['path'],model['pointer']['sha256'])]:
   if sha(base/relative)!=digest:raise ValueError('NEW_FROZEN_ARTIFACT_CHANGED')
 for folder in ('overfit_diagnostic','overfit_diagnostic_v2'):
  info=read(SEVEN/folder/'result.json')
  if sha(info['checkpoint_path'])!=info['checkpoint_sha256']:raise ValueError('DIAGNOSTIC_CHECKPOINT_CHANGED')
 result.update(new_frozen_models_verified=True,old14_and_isolated2620_unchanged=True,reserved17_evaluated=False,production_changed=False)
 return result

def gpu_samples(root,pattern):
 records=[]
 for path in (root/'logs').glob(pattern):
  for r in csv.reader(path.open()):
   if len(r)!=5:continue
   try:used=float(r[2]);total=float(r[3])
   except ValueError:continue
   try:util=float(r[4])
   except ValueError:util=None
   records.append({'used_mib':used,'total_mib':total,'utilization_percent':util})
 return {'samples':len(records),'interval_ms':200,'peak_whole_device_used_mib':max((r['used_mib'] for r in records),default=None),
         'peak_sampled_utilization_percent':max((r['utilization_percent'] for r in records if r['utilization_percent'] is not None),default=None),
         'utilization_unavailable_samples':sum(r['utilization_percent'] is None for r in records),
         'scope':'Queries name,memory.used,memory.total,utilization; whole-device occupancy,200ms sampling can miss short peaks. Separate PyTorch allocator peaks are authoritative for the framework.'}

def acquisition():
 four_bytes=sum(p.stat().st_size for p in (ACQUIRED/'data/raw').rglob('*') if p.is_file() and not p.name.endswith(('.receipt.json','.http.json')))
 receipts=[read(p) for p in (SIX/'data/raw').rglob('*.receipt.json')]
 unique={r['source_url']:r for r in receipts}
 acquired=1895291702+four_bytes+sum(r['bytes'] for r in unique.values())
 cap=read(SIX/'acquisition_limits.v2.json')['total_acquired_payload_cap_bytes']
 if acquired>cap or shutil.disk_usage(ROOT).free<10737418240:raise ValueError('ACQUISITION_CAP_OR_DISK_HEADROOM')
 return {'measured_local_acquired_payload_bytes_including_prior_How2Sign':acquired,'authorized_cap_bytes':cap,
         'remaining_cap_bytes':cap-acquired,'expand06_unique_download_urls':len(unique),'expand06_download_bytes':sum(r['bytes'] for r in unique.values()),
         'extraction_not_recounted_as_network_download':True,'free_disk_bytes':shutil.disk_usage(ROOT).free,
         'scope':'Downloaded payloads/receipts; excludes request headers, network overhead and local derived/extracted copies.'}

def inventory(root):
 path=root/'artifact_manifest.json'
 if path.exists():raise ValueError('ARTIFACT_INDEX_ALREADY_EXISTS_NO_OVERWRITE:'+str(root))
 files=[]
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if not p.is_file() or p.name=='artifact_manifest.json' or any(x in ('cache','__pycache__','.pytest_cache') for x in rel.parts):continue
  files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(root,'artifact_manifest.json',{'schema':'apphear.continuous_asl.artifacts.v1','created_utc':datetime.now(timezone.utc).isoformat(),
      'files':files,'excludes':['self','cache/**','**/__pycache__/**','**/.pytest_cache/**'],'production_enabled':False})
 return len(files)

def main():
 if (RUN/'execution_summary.json').exists():raise ValueError('FINAL_REPORT_ALREADY_CREATED_NO_OVERWRITE')
 comparison=read(RUN/'comparison_dev.json');canonical=read(SIX/'canonical/quality.json');pilot=read(NINE/'pilot_outcome.json')
 protected=preservation();budget=acquisition();status=read(RUN/'models/reference48/training_status.json');plan=read(RUN/'dataset_plan.json')
 candidate=comparison['candidate'];baseline=comparison['frozen_baseline'];diag=read(SEVEN/'overfit_diagnostic_v2/result.json')
 selected_train=read(RUN/'selected_checkpoint_train_diagnostic.json');last_train=read(RUN/'last_checkpoint_train_diagnostic.json')
 for evidence in (selected_train,last_train):
  if evidence['selection_freeze_sha256']!=sha(RUN/'model_selection_freeze.json'):raise ValueError('TRAIN_DIAGNOSTIC_SELECTION_CHANGED')
 if sha(last_train['checkpoint_path'])!=last_train['checkpoint_sha256']:raise ValueError('LAST_TRAIN_CHECKPOINT_CHANGED')
 corpus=read(SIX/'canonical/manifest.json');fit=read(RUN/'data/processed/train_dev/manifest.json')
 current_ids={r['sample_id'] for r in fit if r['split']=='train'}
 pilot_ids={r['sample_id'] for r in read(NINE/'data/processed/train_dev/manifest.json') if r['split']=='train'}
 counts=canonical['stage_counts'];total_eligible=canonical['eligible_train_total_across_separate_pools'];primary=plan['fit_counts']['train']
 source_cache=[read(p) for p in (SIX/'data/processed/source_cache').glob('*.raw.json')]
 counts.update(new_ncslgr_media_fully_decoded_with_saved_evidence=len(source_cache),
               unique_TRAIN_consumed_in_main=primary,unique_TRAIN_consumed_in_906pilot=pilot['train'],
               unique_eligible_TRAIN_consumed_union_in_906pilot_and_main=len(current_ids|pilot_ids),
               invalid_v1_diagnostic_only_consumed_not_counted=read(SEVEN/'overfit_gate_adjudication.v2.json')['excluded_diagnostic_ids'])
 summary={'created_utc':datetime.now(timezone.utc).isoformat(),
  'execution_status':'PASS' if primary>=2000 else 'PARTIAL_TARGET_SHORTFALL','recognition_readiness':'PARTIAL_RESEARCH_ONLY',
  'canonical':canonical,'stage_counts':counts,'main_fit_counts':plan['fit_counts'],'reserved_test_untouched':17,'new_test_evaluated':0,
  'main_train_manifest_sha256':plan['manifest_sha256'],'main_training':status,'main_comparison':comparison,'bounded906pilot':pilot,
  'main_train_diagnostics':{'selected_epoch':candidate['selected_epoch'],'selected_checkpoint_full_train_wer':selected_train['result']['gloss_wer'],
     'last_epoch':last_train['epoch'],'last_checkpoint_full_train_wer':last_train['result']['gloss_wer'],
     'last_checkpoint_inference_seconds':last_train['seconds'],'scope':'Post-selection TRAIN inference only; no refit, reselection or TEST evaluation.'},
  'clean_overfit_diagnostic':diag,'v1_diagnostic_eligibility_adjudication':read(SEVEN/'overfit_gate_adjudication.v2.json'),
  'minimum_target':2000,'stretch_target':5000,'unique_eligible_across_separate_pools':total_eligible,
  'minimum_shortfall':max(0,2000-total_eligible),'stretch_shortfall':max(0,5000-total_eligible),
  'single_main_pool_minimum_shortfall':max(0,2000-primary),'single_main_pool_stretch_shortfall':max(0,5000-primary),
  'acquisition_budget':budget,'main_gpu_device_samples':gpu_samples(RUN,'gpu_samples_*.csv'),
  'pilot_gpu_device_samples':gpu_samples(NINE,'gpu_v2_*.csv'),'preservation':protected,
  'source_status_snapshot':str(SIX/'source_status.json'),'source_status_final':'Official ASLLRP archive downloaded/CRC verified/joins verified; features and canonical exclusions complete.',
  'boundary_and_streaming_status':{'publisher_bounds':'Available and preserved; six ASLLRP overlapping utterances excluded without trimming.',
     'independently_reviewed_visual_alignment':'NOT_VERIFIED','signer_independent_generalization':'NOT_ESTABLISHED',
     'consented_nonsign_recordings':'BLOCKED_NO_ADDITIONAL_AUTHORIZED_DATA','false_activations':'NOT_MEASURED',
     'missed_sign_events':'NOT_MEASURED; WER deletions are not detection-event misses','boundary_event_F1':'NOT_MEASURED',
     'normal_Android_end_to_end_latency_and15FPS':'NOT_MEASURED; research sampling max10Hz; model-only timing cannot pass this gate'},
  'production_promoted':False,'limitations':['Fixed9DEV/57tokens; cross-corpus/version diagnostic for ASLLRP3, incomplete signer mapping.',
     'ASLLRP3 targets cover dominant-realized source events including hidden events, not all two-hand/nonmanual ASL structure.',
     'No exhaustive perceptual duplicate or semantic-equivalence detector; exact hashes and stable source/timing identities audited.',
     'Tiny overfit/TRAIN loss improvement is not generalization. Compare measured WER against frozen baseline.'],
  'next_action':'Use read-only Validate-Completed.ps1. For stronger recognition claims, collect/release authorized signer-disjoint ASLLRP-compatible evaluation plus consented nonsign exposure under the unchanged existing protocol.'}
 save(RUN,'preservation_final.json',protected);save(RUN,'execution_summary.json',summary)
 lines=['# Executed continuous-ASL expansion and training','',
   f"**Execution {summary['execution_status']}; recognition readiness PARTIAL.**",'',
   f"Canonical corpus: {canonical['unique_records']} verified unique records; {total_eligible} training-eligible utterances across separate publisher label pools. Main fit consumed {primary} unique TRAIN utterances and used the exact original9DEV;17 reserved TEST utterances remain untouched.",'',
   '| Run | TRAIN | Epochs / selected | DEV WER | S / D / I | Fit and reload seconds | Peak PyTorch MiB |','|---|---:|---:|---:|---:|---:|---:|',
   f"| Frozen expand03 reference |100|14 /4|{100*baseline['gloss_wer']:.2f}%|{baseline['substitutions']}/{baseline['deletions']}/{baseline['insertions']}|5.7732|68.08|",
   f"| Bounded expand09 pilot |906|{pilot['epochs']} /{pilot['selected_epoch']}|{100*pilot['dev_best']['gloss_wer']:.2f}%|{pilot['dev_best']['substitutions']}/{pilot['dev_best']['deletions']}/{pilot['dev_best']['insertions']}|{pilot['fit_reload_seconds']:.4f}|{pilot['peak_allocated_vram_bytes']/2**20:.2f}|",
   f"| Main expand08 |{primary}|{candidate['completed_epochs']} /{candidate['selected_epoch']}|{100*candidate['gloss_wer']:.2f}%|{candidate['substitutions']}/{candidate['deletions']}/{candidate['insertions']}|{candidate['measured_fit_reload_seconds']:.4f}|{candidate['peak_pytorch_allocated_vram_bytes']/2**20:.2f}|",'',
   f"Main checkpoint: `{candidate['checkpoint_path']}`",f"SHA-256: `{candidate['checkpoint_sha256']}`",f"TRAIN/DEV manifest SHA-256: `{plan['manifest_sha256']}`",'',
   f"The main candidate {'improved' if comparison['beats_frozen_baseline'] else 'did not improve'} the frozen reference on identical9DEV. This is a limited57-reference-token cross-corpus diagnostic, not an architecture-only comparison or signer-independent benchmark. DEV OOV: {candidate['out_of_vocabulary_tokens']}/{candidate['reference_tokens']}. No experimental model is promoted.",'',
   f"Clean diagnostic v2: {100*diag['best_train']['gloss_wer']:.2f}% TRAIN WER on8 eligible utterances; loss {diag['initial_train']['loss']:.4f}→{diag['best_train']['loss']:.4f};{diag['completed_epochs']} optimizer updates; checkpoint reload matched. Numerical v1 is retained but excluded as a gate because one source utterance overlapped another publisher span.",'',
   f"906-pilot final-epoch full TRAIN WER: {100*pilot['final_epoch_train_wer']:.2f}%; DEV-selected checkpoint TRAIN WER: {100*pilot['selected_checkpoint_train_wer']:.2f}%. Selection stayed on original9DEV. Optimization worked while generalization remained poor.",'',
   f"Main final epoch{last_train['epoch']} full TRAIN WER: {100*last_train['result']['gloss_wer']:.2f}%; DEV-selected epoch{candidate['selected_epoch']} full TRAIN WER: {100*selected_train['result']['gloss_wer']:.4f}%. These separately saved, post-selection TRAIN diagnostics did not change checkpoint selection or score TEST. Final-epoch inference took {last_train['seconds']:.4f}s in addition to the147.0251s fit/reload. The later checkpoint learned TRAIN while DEV selected an early blank-output checkpoint. The2739-label vocabulary includes1331 singleton labels and6/57DEV tokens unseen in TRAIN; cross-corpus label and distribution differences limit this comparison, without establishing a unique cause of regression.",'',
   f"Main median epoch: {candidate['epoch_median_seconds']:.4f}s; observed TRAIN throughput including epoch validation/checkpoint work: {candidate['train_utterances_per_complete_epoch_second']:.2f} utterances/s and {candidate['train_observations_per_complete_epoch_second']:.2f} observations/s. CPU model-only p50/p95: {candidate['cpu_model_p50_ms']:.4f}/{candidate['cpu_model_p95_ms']:.4f}ms per complete saved feature sequence,2threads,180measurements. These exclude perception/capture/UI and do not establish per-frame streaming latency, phone FPS or end-to-end latency.",'',
   f"Progress: {total_eligible}/2,000 eligible across documented separate pools; stretch5,000 shortfall {summary['stretch_shortfall']}. One internally consistent main fit used {primary}; its own5,000 shortfall is {max(0,5000-primary)}. No alternative views, feature copies, isolated signs or English translations inflate counts.",'',
   'Stage counters, exact bounds, vocabulary/normalization, environment/seed, full predictions, GPU samples, failures and hashes are in execution_summary.json, canonical manifest/quality, training_plan.v2.json and models/reference48. Per-epoch wall times include validation/checkpoint work. Command wall times are separate in logs; fit/reload timing also includes explicitly saved post-loop diagnostics where present. CUDA CTC backward warns that fixed seeds do not guarantee bitwise determinism.','',
   'The six ASLLRP overlapping utterances were rejected without trimming; four were stopped at metadata before completion and two cached features retained but excluded. Workers were reassigned before primary/tail catch-up, then balanced into disjoint modulo4 shards using measured clip runtimes.854 and then1115 completed feature hashes passed, with no incomplete output found. Interrupted unfinished work was resumed; completed extraction was reused. A pre-optimization smoke monitor error was preserved and fixed in a new smoke_v2 output; the906 main snapshot did not change.','',
   'How2Sign: existing archive verification reused; available inspected labels are English translations, so no fabricated gloss training. Prior failed URL `https://drive.google.com/uc?export=download&id=1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1`, errorHTTP_404 at2026-10-19T16:53:14.5183865Z remains preserved. Cokely official EAF URLs and Gallaudet OSF transcript endpoints returnedHTTP403; no bypass or unlabeled bulk download followed.','',
   'ASLLRP/NCSLGR media stay local under research/education-only terms; no redistribution or commercial deployment. Attribution: [ASLLRP resources](https://www.bu.edu/asllrp/ASL-SignBank-and-other-Resources.html), [publisher download documentation](https://www.bu.edu/asllrp/rpt24/asllrpr24.pdf), [terms](https://www.bu.edu/asllrp/dai-terms.html), [official DePaul ELAN conversion](https://asl.cs.depaul.edu/corpus/index.html). Literal source label conventions remain separate; CSV occurrence labels are not English translations.','',
   'Streaming qualification remains blocked without verified signer-disjoint evaluation, independently reviewed boundaries and consented nonsign recordings. Reuse the unchanged acceptance_protocol.md: paired WER improvement≥5points with signer-aware uncertainty; negative-exposure upper false-commit rate≤0.1/min; miss upper bound≤10%; eventF1≥0.90 and boundaryp95≤200ms; normal-phone deliveryp95≤1000ms and completed15FPS/5min. These are proposed criteria, not measured passes.','',
   'Original frozen results, checkpoints, production thresholds, Android application/GLB assets and prior reports were preserved. No APK was installed. Exact safe validation and completed-run verification commands are in REPRODUCE.ps1. The completed comparison is unchanged; comparison_provenance.json distinguishes its executed version from an unexecuted additive diagnostic edit.']
 save(RUN,'execution_report.md','\n'.join(lines)+'\n',True)
 for root in (ACQUIRED,FIVE):
  c=read(root/'comparison_dev.json')
  save(root,'completion_addendum.20261020.json',{'status':'COMPLETED_PREVIOUSLY_REUSED_NO_REFIT','comparison':c,
       'new_reserved17_test_evaluated':False,'current_mission_report':str(RUN/'execution_report.md'),'production_promoted':False})
 save(SIX,'source_summary.final.json',{'canonical':canonical,'stage_counts':counts,'budget':budget,'all_sources_local_no_redistribution':True})
 save(SEVEN,'diagnostic_summary.final.json',{'eligible_v2':diag,'v1_adjudication':read(SEVEN/'overfit_gate_adjudication.v2.json'),'warm_start_draft_executed':False})
 indexed={root.name:inventory(root) for root in NEW_ROOTS}
 print(json.dumps({'status':summary['execution_status'],'readiness':'PARTIAL_RESEARCH_ONLY','main_train':primary,'eligible_total':total_eligible,
                   'dev_wer':candidate['gloss_wer'],'artifact_index_counts':indexed}),flush=True)

if __name__=='__main__':main()
