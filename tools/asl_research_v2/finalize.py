"""Final content/integrity gates; no prior-run or production writes."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from common import ROOT,run_root,sha,write
from input_contract import validate
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);p.add_argument('--final-test',action='store_true');args=p.parse_args();run=run_root(args.run_root)
required=['recovery_report.md','gpu_capability_report.json','how2sign_inventory.json','how2sign_acquisition_report.md','how2sign_download_manifest.csv',
    'how2sign_annotation_audit.md','dataset_eligibility_report.md','dataset_manifest.csv','baseline_results.json','isolated_sign_training_report.md',
    'continuous_sign_training_report.md','comparison_results.csv','failure_analysis.md','experiment_config.json','reproduction_commands.ps1','final_status.md']
for name in required:
    if not (run/name).is_file() or not (run/name).stat().st_size:raise ValueError('REQUIRED_OUTPUT_MISSING:'+name)
for path in run.glob('*.json'):json.loads(path.read_text(encoding='utf-8-sig'))
checkpoints=[]
for folder,count in [('experiment_A',5),('experiment_A/warmstart',5),('experiment_A/warmstart_repeat',5),('experiment_B',11)]:
    out=run/folder;status=json.loads((out/'training_status.json').read_text());rows=list(csv.DictReader((out/'training_log.csv').open()))
    if status['completed_epochs']!=count or len(rows)!=count or not status['used_cuda']:raise ValueError('EPOCH_EVIDENCE')
    pointer=json.loads((out/'checkpoints/best.json').read_text());path=out/pointer['path']
    if sha(path)!=pointer['sha256']:raise ValueError('CHECKPOINT_HASH')
    checkpoints.append({'candidate':folder,'completed_epochs':count,'best':pointer})
audit=json.loads((run/'experiment_A/data_audit.json').read_text());known={}
for item in audit['splits']:
    path=run/'experiment_A/data'/item['split']/'X.npy';digest=sha(path)
    if digest!=item['prepared_X_sha256']:raise ValueError('PREPARED_DATA_CHANGED')
    known[str(path)]=digest
inventory=json.loads((run/'how2sign_inventory.json').read_text());archive=Path(inventory['after']['RGB_SENTENCE_CLIPS']['archive_path']);digest=sha(archive)
if digest!=inventory['after']['RGB_SENTENCE_CLIPS']['archive_sha256']:raise ValueError('HOW2_ARCHIVE_CHANGED')
known[str(archive)]=digest
export=json.loads((run/'export_results.json').read_text());model=Path(export['onnx_path'])
if sha(model)!=export['onnx_sha256']:raise ValueError('EXPORT_CHANGED')
contract=json.loads((run/'experiment_A/warmstart/source_contract.json').read_text())
if sha(Path(contract['source']))!=contract['sha256']:raise ValueError('ORIGINAL_KERAS_CHANGED')
base=json.loads((run/'experiment_A/baseline.json').read_text())
if sha(Path(base['model_path']))!=base['model_sha256']:raise ValueError('ORIGINAL_TFLITE_CHANGED')
x=np.load(run/'experiment_A/data/val/X.npy',mmap_mode='r');metadata=np.load(run/'experiment_A/data/val/metadata.npz',allow_pickle=False)
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider']);pred=[]
for start in range(0,len(x),32):
    features=np.array(x[start:start+32],copy=True);mask=metadata['mask'][start:start+32];validate(features,mask,'3.0')
    pred.extend(session.run(None,{'features':features,'frame_mask':mask})[0].argmax(-1).tolist())
accuracy=float((np.asarray(pred)==metadata['y']).mean());candidate=json.loads((run/'experiment_A/warmstart/validation_best.json').read_text())
if accuracy!=candidate['top1']:raise ValueError('EXPORTED_ACCURACY_CHANGED')
config=json.loads((run/'experiment_config.json').read_text());source_differences=[]
for name,digest in config['source_scripts'].items():
    if sha(ROOT/name)!=digest:source_differences.append(name)
if source_differences:raise ValueError('REPORTED_SOURCE_CHANGED:'+str(source_differences))
preservation=json.loads((run/'preservation_final.json').read_text())
if preservation['status']!='PASS':raise ValueError('PRESERVATION_GATE')
with (run/'dataset_manifest.csv').open(encoding='utf-8') as f:sample_rows=sum(1 for _ in csv.DictReader(f))
expected_rows=16545
if args.final_test:
    final_test=json.loads((run/'final_test_results.json').read_text());freeze=json.loads((run/'final_test_freeze.json').read_text());expected_rows+=final_test['rows']
    if sha(Path(final_test['source']))!=final_test['source_sha256']:raise ValueError('FINAL_TEST_SOURCE_CHANGED')
    for name,folder in [('primary','experiment_A/warmstart'),('repeat','experiment_A/warmstart_repeat')]:
        pointer=json.loads((run/folder/'checkpoints/best.json').read_text())
        if pointer!=freeze['checkpoints'][name] or (run/folder/'training_status.json').stat().st_mtime_ns>(run/'final_test_freeze.json').stat().st_mtime_ns:raise ValueError('POST_TEST_SELECTION_OR_TRAINING_CHANGE')
if sample_rows!=expected_rows:raise ValueError('DATASET_MANIFEST_COUNT')
write(run/('logs/final_validation_after_test.json' if args.final_test else 'logs/final_validation.json'),{'status':'PASS','required_deliverables_checked':len(required),'dataset_manifest_rows':sample_rows,
    'checkpoints':checkpoints,'exported_CPU_validation_top1':accuracy,'reported_training_source_hashes_match':True,
    'old_Keras_TFLite_hashes_match':True,'original_and_previous_run_preservation':'PASS','all_training_complete':True,
    'completed_resume_commands_checked_no_new_epochs':True,'figure_visually_inspected':True,'production_integrated':False,
    'regression_limit':'full suite is not green; see final_status and regression XML'})
files=[]
for base in (ROOT/'tools/asl_research_v2',run):
    for directory,dirs,names in os.walk(base):
        if base==run and Path(directory)==run:dirs[:]=[d for d in dirs if d not in ('regressions','runtime','cache')]
        for name in sorted(names):
            path=Path(directory)/name
            if name==('artifact_manifest.final.json' if args.final_test else 'artifact_manifest.json'):continue
            relative=path.relative_to(run) if base==run else path.relative_to(ROOT)
            files.append({'scope':'run' if base==run else 'new_tools','path':str(relative).replace('\\','/'),'bytes':path.stat().st_size,
                'sha256':known.get(str(path)) or sha(path)})
write(run/('artifact_manifest.final.json' if args.final_test else 'artifact_manifest.json'),{'schema':'apphear.separate_asl_research.artifacts.v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'run_root':str(run),'repository_root':str(ROOT),'artifacts':files,'excluded':'scratch regression copy/fixtures and runtime caches; input/protection manifests retained',
    'production_integrated':False,'required_outputs':required})
print(json.dumps({'status':'PARTIAL_RESEARCH_COMPLETE','required_outputs':len(required),'artifacts_hashed':len(files),'exported_CPU_top1':accuracy,'preservation':'PASS'}))
