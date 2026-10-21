"""Retain pre-test report snapshots, then add the frozen test result explicitly."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from common import run_root,write
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
test=json.loads((run/'final_test_results.json').read_text());base=test['baseline'];primary=test['candidates']['primary'];repeat=test['candidates']['repeat']
snapshot=run/'logs/pre_final_test_report_snapshot';snapshot.mkdir()
for name in ('final_status.md','isolated_sign_training_report.md','recovery_report.md','baseline_results.json','comparison_results.csv','dataset_manifest.csv','experiment_config.json','acceptance_status.json','artifact_manifest.json'):
    shutil.copyfile(run/name,snapshot/name)
paragraph=f'''\n\nFinal native-test evaluation was added after training/checkpoint/threshold decisions were frozen. On {test['rows']} rows from {test['signer_count']} test signer IDs, disjoint from TRAIN/VAL: frozen FP32 top1 {base['top1']*100:.2f}% / macro-F1 {base['macro_f1']*100:.2f}%; primary {primary['top1']*100:.2f}% / {primary['macro_f1']*100:.2f}%; repeat {repeat['top1']*100:.2f}% / {repeat['macro_f1']*100:.2f}%. Primary gains {100*(primary['top1']-base['top1']):.2f} top1 points (35 additional correct rows), repeat {100*(repeat['top1']-base['top1']):.2f} points (25 additional correct rows). No post-test training or selection occurred. This is a legacy258 isolated-sign benchmark; current live-app/negative/boundary behavior remains unmeasured. See final_test_report.md and final_test_freeze.json.\n'''
path=run/'isolated_sign_training_report.md';path.write_text(path.read_text()+paragraph)
path=run/'recovery_report.md';text=path.read_text().replace('No TEST tensors loaded for model work.','No TEST tensors used for training or selection; the final native test was opened only after the recorded freeze.');path.write_text(text+paragraph)
path=run/'final_status.md';text=path.read_text();text=text.replace('not independent final-test or live-app evidence.','validation evidence, with the later frozen final-test results reported below; no live-app claim.')
text=text.replace('independent final evaluation and more signer clusters','current-path independent final evaluation and more signer clusters')
text=text.replace('Validate the modest isolated gain on a separately reserved benchmark before considering any adapter/integration.','The isolated gain also persisted on the subsequently frozen native test; a compatible current-path streaming benchmark is still required before any adapter/integration.')
text=text.replace('Acceptance statuses:',paragraph+'\nAcceptance statuses:')
text+='\nArtifact integrity: artifact_manifest.final.json is authoritative after final evaluation. artifact_manifest.json and logs/pre_final_test_report_snapshot retain pre-test evidence. Final-test results were never used to choose a different seed/checkpoint.\n'
path.write_text(text)
baseline=json.loads((run/'baseline_results.json').read_text());baseline['native_final_test_after_freeze']=test;baseline['test_data_used']=True;baseline['test_use']='FINAL_EVALUATION_ONLY_NO_TRAINING_SELECTION_OR_CALIBRATION';(run/'baseline_results.json').write_text(json.dumps(baseline,indent=2))
config=json.loads((run/'experiment_config.json').read_text());config['test_payload_used_for_model_work']='FINAL_EVALUATION_ONLY';config['test_payload_used_for_training_or_selection']=False
config['final_test']={'freeze':'final_test_freeze.json','results':'final_test_results.json','primary_preselected_before_test':True,'post_test_training':False};(run/'experiment_config.json').write_text(json.dumps(config,indent=2))
acceptance=json.loads((run/'acceptance_status.json').read_text());acceptance['Accuracy improvement against matched legacy final-test baseline']='PASS_PRIMARY_AND_REPEAT';acceptance['Current production live-path accuracy improvement']='NOT MEASURED';(run/'acceptance_status.json').write_text(json.dumps(acceptance,indent=2))
with (run/'comparison_results.csv').open('a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('task','candidate','metric','value','comparison','status'))
    for name,result in [('frozen_FP32',base),('preselected_primary',primary),('independent_repeat',repeat)]:
        for metric in ('top1','macro_f1'):w.writerow(dict(task='isolated_gloss_native_final_test',candidate=name,metric=metric,value=result[metric],comparison='2620 native rows, 11 held-out signer IDs, frozen selection',status='FINAL_TEST_EVALUATION_ONLY'))
fields=('sample_id','sample_kind','source','source_path','modality','label_type','label','signer','split','duration_seconds','duration_status','license_notes','validation_status','acquisition_status')
inverse={int(v):k for k,v in json.loads((run/'experiment_A/vocabulary.json').read_text()).items()}
with np.load(test['source'],allow_pickle=False) as data,(run/'dataset_manifest.csv').open('a',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields)
    for i,(label,signer) in enumerate(zip(data['y'],data['participant_ids'])):
        w.writerow(dict(sample_id=f'canonical_native_final_test_feature_row_{i:05d}',sample_kind='DERIVED_FEATURE_ROW_NOT_RAW_CLIP_ID',source='ASL Citizen-derived canonical bundle',source_path=test['source'],
            modality='LEGACY_NORMALIZED258_NATIVE',label_type='ISOLATED_ASL_GLOSS',label=inverse[int(label)],signer=str(signer),split='FINAL_TEST_OPENED_AFTER_FREEZE',duration_seconds='',duration_status='UNKNOWN_NO_ORIGINAL_TIMESTAMPS_IN_NPZ',
            license_notes='Microsoft Research License private noncommercial research',validation_status='HASH_SCHEMA_FINITE_MASK_AND_DISJOINT_SIGNERS_VERIFIED; no post-test model changes',acquisition_status='EXISTING_FINAL_EVALUATION_ONLY'))
interval=test['paired_signer_bootstrap_top1_delta_95pct']['primary']
report=f'''# Frozen native final-test evaluation

Training, augmentation, checkpoint and threshold decisions were frozen in final_test_freeze.json before opening the final tensor payload. Primary checkpoint remains the predeclared seed20261020 epoch4; repeat seed20261021 is replication evidence, not a winner selected using test outcomes. No further training or threshold/model-selection changes followed.

Source: `{test['source']}`; SHA256 `{test['source_sha256']}`. 2,620 native-cadence isolated feature rows, 236-class schema3.0, 11 signer IDs; zero TRAIN/VAL signer overlap. Whole feature sequences/masks were preserved. No English-translation or CTC metric is mixed into this comparison. Source IDs/timing for raw clips are absent from the canonical NPZ, so raw lineage and repeated-view independence are not overclaimed.

| Frozen comparison | Top1 | Macro-F1 |
|---|---:|---:|
| Unchanged legacy FP32 baseline | {base['top1']*100:.2f}% | {base['macro_f1']*100:.2f}% |
| Preselected primary fine-tune | {primary['top1']*100:.2f}% | {primary['macro_f1']*100:.2f}% |
| Independent training-seed repeat | {repeat['top1']*100:.2f}% | {repeat['macro_f1']*100:.2f}% |

Primary paired signer-bootstrap top1-difference95% interval: [{interval[0]*100:.2f}, {interval[1]*100:.2f}] percentage points,2,000 resamples of11signer clusters. This is descriptive evidence on the available benchmark; it does not establish all recording conditions or live-product performance. No checkpoint was promoted. False activations, false rejection, current350-feature runtime, boundaries, deployment rights and Android device gates remain unresolved.

Complete predictions, per-class precision/recall/F1/confusion and per-signer outcomes: experiment_A/final_test/baseline_fp32.json, primary.json, repeat.json. Final evaluation command (already completed; do not rerun into this run):

```powershell
& '{run.parent/'20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'}' -B '{Path(__file__).with_name('final_test.py')}' --run-root '{run}'
```

The original final-test NPZ hash and modification time were checked before and after evaluation; neither changed. The staged test X.npy is read-only mapped. Model checkpoint hashes were verified against the pre-test freeze.
'''
with (run/'final_test_report.md').open('x') as f:f.write(report)
print(json.dumps({'final_test_report_added':True,'primary_gain_pp':100*(primary['top1']-base['top1']),'repeat_gain_pp':100*(repeat['top1']-base['top1'])}))
