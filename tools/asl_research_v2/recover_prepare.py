"""Verify old evidence, protect pre-existing files, and stage only eligible TRAIN/VAL."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from common import ROOT,PREVIOUS,SOURCE,sha,run_root,write
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
original=json.loads((PREVIOUS/'preservation_baseline.json').read_text());prior=json.loads((PREVIOUS/'reproducibility_manifest.json').read_text())
bad=[];protected=[]
for item in original['files']:
    path=ROOT/item['path']
    if not path.is_file() or sha(path)!=item['sha256']:bad.append(str(path))
    protected.append({'path':str(path),'sha256':item['sha256'],'mtime_ns':path.stat().st_mtime_ns})
for item in prior['files']:
    path=(PREVIOUS if item['scope']=='run' else ROOT)/item['path']
    if not path.is_file() or sha(path)!=item['sha256']:bad.append(str(path))
    protected.append({'path':str(path),'sha256':item['sha256'],'mtime_ns':path.stat().st_mtime_ns})
protected.append({'path':str(PREVIOUS/'reproducibility_manifest.json'),'sha256':sha(PREVIOUS/'reproducibility_manifest.json'),'mtime_ns':(PREVIOUS/'reproducibility_manifest.json').stat().st_mtime_ns})
write(run/'protected_sources.json',protected)
write(run/'logs/recovery_integrity.json',{'original_files':len(original['files']),'prior_artifacts':len(prior['files']),'mismatches':bad,'status':'PASS' if not bad else 'FAIL'})
if bad:raise RuntimeError('PRESERVATION_FAILED')
plan=json.loads((SOURCE/'data/DATASET_MANIFEST.json').read_text())['files'];datasets=[];signers={};ids_by_split={}
for split in ('train','val'):
    src=SOURCE/f'data/train/{split}.npz';expected=plan[f'data/train/{split}.npz']
    if src.stat().st_size!=expected['bytes'] or sha(src)!=expected['sha256']:raise ValueError('A_DATA_HASH')
    out=run/'experiment_A/data'/split;out.mkdir(parents=True)
    with np.load(src,allow_pickle=False) as data:
        schema=str(data['schema_version'].item());y=data['y'];mask=data['frame_mask'];lengths=data['lengths'];ids=data['participant_ids']
        if schema!='3.0' or len(set(y.tolist()))!=236 or y.min()!=0 or y.max()!=235:raise ValueError('A_SCHEMA_LABELS')
        if not np.array_equal(mask.sum(1),lengths):raise ValueError('A_LENGTHS')
        signers[split]=set(ids.tolist());ids_by_split[split]=ids
        np.savez_compressed(out/'metadata.npz',y=y,mask=mask,lengths=lengths,participant_ids=ids,schema_version=np.asarray(schema))
    # One verified member to a fresh output; stream 1.31GB into a CPU mmap, never VRAM.
    target=out/'X.npy'
    with zipfile.ZipFile(src) as archive, archive.open('X.npy') as inp,target.open('xb') as dest:
        shutil.copyfileobj(inp,dest,length=1048576)
    x=np.load(target,mmap_mode='r',allow_pickle=False)
    if x.shape!=(len(y),96,258) or x.dtype!=np.float32:raise ValueError('A_TENSOR_SHAPE')
    nonfinite=0
    for start in range(0,len(x),128):nonfinite+=int(np.count_nonzero(~np.isfinite(x[start:start+128])))
    if nonfinite:raise ValueError('A_NONFINITE')
    datasets.append({'split':split,'rows':len(y),'signers':len(signers[split]),'classes':236,
        'class_count_min':int(np.bincount(y).min()),'class_count_max':int(np.bincount(y).max()),
        'source':str(src),'source_sha256':sha(src),'prepared_X_sha256':sha(target),'prepared_X_bytes':target.stat().st_size,
        'label_type':'ISOLATED_ASL_GLOSS','schema_version':'3.0','feature_shape':[96,258],
        'provenance':'ASL Citizen-derived canonical 236-sign bundle; no local recorder data used',
        'license':'Microsoft Research License; private noncommercial nonrevenue research; no redistribution',
        'feature_lineage_limitation':'legacy normalized258, not current Candidate01/RecognitionV2 raw350'})
    print(json.dumps({'prepared':split,'rows':len(y),'bytes':target.stat().st_size}),flush=True)
if signers['train']&signers['val']:raise ValueError('A_SIGNER_LEAKAGE')
vocab=SOURCE/'data/vocab/vocab.json'
if sha(vocab)!=plan['data/vocab/vocab.json']['sha256']:raise ValueError('A_VOCAB_HASH')
shutil.copyfile(vocab,run/'experiment_A/vocabulary.json')
for path in (SOURCE/'data/train/train.npz',SOURCE/'data/train/val.npz',vocab,Path('D:/ASL_Citizen/ASL_Citizen/use.txt')):
    protected.append({'path':str(path),'sha256':sha(path),'mtime_ns':path.stat().st_mtime_ns})
write(run/'external_inputs_protected.json',protected[-4:])
write(run/'experiment_A/data_audit.json',{'status':'PASS','splits':datasets,'signer_overlap':0,'test_payload_opened':False,
    'permissions_review':'local use.txt and official project checked; research only','augmentation':'none in first bounded run'})
print('RECOVERY_AND_A_DATA_PASS',flush=True)
