"""One frozen native test evaluation, after every model-selection decision is complete."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch
from common import SOURCE,run_root,write,sha
from warmstart_a import LegacyTCN,evaluate
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
primary=run/'experiment_A/warmstart';repeat=run/'experiment_A/warmstart_repeat'
selection={name:json.loads((folder/'checkpoints/best.json').read_text()) for name,folder in [('primary',primary),('repeat',repeat)]}
write(run/'final_test_freeze.json',{'primary_candidate':'warmstart_seed20261020','checkpoints':selection,
    'test_partition':'data/test_native/test.npz','selection_frozen_before_test_open':True,
    'no_more_training_or_threshold_changes':True,'repeat_is_replication_not_selected_using_test':True})
source=SOURCE/'data/test_native/test.npz';manifest=json.loads((SOURCE/'data/DATASET_MANIFEST.json').read_text())['files']['data/test_native/test.npz']
source_hash=sha(source);source_mtime=source.stat().st_mtime_ns
if source_hash!=manifest['sha256'] or source.stat().st_size!=manifest['bytes']:raise ValueError('TEST_SOURCE_HASH')
with np.load(source,allow_pickle=False) as f:
    metadata={k:f[k].copy() for k in ('y','frame_mask','lengths','participant_ids','schema_version')}
if str(metadata['schema_version'].item())!='3.0':raise ValueError('TEST_SCHEMA')
if not np.array_equal(metadata['frame_mask'].sum(1),metadata['lengths']):raise ValueError('TEST_MASK')
test_ids=set(metadata['participant_ids'].tolist());overlaps={}
for split in ('train','val'):
    with np.load(run/'experiment_A/data'/split/'metadata.npz',allow_pickle=False) as f:overlaps[split]=sorted(test_ids & set(f['participant_ids'].tolist()))
if any(overlaps.values()):raise ValueError('TEST_SIGNER_OVERLAP:'+str(overlaps))
out=run/'experiment_A/final_test';out.mkdir();target=out/'X.npy'
with zipfile.ZipFile(source) as z:
    info=z.getinfo('X.npy')
    if info.file_size>1073741824:raise ValueError('TEST_STAGING_1GIB_BUDGET')
    with z.open(info) as inp,target.open('xb') as dst:shutil.copyfileobj(inp,dst,1048576)
x=np.load(target,mmap_mode='r');data={'X':x,'y':metadata['y'],'mask':metadata['frame_mask'],'lengths':metadata['lengths'],'participant_ids':metadata['participant_ids']}
if x.shape!=(len(metadata['y']),96,258) or x.dtype!=np.float32:raise ValueError('TEST_SHAPE')
for start in range(0,len(x),128):
    if not np.isfinite(x[start:start+128]).all():raise ValueError('TEST_NONFINITE')
torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
contract=json.loads((primary/'source_contract.json').read_text());eps=contract['configuration']['0_ln1']['epsilon'];model=LegacyTCN(eps)
with np.load(primary/'source_weights.npz',allow_pickle=False) as w,torch.no_grad():
    for i,block in enumerate(model.blocks):
        for name in ('conv1','conv2','ln1','ln2'):
            layer=getattr(block,name);a=w[f'{i}_{name}_0'];layer.weight.copy_(torch.from_numpy(a.transpose(2,1,0).copy() if name.startswith('conv') else a));layer.bias.copy_(torch.from_numpy(w[f'{i}_{name}_1']))
        if i==0:block.proj.weight.copy_(torch.from_numpy(w['0_proj_0'].transpose(2,1,0).copy()));block.proj.bias.copy_(torch.from_numpy(w['0_proj_1']))
    model.head.weight.copy_(torch.from_numpy(w['head_0'].T.copy()));model.head.bias.copy_(torch.from_numpy(w['head_1']))
if not torch.cuda.is_available():raise RuntimeError('CUDA_TEST_BACKEND_REQUIRED')
model=model.cuda().eval();baseline,bprob=evaluate(model,data,'cuda:0');write(out/'baseline_fp32.json',baseline)
results={};preds={'baseline':np.asarray(baseline['predictions'])};np.save(out/'baseline_probabilities.npy',bprob)
for name,folder in [('primary',primary),('repeat',repeat)]:
    pointer=selection[name];path=folder/pointer['path']
    if sha(path)!=pointer['sha256']:raise ValueError('FROZEN_TEST_CHECKPOINT_HASH')
    ck=torch.load(path,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);result,prob=evaluate(model,data,'cuda:0')
    write(out/(name+'.json'),result);np.save(out/(name+'_probabilities.npy'),prob);results[name]={k:result[k] for k in ('top1','macro_f1','rows','per_signer')};preds[name]=np.asarray(result['predictions'])
rng=np.random.default_rng(20261020);ids=metadata['participant_ids'];signers=np.unique(ids);intervals={}
for name in ('primary','repeat'):
    deltas=[]
    for _ in range(2000):
        selected=rng.choice(signers,len(signers),replace=True);ix=np.concatenate([np.flatnonzero(ids==s) for s in selected]);deltas.append(float((preds[name][ix]==metadata['y'][ix]).mean()-(preds['baseline'][ix]==metadata['y'][ix]).mean()))
    intervals[name]=np.percentile(deltas,[2.5,97.5]).tolist()
if sha(source)!=source_hash or source.stat().st_mtime_ns!=source_mtime:raise ValueError('TEST_SOURCE_MUTATED')
write(run/'final_test_results.json',{'status':'PASS_EVALUATION_COMPLETE','partition':'native final test','source':str(source),'source_sha256':source_hash,
    'rows':len(x),'signer_count':len(signers),'train_val_signer_overlap':overlaps,'baseline':{k:baseline[k] for k in ('top1','macro_f1','rows','per_signer')},
    'candidates':results,'paired_signer_bootstrap_top1_delta_95pct':intervals,'checkpoint_selection_used_test':False,
    'post_test_training_performed':False,'same_input_same_FP32_backend':True,'false_activations':'NOT_MEASURED','production_path':'legacy258 only, current350 incompatible'})
print(json.dumps({'rows':len(x),'test_signers':len(signers),'baseline_top1':baseline['top1'],'primary_top1':results['primary']['top1'],'repeat_top1':results['repeat']['top1'],'intervals':intervals}))
