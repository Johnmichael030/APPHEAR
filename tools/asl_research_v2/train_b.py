"""Native-cadence genuine-observation CTC versus the verified old checkpoint."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from common import PREVIOUS,run_root,write,sha
from ctc_experiment import CausalCTC,batch,evaluate
from train_a import atomic_checkpoint
from replay_profile import memory
from prepare_b_features import SCHEMA
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);p.add_argument('--resume',action='store_true');args=p.parse_args();run=run_root(args.run_root);out=run/'experiment_B'
plan=json.loads((run/'experiment_B_plan.json').read_text());seed=plan['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
if not torch.cuda.is_available():raise RuntimeError('B_CUDA_REQUIRED')
quality=json.loads((out/'data/processed/full/quality.json').read_text())
if quality['status']!='PASS' or quality['collection_overlap']:raise ValueError('B_DATA_GATE')
rows=json.loads((out/'data/processed/full/manifest.json').read_text());data={}
for row in rows:
    path=run/row['feature_path']
    if sha(path)!=row['feature_sha256']:raise ValueError('B_FEATURE_HASH')
    with np.load(path,allow_pickle=False) as f:
        if str(f['schema_version'].item())!=SCHEMA:raise ValueError('B_SCHEMA')
        data[row['sample_id']]=f['X'].copy()
train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
vocab=['<BLANK>','<UNK>']+sorted({g for r in train for g in r['glosses']})
write(out/'vocabulary.json',{'schema_version':SCHEMA,'tokens':vocab,'fitted_split':'train'}) if not args.resume else None
values=np.concatenate([data[r['sample_id']] for r in train]).astype(np.float64);mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32)
model=CausalCTC(len(vocab),48,mean,std).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);criterion=nn.CTCLoss(blank=0,zero_infinity=False)
checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True);best=(float('inf'),float('inf'));patience=0;start_epoch=1
if args.resume:
    pointer=json.loads((checkpoints/'latest.json').read_text());cp=out/pointer['path']
    if sha(cp)!=pointer['sha256']:raise ValueError('B_RESUME_HASH')
    ck=torch.load(cp,map_location='cuda',weights_only=False)
    if ck['plan']!=plan:raise ValueError('B_RESUME_PLAN')
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);best=tuple(ck['best']);patience=ck['patience'];start_epoch=ck['epoch']+1
    random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
    if patience>=6 or start_epoch>30:print('B_ALREADY_COMPLETE');raise SystemExit(0)
else:
    # Recompute old WER/tokenization on original data, without retraining the completed model.
    old_pointer=json.loads((PREVIOUS/'checkpoints/best.json').read_text());cp=PREVIOUS/old_pointer['path']
    if sha(cp)!=old_pointer['sha256']:raise ValueError('PRIOR_CHECKPOINT_HASH')
    old=torch.load(cp,map_location='cpu',weights_only=False);s=old['model'];control=CausalCTC(len(vocab),48,s['mean'].numpy(),s['std'].numpy()).eval();control.load_state_dict(s)
    old_data={}
    old_rows=json.loads((PREVIOUS/'data/processed/full/manifest.json').read_text());old_val=[r for r in old_rows if r['split']=='val']
    for row in old_val:
        with np.load(PREVIOUS/row['feature_path'],allow_pickle=False) as f:old_data[row['sample_id']]=f['X'].copy()
    result=evaluate(control,old_val,old_data,vocab,'cpu')
    result['hypothesis_tokens']=sum(len(r['hypothesis']) for r in result['predictions']);result['checkpoint']=old_pointer
    write(out/'verified_10hz_baseline.json',result)
    native_control=evaluate(control,val,data,vocab,'cpu');native_control['hypothesis_tokens']=sum(len(r['hypothesis']) for r in native_control['predictions'])
    write(out/'frozen_model_native_cadence.json',native_control)
started=time.perf_counter();completed=start_epoch-1;status='MAX_EPOCHS'
for epoch in range(start_epoch,31):
    tick=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats();order=list(train);random.shuffle(order);losses=[]
    for i in range(0,len(order),4):
        x,y,l,tl=batch(order[i:i+4],data,vocab,'cuda:0');opt.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits.log_softmax(-1).transpose(0,1),y,l,tl)
        if not bool(torch.isfinite(loss)):raise RuntimeError('B_NONFINITE')
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not bool(torch.isfinite(norm)):raise RuntimeError('B_NONFINITE_GRAD')
        opt.step();losses.append(float(loss.detach()))
    result=evaluate(model,val,data,vocab,'cuda:0');score=(result['gloss_wer'],result['loss']);improved=score<best
    if improved:best=score;patience=0
    else:patience+=1
    record={'epoch':epoch,'train_ctc_loss':float(np.mean(losses)),'val_gloss_wer':result['gloss_wer'],'val_ctc_loss':result['loss'],
        'epoch_seconds':time.perf_counter()-tick,'peak_vram_allocated':torch.cuda.max_memory_allocated(),**memory()}
    with (out/'training_log.csv').open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(record))
        if epoch==1:w.writeheader()
        w.writerow(record)
    path=checkpoints/f'epoch_{epoch:03d}.pt';atomic_checkpoint(path,{'schema':SCHEMA,'epoch':epoch,'plan':plan,'best':best,'patience':patience,'model':model.state_dict(),'optimizer':opt.state_dict(),
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
    pointer={'path':str(path.relative_to(out)),'sha256':sha(path),'epoch':epoch};(checkpoints/'latest.json').write_text(json.dumps(pointer))
    if improved:
        (checkpoints/'best.json').write_text(json.dumps(pointer));result['hypothesis_tokens']=sum(len(r['hypothesis']) for r in result['predictions']);(out/'validation_best.json').write_text(json.dumps(result))
    completed=epoch;print(json.dumps(record),flush=True)
    if patience>=6:status='EARLY_STOP';break
write(out/'training_status.json',{'status':status,'completed_epochs':completed,'used_cuda':True,'best_score':best,'seconds':time.perf_counter()-started,'worker_memory':memory(),
    'schema_version':SCHEMA,'manifest_sha256':sha(out/'data/processed/full/manifest.json'),'script_sha256':sha(Path(__file__)),'signer_independence':'UNVERIFIED','non_sign_boundaries':'BLOCKED_NO_LABELS'})
