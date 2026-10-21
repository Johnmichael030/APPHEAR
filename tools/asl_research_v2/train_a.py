"""Bounded 236-class isolated-sign candidate; never selects a production model."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from torch import nn
from common import ROOT,run_root,sha,write
from ctc_experiment import CausalBlock
from replay_profile import memory

class IsolatedTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__();self.input=nn.Linear(258,64)
        self.blocks=nn.Sequential(*(CausalBlock(64,d) for d in (1,2,4,8)))
        self.dropout=nn.Dropout(.1);self.output=nn.Linear(64,236)
    def forward(self,x: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
        x=torch.relu(self.input(x)).transpose(1,2);x=self.blocks(x).transpose(1,2)
        weight=mask.to(x.dtype).unsqueeze(-1)
        pooled=(x*weight).sum(1)/weight.sum(1).clamp_min(1.)
        return self.output(self.dropout(pooled))

def load(run: Path,split: str) -> dict:
    root=run/'experiment_A/data'/split;x=np.load(root/'X.npy',mmap_mode='r',allow_pickle=False)
    with np.load(root/'metadata.npz',allow_pickle=False) as f:metadata={k:f[k].copy() for k in f.files}
    if str(metadata['schema_version'].item())!='3.0' or x.shape[-1]!=258:raise ValueError('A_SCHEMA')
    return {'X':x,**metadata}

def batch(data: dict,indices: np.ndarray,device: str) -> tuple:
    mask=data['mask'][indices];max_t=max(1,int(np.flatnonzero(mask.any(0))[-1])+1)
    return (torch.as_tensor(np.array(data['X'][indices,:max_t],copy=True),device=device),
        torch.as_tensor(mask[:,:max_t],device=device),torch.as_tensor(data['y'][indices].astype(np.int64),device=device))

def metrics(y: np.ndarray,pred: np.ndarray,ids: np.ndarray) -> dict:
    cm=np.zeros((236,236),np.int64);np.add.at(cm,(y,pred),1)
    tp=cm.diagonal();precision=np.divide(tp,cm.sum(0),out=np.zeros(236),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(236),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(236),where=precision+recall>0)
    return {'top1':float((y==pred).mean()),'macro_f1':float(f1.mean()),'rows':len(y),
        'per_signer':{str(i):{'rows':int((ids==i).sum()),'top1':float((pred[ids==i]==y[ids==i]).mean())} for i in np.unique(ids)},
        'per_class':[{'class_id':i,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(cm[i].sum())} for i in range(236)],
        'confusion_matrix':cm.tolist(),'predictions':pred.tolist(),
        'scope':'isolated-sign validation on held-out signer IDs, legacy258; no streaming/negative/device claim'}

def evaluate(model: nn.Module,data: dict,device: str) -> tuple[dict,np.ndarray]:
    model.eval();pred=[];probs=[]
    with torch.inference_mode():
        for start in range(0,len(data['y']),32):
            x,m,y=batch(data,np.arange(start,min(start+32,len(data['y']))),device)
            p=model(x,m).softmax(-1);pred.extend(p.argmax(-1).cpu().tolist());probs.append(p.cpu().numpy())
    return metrics(data['y'],np.asarray(pred),data['participant_ids']),np.concatenate(probs)

def atomic_checkpoint(path: Path,value: dict) -> None:
    if path.exists():raise FileExistsError('CHECKPOINT_NO_OVERWRITE')
    temp=path.with_suffix('.pt.partial');torch.save(value,temp);os.replace(temp,path)

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args();run=run_root(args.run_root)
    out=run/'experiment_A';checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    audit=json.loads((out/'data_audit.json').read_text())
    if audit['status']!='PASS' or audit['signer_overlap']:raise ValueError('A_DATA_GATE')
    cfg={'task':'isolated_gloss_classification','schema':'3.0','shape':[96,258],'seed':20261020,'max_epochs':5,
        'wall_budget_seconds':600,'batch_candidates':[1,2,4],'accumulation':8,'optimizer':'AdamW','lr':.001,
        'channels':64,'dropout':.1,'augmentation':'none','selection':'validation macro F1 then top1',
        'classes':236,'source_audit_sha256':sha(out/'data_audit.json'),'production_enabled':False}
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    torch.backends.cuda.matmul.allow_tf32=False
    train=load(run,'train');val=load(run,'val');model=IsolatedTCN().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    start_epoch=1;best=(-1.,-1.);batch_size=4
    if args.resume:
        pointer=json.loads((checkpoints/'latest.json').read_text());path=out/pointer['path']
        if sha(path)!=pointer['sha256']:raise ValueError('RESUME_HASH')
        ckpt=torch.load(path,map_location='cuda',weights_only=False)
        if ckpt['config']!=cfg:raise ValueError('RESUME_CONFIG')
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);best=tuple(ckpt['best']);start_epoch=ckpt['epoch']+1
        random.setstate(ckpt['python_rng']);np.random.set_state(ckpt['numpy_rng']);torch.set_rng_state(ckpt['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ckpt['cuda_rng']])
        if start_epoch>cfg['max_epochs']:print('ALREADY_COMPLETE');return
    else:
        if (out/'training_log.csv').exists():raise ValueError('USE_RESUME')
        write(out/'config.json',cfg)
        initial=copy.deepcopy(model.state_dict());probes=[]
        for size in cfg['batch_candidates']:
            model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr']);torch.cuda.reset_peak_memory_stats()
            indices=np.argsort(train['lengths'])[-size:];x,m,y=batch(train,indices,'cuda:0');opt.zero_grad();old=model.output.weight.detach().clone()
            logits=model(x,m);loss=nn.functional.cross_entropy(logits,y);loss.backward();finite=bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            opt.step();torch.cuda.synchronize();delta=float((model.output.weight-old).abs().max());peak=torch.cuda.max_memory_allocated()
            probes.append({'batch':size,'input':str(x.device),'parameter':str(next(model.parameters()).device),'loss':str(loss.device),
                'finite_output':bool(torch.isfinite(logits).all()),'finite_gradients':finite,'optimizer_parameter_max_delta':delta,'peak_allocated_vram_bytes':peak,'memory':memory()})
            if not finite or delta<=0:raise RuntimeError('GPU_OPTIMIZER_GATE')
            if peak>1073741824:raise RuntimeError('VRAM_BUDGET')
        model.load_state_dict(initial);del initial;optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        cpu=IsolatedTCN();cx,cm,cy=batch(train,np.asarray([0]),'cpu');closs=nn.functional.cross_entropy(cpu(cx,cm),cy);closs.backward()
        write(run/'gpu_capability_report.json',{'status':'PASS','GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
            'probes':probes,'selected_batch_size':4,'gradient_accumulation':8,'CPU_backward_finite':bool(torch.isfinite(closs)),
            'test_type':'real isolated-sign data, forward/loss/backward/AdamW update','worker_PID':os.getpid()})
        del cpu;random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    started=time.perf_counter();completed=start_epoch-1;status='MAX_EPOCHS';records=[]
    for epoch in range(start_epoch,cfg['max_epochs']+1):
        if shutil.disk_usage(run).free<4*1073741824:raise RuntimeError('DISK_STOP')
        tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train();order=np.random.permutation(len(train['y']));loss_sum=0.;seen=0;optimizer.zero_grad(set_to_none=True)
        chunks=list(range(0,len(order),batch_size))
        for step,offset in enumerate(chunks):
            indices=order[offset:offset+batch_size];x,m,y=batch(train,indices,'cuda:0');logits=model(x,m);loss=nn.functional.cross_entropy(logits,y)
            if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_LOSS')
            # Mean gradient over the actual accumulation group, including its shorter last group.
            group_size=min(cfg['accumulation'],len(chunks)-(step//cfg['accumulation'])*cfg['accumulation'])
            (loss/group_size).backward();loss_sum+=float(loss.detach())*len(indices);seen+=len(indices)
            if (step+1)%cfg['accumulation']==0 or step+1==len(chunks):
                norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not bool(torch.isfinite(norm)):raise RuntimeError('NONFINITE_GRADIENT')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if step%500==0:print(json.dumps({'epoch':epoch,'train_rows_seen':seen,'loss':loss_sum/seen,'memory':memory()}),flush=True)
            if time.perf_counter()-started>cfg['wall_budget_seconds']:raise RuntimeError('TIME_BUDGET_STOP_RESUME_FROM_LAST_COMPLETE_EPOCH')
        result,prob=evaluate(model,val,'cuda:0');score=(result['macro_f1'],result['top1']);is_best=score>best
        if is_best:best=score
        record={'epoch':epoch,'train_loss':loss_sum/seen,'val_top1':result['top1'],'val_macro_f1':result['macro_f1'],
            'epoch_seconds':time.perf_counter()-tick,'peak_vram_allocated':torch.cuda.max_memory_allocated(),
            'peak_vram_reserved':torch.cuda.max_memory_reserved(),**memory()}
        with (out/'training_log.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(record))
            if epoch==1:w.writeheader()
            w.writerow(record)
        path=checkpoints/f'epoch_{epoch:03d}.pt';atomic_checkpoint(path,{'epoch':epoch,'config':cfg,'best':best,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
        pointer={'path':str(path.relative_to(out)),'sha256':sha(path),'epoch':epoch}
        (checkpoints/'latest.json').write_text(json.dumps(pointer))
        if is_best:
            (checkpoints/'best.json').write_text(json.dumps(pointer));(out/'validation_best.json').write_text(json.dumps(result))
            np.save(out/'validation_best_probabilities.npy',prob)
        completed=epoch;print(json.dumps(record),flush=True)
    write(out/'training_status.json',{'status':status,'completed_epochs':completed,'used_cuda':True,'best_score':best,
        'seconds':time.perf_counter()-started,'actual_worker_memory':memory(),'script_sha256':sha(Path(__file__))})
if __name__=='__main__':main()
