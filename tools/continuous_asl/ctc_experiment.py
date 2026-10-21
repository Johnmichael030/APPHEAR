"""Small causal gloss CTC research baseline, isolated from production recognition."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import sys
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from prepare_features import SCHEMA, checked_run, sha

class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__(); self.pad=2*dilation
        self.conv1=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.conv2=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.norm1=nn.LayerNorm(channels); self.norm2=nn.LayerNorm(channels)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        a=self.conv1(F.pad(x,(self.pad,0))); a=F.relu(self.norm1(a.transpose(1,2))).transpose(1,2)
        a=self.conv2(F.pad(a,(self.pad,0))); a=self.norm2(a.transpose(1,2)).transpose(1,2)
        return F.relu(x+a)

class CausalCTC(nn.Module):
    def __init__(self, classes: int, channels: int, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.input=nn.Linear(378,channels)
        self.blocks=nn.Sequential(*(CausalBlock(channels,d) for d in (1,2,4,8)))
        self.output=nn.Linear(channels,classes)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x=torch.clamp((x-self.mean)/self.std,-10.,10.)
        x=F.relu(self.input(x)).transpose(1,2)
        return self.output(self.blocks(x).transpose(1,2))

def load_data(run: Path) -> tuple[list[dict],dict[str,np.ndarray]]:
    path=run/'data/processed/full/manifest.json'
    quality=json.loads((path.parent/'quality.json').read_text())
    if quality['status']!='PASS' or quality['collection_overlap'] or quality['video_sha256_duplicates']:
        raise ValueError('DATA_QUALITY_GATE')
    rows=json.loads(path.read_text()); data={}
    groups={s:{r['collection'] for r in rows if r['split']==s} for s in ('train','val')}
    if not all(groups.values()) or groups['train']&groups['val']: raise ValueError('COLLECTION_SPLIT')
    for row in rows:
        path=run/row['feature_path']; path.resolve().relative_to(run/'data/processed/full')
        if sha(path)!=row['feature_sha256']: raise ValueError('FEATURE_HASH')
        with np.load(path,allow_pickle=False) as f:
            if str(f['schema_version'].item())!=SCHEMA: raise ValueError('FEATURE_SCHEMA')
            x=f['X'].copy()
        if x.ndim!=2 or x.shape[1]!=378 or not np.isfinite(x).all(): raise ValueError('FEATURE_VALUES')
        data[row['sample_id']]=x
    return rows,data

def edit_counts(ref: list[str], hyp: list[str]) -> dict[str,int]:
    # Minimum edit distance, deterministic tie order match/substitute, delete, insert.
    n,m=len(ref),len(hyp); d=np.zeros((n+1,m+1),dtype=np.int32)
    d[:,0]=np.arange(n+1);d[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1): d[i,j]=min(d[i-1,j-1]+(ref[i-1]!=hyp[j-1]),d[i-1,j]+1,d[i,j-1]+1)
    counts=dict(substitutions=0,deletions=0,insertions=0,reference_tokens=n)
    i,j=n,m
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
            counts['substitutions']+=int(ref[i-1]!=hyp[j-1]);i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1: counts['deletions']+=1;i-=1
        else: counts['insertions']+=1;j-=1
    return counts

def decode(logits: torch.Tensor, vocabulary: list[str]) -> list[str]:
    values=logits.argmax(-1).detach().cpu().tolist(); previous=-1; result=[]
    for token in values:
        if token!=previous and token!=0: result.append(vocabulary[token])
        previous=token
    return result

def batch(rows: list[dict], data: dict, vocab: list[str], device: str) -> tuple:
    mapping={s:i for i,s in enumerate(vocab)}
    lengths=torch.tensor([len(data[r['sample_id']]) for r in rows],dtype=torch.long)
    x=torch.zeros((len(rows),int(lengths.max()),378),device=device)
    targets=[];sizes=[]
    for i,row in enumerate(rows):
        v=data[row['sample_id']];x[i,:len(v)]=torch.as_tensor(v,device=device)
        ids=[mapping.get(s,1) for s in row['glosses']];targets.extend(ids);sizes.append(len(ids))
    return x,torch.tensor(targets,device=device,dtype=torch.long),lengths,torch.tensor(sizes,dtype=torch.long)

def evaluate(model: nn.Module, rows: list[dict], data: dict, vocab: list[str], device: str) -> dict:
    model.eval(); criterion=nn.CTCLoss(blank=0,zero_infinity=False);losses=[];predictions=[]
    counts=Counter(substitutions=0,deletions=0,insertions=0,reference_tokens=0)
    with torch.inference_mode():
        for row in rows:
            x,y,l,tl=batch([row],data,vocab,device); logits=model(x)
            loss=criterion(logits.log_softmax(-1).transpose(0,1),y,l,tl)
            if not torch.isfinite(loss): raise ValueError('NONFINITE_EVALUATION')
            losses.append(float(loss)); hyp=decode(logits[0],vocab); edits=edit_counts(row['glosses'],hyp);counts.update(edits)
            predictions.append({'sample_id':row['sample_id'],'reference':row['glosses'],'hypothesis':hyp,**edits})
    wer=(counts['substitutions']+counts['deletions']+counts['insertions'])/counts['reference_tokens']
    return {'loss':float(np.mean(losses)),'gloss_wer':wer,**dict(counts),'predictions':predictions,
        'scope':'COLLECTION-DISJOINT DIAGNOSTIC, SIGNER UNVERIFIED; NOT PRODUCT ACCURACY',
        'out_of_vocabulary_tokens':sum(s not in vocab for r in rows for s in r['glosses'])}

def seeds(seed: int) -> None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def save_checkpoint(run: Path, epoch: int, model: nn.Module, optimizer: Any, config: dict,
                    manifest_hash: str, score: tuple, best: tuple, patience: int) -> Path:
    path=run/'checkpoints'/f'epoch_{epoch:03d}.pt'
    if path.exists(): raise ValueError('CHECKPOINT_EXISTS')
    torch.save({'schema':SCHEMA,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'config':config,'manifest_sha256':manifest_hash,'score':score,'best_score':best,'patience_count':patience,
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all()},path)
    pointer={'path':str(path.relative_to(run)),'sha256':sha(path),'epoch':epoch}
    (run/'checkpoints/latest.json').write_text(json.dumps(pointer,indent=2)+'\n')
    if score==best: (run/'checkpoints/best.json').write_text(json.dumps(pointer,indent=2)+'\n')
    return path

def train(run: Path, resume: bool) -> None:
    config=yaml.safe_load((run/'training_config.yaml').read_text()); seeds(config['seed'])
    if config['schema_version']!=SCHEMA: raise ValueError('CONFIG_SCHEMA')
    expected={'dilations':[1,2,4,8],'convolutions_per_block':2,'kernel_size':3,
        'receptive_field_observations':61,'normalization_clip':10.,'blank_id':0,'unknown_id':1,
        'mixed_precision':False,'gradient_accumulation':1,'gradient_checkpointing':False}
    if any(config[k]!=v for k,v in expected.items()): raise ValueError('UNSUPPORTED_ARCHITECTURE_CONFIG')
    if json.loads((run/'logs/baseline_s2.json').read_text())['status']!='PASS': raise ValueError('BASELINE_GATE')
    if not torch.cuda.is_available(): raise ValueError('CUDA_REQUIRED_NO_SILENT_CPU_TRAINING')
    if not json.loads((run/'logs/gpu_probe.json').read_text())['gpu_execution_verified']: raise ValueError('GPU_GATE')
    rows,data=load_data(run);train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='val']
    manifest_hash=sha(run/'data/processed/full/manifest.json'); device='cuda:0'
    vocab=['<BLANK>','<UNK>']+sorted({s for r in train_rows for s in r['glosses']})
    values=np.concatenate([data[r['sample_id']] for r in train_rows]).astype(np.float64)
    mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32)
    model=CausalCTC(len(vocab),config['channels'],mean,std).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion=nn.CTCLoss(blank=0,zero_infinity=False)
    start_epoch=1;best=(float('inf'),float('inf'));patience=0
    if resume:
        pointer=json.loads((run/'checkpoints/latest.json').read_text()); path=(run/pointer['path']).resolve()
        path.relative_to(run/'checkpoints')
        if sha(path)!=pointer['sha256']: raise ValueError('CHECKPOINT_HASH')
        ckpt=torch.load(path,map_location=device,weights_only=False) # Own hash-verified checkpoint only.
        if ckpt['manifest_sha256']!=manifest_hash or ckpt['config']!=config: raise ValueError('RESUME_CONTRACT')
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);start_epoch=ckpt['epoch']+1
        best=tuple(ckpt['best_score']);patience=ckpt['patience_count']
        random.setstate(ckpt['python_rng']);np.random.set_state(ckpt['numpy_rng']);torch.set_rng_state(ckpt['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in ckpt['cuda_rng']])
        batch_size=json.loads((run/'logs/batch_probe.json').read_text())['selected_batch_size']
    else:
        if (run/'training_log.csv').exists() or list((run/'checkpoints').glob('*.pt')): raise ValueError('TRAINING_EXISTS_USE_RESUME')
        (run/'vocabulary.json').write_text(json.dumps({'schema_version':SCHEMA,'tokens':vocab,'fitted_split':'train'},indent=2)+'\n')
        # Probe actual longest sequences. Restore initialization/RNG before learning.
        initial={k:v.detach().clone() for k,v in model.state_dict().items()};probes=[];batch_size=0
        longest=max(train_rows,key=lambda r:len(data[r['sample_id']]))
        free,_=torch.cuda.mem_get_info()
        for size in config['batch_candidates']:
            optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            try:
                x,y,l,tl=batch([longest]*size,data,vocab,device); logits=model(x)
                loss=criterion(logits.log_softmax(-1).transpose(0,1),y,l,tl);loss.backward();torch.cuda.synchronize()
                finite=bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                peak=torch.cuda.max_memory_allocated();probes.append({'batch_size':size,'peak_allocated_bytes':peak,
                    'elapsed_ms':(time.perf_counter()-tick)*1000,'input_device':str(x.device),'logits_device':str(logits.device),
                    'loss_device':str(loss.device),'gradient_device':str(next(model.parameters()).grad.device),'finite':finite})
                if finite and peak < free*config['vram_budget_fraction']: batch_size=size
                del x,y,logits,loss
            except torch.cuda.OutOfMemoryError:
                probes.append({'batch_size':size,'status':'CUDA_OUT_OF_MEMORY'});torch.cuda.empty_cache();break
        if not batch_size: raise ValueError('NO_SAFE_BATCH_SIZE')
        optimizer.zero_grad(set_to_none=True);model.load_state_dict(initial);del initial;seeds(config['seed'])
        (run/'logs/batch_probe.json').write_text(json.dumps({'probes':probes,'selected_batch_size':batch_size,'free_vram_before':free},indent=2)+'\n')
        before=evaluate(model,val_rows,data,vocab,device)
        (run/'logs/untrained_baseline.json').write_text(json.dumps(before,indent=2)+'\n')
    epochs_completed=start_epoch-1; status='MAX_EPOCHS';started=time.perf_counter()
    for epoch in range(start_epoch,config['max_epochs']+1):
        if shutil.disk_usage(run).free<config['disk_headroom_bytes']: raise RuntimeError('DISK_HEADROOM_STOP')
        tick=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train();losses=[]
        order=list(train_rows);random.shuffle(order)
        for offset in range(0,len(order),batch_size):
            x,y,l,tl=batch(order[offset:offset+batch_size],data,vocab,device);optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=criterion(logits.log_softmax(-1).transpose(0,1),y,l,tl)
            if not bool(torch.isfinite(loss)): raise RuntimeError('NONFINITE_LOSS_STOP')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
            if not bool(torch.isfinite(norm)): raise RuntimeError('NONFINITE_GRADIENT_STOP')
            optimizer.step();losses.append(float(loss.detach()))
        val=evaluate(model,val_rows,data,vocab,device);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        score=(val['gloss_wer'],val['loss']);improved=score<best
        if improved: best=score;patience=0
        else: patience+=1
        record={'epoch':epoch,'train_ctc_loss':float(np.mean(losses)),'val_ctc_loss':val['loss'],'val_gloss_wer':val['gloss_wer'],
            'epoch_seconds':elapsed,'train_observations_per_second':sum(len(data[r['sample_id']]) for r in train_rows)/elapsed,
            'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
            'batch_size':batch_size,'device':device,'best':improved,'elapsed_run_seconds':time.perf_counter()-started}
        with (run/'training_log.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(record))
            if epoch==1: w.writeheader()
            w.writerow(record)
        # Each epoch is resumable; no original model path is a possible destination.
        save_checkpoint(run,epoch,model,optimizer,config,manifest_hash,score,best,patience)
        epochs_completed=epoch;print(json.dumps(record),flush=True)
        if patience>=config['early_stopping_patience']: status='EARLY_STOP';break
    (run/'logs/training_status.json').write_text(json.dumps({'status':status,'epochs_completed':epochs_completed,
        'training_used_gpu':True,'device':device,'gpu_name':torch.cuda.get_device_name(0),'best_validation_score':best,
        'parameter_count':sum(p.numel() for p in model.parameters()),'manifest_sha256':manifest_hash,
        'config_sha256':sha(run/'training_config.yaml'),'training_script_sha256':sha(Path(__file__)),
        'scope':'PRIVATE_RESEARCH_DIAGNOSTIC_ONLY','nonsign_and_boundary_heads':'NOT_TRAINED_NO_VALID_LABELS',
        'mixed_precision':False,'determinism':'fixed seeds; deterministic algorithms warn_only, CUDA CTC may remain nondeterministic'},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    run=checked_run(args.run_root);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    train(run,args.resume)
