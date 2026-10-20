"""Expanded fitting: port of completed trainer; no frozen-source writes or TEST evaluation."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from core import RUN, ROOT, SOURCE, OLD, SCHEMA, SEED, read, write, sha
from ctc_experiment import CausalCTC, batch, evaluate, seeds
from replay_profile import memory


def dataset(partition='train_dev'):
    base=RUN/'data/processed'/partition
    quality=read(base/'quality.json')
    if quality['status']!='PASS' or quality['source_media_hash_duplicates'] or quality['exact_decoded_duplicates']:
        raise ValueError('DATA_QUALITY_GATE')
    rows=read(base/'manifest.json'); data={}
    for row in rows:
        path=Path(row['feature_path']).resolve()
        if not (RUN in path.parents or SOURCE in path.parents or OLD in path.parents): raise ValueError('FEATURE_PATH')
        if sha(path)!=row['feature_sha256']: raise ValueError('FEATURE_HASH')
        with np.load(path,allow_pickle=False) as f:
            if str(f['schema_version'].item())!=SCHEMA: raise ValueError('FEATURE_SCHEMA')
            x=f['X'].copy()
        if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=378 or not np.isfinite(x).all(): raise ValueError('FEATURE_VALUES')
        data[row['sample_id']]=x
    return rows,data

def atomic_json(path, obj):
    path=Path(path); temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2);f.write('\n')
    os.replace(temp,path)

def checkpoint(out, epoch, model, optimizer, config, vocabulary, manifest_hash, best, patience):
    path=out/'checkpoints'/f'epoch_{epoch:03d}.pt'
    if path.exists(): raise ValueError('CHECKPOINT_EXISTS')
    partial=path.with_suffix('.pt.partial')
    torch.save({'schema':SCHEMA,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'config':config,'vocabulary':vocabulary,'manifest_sha256':manifest_hash,'best_score':best,'patience':patience,
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all()},partial)
    os.replace(partial,path)
    pointer={'path':str(path.relative_to(out)),'sha256':sha(path),'epoch':epoch}
    atomic_json(out/'checkpoints/latest.json',pointer)
    return pointer

def train(name, resume=False):
    if (RUN/'model_selection_freeze.json').exists(): raise ValueError('SELECTION_ALREADY_FROZEN_NO_TRAINING')
    if (RUN/'data/processed/test').exists(): raise ValueError('HOLDOUT_PRESENT_BEFORE_TRAINING')
    if name!='smoke':
        smoke=read(RUN/'models/smoke/training_status.json')
        if not smoke['checkpoint_reload_predictions_match'] or smoke['completed_epochs']!=1: raise ValueError('SMOKE_GATE')
    if name=='compact24' and not (RUN/'models/reference48/training_status.json').is_file(): raise ValueError('REFERENCE_MUST_BE_FROZEN_FIRST')
    plan=read(RUN/'training_plan.v2.json'); config={**{k:v for k,v in plan.items() if k!='models'},**plan['models'][name],'name':name}
    out=RUN/'models'/name
    if not resume:
        out.mkdir(parents=True,exist_ok=False); (out/'checkpoints').mkdir()
        write(out/'config.json',config)
    if not torch.cuda.is_available(): raise ValueError('CUDA_REQUIRED')
    seeds(SEED); os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    rows,data=dataset(); fit=[r for r in rows if r['split']=='train']; dev=[r for r in rows if r['split']=='dev']
    if dict(Counter(r['split'] for r in rows))!=read(RUN/'dataset_plan.json')['fit_counts']: raise ValueError('FROZEN_SPLIT_COUNTS')
    if {r['collection'] for r in fit}&{r['collection'] for r in dev}: raise ValueError('COLLECTION_LEAKAGE')
    vocabulary=['<BLANK>','<UNK>']+sorted({s for r in fit for s in r['glosses']})
    values=np.concatenate([data[r['sample_id']] for r in fit]).astype(np.float64)
    mean=values.mean(0).astype(np.float32); std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32)
    del values
    model=CausalCTC(len(vocabulary),config['channels'],mean,std).to('cuda:0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
    manifest_hash=sha(RUN/'data/processed/train_dev/manifest.json')
    best=(float('inf'),float('inf')); patience=0; first=1; completed=0
    if resume:
        ptr=read(out/'checkpoints/latest.json'); cp=out/ptr['path']
        if sha(cp)!=ptr['sha256']: raise ValueError('CHECKPOINT_HASH')
        state=torch.load(cp,map_location='cuda:0',weights_only=False)
        if state['config']!=config or state['manifest_sha256']!=manifest_hash: raise ValueError('RESUME_CONTRACT')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['epoch']+1
        best=tuple(state['best_score']);patience=state['patience'];completed=state['epoch']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        if completed>=config['max_epochs'] or patience>=config['patience']:
            print(json.dumps({'status':'VERIFIED_COMPLETE','name':name,'epochs':completed}));return
    else:
        write(out/'vocabulary.json',{'tokens':vocabulary,'fitted_split':'train','schema':SCHEMA})
        write(out/'untrained_dev.json',evaluate(model,dev[:2] if name=='smoke' else dev,data,vocabulary,'cuda:0'))
    if name=='smoke': fit=fit[:4];dev=dev[:2]
    start=time.perf_counter(); status='MAX_EPOCHS'; torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(first,config['max_epochs']+1):
            if time.perf_counter()-start>=config['wall_seconds']: status='WALL_BUDGET';break
            if shutil.disk_usage(RUN).free<10737418240: raise RuntimeError('DISK_HEADROOM')
            model.train();order=list(fit);random.shuffle(order);losses=[];tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            batches=list(range(0,len(order),config['batch']))
            for j,offset in enumerate(batches):
                free,total=torch.cuda.mem_get_info()
                if free<config['minimum_free_vram_bytes']: raise RuntimeError('FREE_VRAM_GUARD')
                x,y,lengths,target_lengths=batch(order[offset:offset+config['batch']],data,vocabulary,'cuda:0')
                logits=model(x); loss=criterion(logits.log_softmax(-1).transpose(0,1),y,lengths,target_lengths)
                if not bool(torch.isfinite(loss)): raise RuntimeError('NONFINITE_LOSS')
                (loss/config['gradient_accumulation']).backward()
                if torch.cuda.max_memory_allocated()>config['max_vram_allocated_bytes']: raise RuntimeError('VRAM_BUDGET')
                if (j+1)%config['gradient_accumulation']==0 or j+1==len(batches):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
                    if not bool(torch.isfinite(norm)): raise RuntimeError('NONFINITE_GRADIENT')
                    gradient_device=str(next(model.parameters()).grad.device)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            result=evaluate(model,dev,data,vocabulary,'cuda:0');torch.cuda.synchronize()
            score=(result['gloss_wer'],result['loss']);improved=score<best
            if improved: best=score;patience=0
            else: patience+=1
            pointer=checkpoint(out,epoch,model,optimizer,config,vocabulary,manifest_hash,best,patience)
            if improved:
                atomic_json(out/'checkpoints/best.json',pointer);atomic_json(out/'dev_best.json',result)
            record={'epoch':epoch,'train_ctc_loss':float(np.mean(losses)),'dev_loss':result['loss'],
                'dev_wer':result['gloss_wer'],'substitutions':result['substitutions'],'deletions':result['deletions'],
                'insertions':result['insertions'],'reference_tokens':result['reference_tokens'],
                'epoch_seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-start,
                'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
                'free_vram_bytes':torch.cuda.mem_get_info()[0],'worker_peak_working_set_bytes':memory()['peak_working_set_bytes'],'best':improved}
            record.update(input_device=str(x.device),logits_device=str(logits.device),loss_device=str(loss.device),
                          gradient_device=gradient_device,optimizer_state_device=str(optimizer.state[next(model.parameters())]['exp_avg'].device))
            with (out/'training_log.csv').open('a',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(record))
                if stream.tell()==0:writer.writeheader()
                writer.writerow(record)
            completed=epoch;print(json.dumps(record),flush=True)
            if patience>=config['patience']:status='EARLY_STOP';break
        if completed==0: raise RuntimeError('NO_EPOCH_COMPLETED')
        # Reload the new checkpoint and reproduce DEV predictions, not merely save a file.
        best_pointer=read(out/'checkpoints/best.json');path=out/best_pointer['path']
        if sha(path)!=best_pointer['sha256']: raise ValueError('SAVED_CHECKPOINT_HASH')
        loaded=torch.load(path,map_location='cuda:0',weights_only=False);model.load_state_dict(loaded['model'])
        reloaded=evaluate(model,dev,data,vocabulary,'cuda:0')
        if reloaded['predictions']!=read(out/'dev_best.json')['predictions']: raise ValueError('CHECKPOINT_RELOAD_PARITY')
        write(out/'training_status.json',{'status':status,'completed_epochs':completed,'used_cuda':True,'gpu_name':torch.cuda.get_device_name(0),
            'seconds':time.perf_counter()-start,'best_dev_score':best,'best_checkpoint':best_pointer,'checkpoint_reload_predictions_match':True,
            'vocabulary_size':len(vocabulary),'parameter_count':sum(p.numel() for p in model.parameters()),'worker_memory':memory(),
            'manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__)),'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),'device':'cuda:0','test_accessed':False})
    except Exception as error:
        write(out/'failure.json',{'error':str(error),'completed_epochs':completed,'resume_from_last_full_epoch':True})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',choices=('smoke','reference48','compact24'),required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    status=RUN/'models'/args.name/'training_status.json'
    if status.exists():
        saved=read(status);pointer=read(RUN/'models'/args.name/'checkpoints/best.json')
        if sha(RUN/'models'/args.name/pointer['path'])!=pointer['sha256']:raise ValueError('COMPLETED_CHECKPOINT_HASH')
        print(json.dumps({'status':'VERIFIED_COMPLETE_NO_TRAINING','name':args.name,'epochs':saved['completed_epochs']}))
    else:train(args.name,args.resume)
