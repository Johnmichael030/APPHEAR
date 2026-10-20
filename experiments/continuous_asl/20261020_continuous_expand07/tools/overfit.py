"""Bounded real CUDA learnability diagnostic, strictly TRAIN-only."""
import csv
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from core import RUN, FIVE, SIX, SCHEMA, SEED, read, write, sha
from ctc_experiment import CausalCTC,batch,evaluate,seeds
from replay_profile import memory

def prepare():
    folder=RUN/'overfit_diagnostic'
    if (folder/'manifest.json').exists():return read(folder/'manifest.json')
    prior=[r for r in read(FIVE/'data/processed/train_dev/manifest.json') if r['split']=='train']
    timed=read(SIX/'timed_preparation/manifest.json')
    pilot=read(SIX/'asllrp2025_catalog/feature_pilot.json')
    asl=[read(SIX/'data/processed/asllrp2025'/(s+'.row.json')) for s in pilot['accepted']]
    if pilot['status']!='PASS_MEDIA_FRAME_FEATURE_JOIN' or len(asl)!=3:raise ValueError('ASLLRP_PILOT_GATE')
    fs=next(r for r in prior if any(t.startswith('fs-') for t in r['glosses']))
    repeated=next((r for r in prior if r['sample_id']!=fs['sample_id'] and len(set(r['glosses']))<len(r['glosses'])),None)
    if repeated is None:raise ValueError('NO_REPEATED_WORD_DIAGNOSTIC_SAMPLE')
    stories={}
    for row in sorted(timed,key=lambda r:r['sample_id']):
        if row['requires_temporal_slice'] and 3<=len(row['glosses'])<=12 and row['collection'] not in stories:
            stories[row['collection']]=row
        if len(stories)==3:break
    rows=[fs,repeated]+list(stories.values())+asl
    if len(rows)!=8 or len({r['sample_id'] for r in rows})!=8 or any(r['split']!='train' for r in rows):raise ValueError('FROZEN_DIAGNOSTIC_SUBSET')
    if any(r['collection'] in {'ncslgr10e.xml','ncslgr10m.xml','ncslgr10p.xml','ncslgr10q.xml'} for r in rows):raise ValueError('PROTECTED_COLLECTION')
    write(folder/'manifest.json',rows)
    write(folder/'selection_provenance.json',{'policy':'First authentic fingerspelled-word record; first different repeated-word record; deterministic3 distinct timed narrative collections with3–12 events;3 source-collection-diverse ASLLRP pilot clips.',
          'count':8,'manifest_sha256':sha(folder/'manifest.json'),'protocol_sha256':sha(RUN/'diagnostic_protocol.v2.json'),
          'test_accessed':False,'development_accessed':False,'synthetic':False})
    return rows

def main(resume=False):
    folder=RUN/'overfit_diagnostic';out=folder/'models';status_path=folder/'result.json'
    if status_path.exists():
        status=read(status_path)
        if sha(status['checkpoint_path'])!=status['checkpoint_sha256']:raise ValueError('DIAGNOSTIC_CHECKPOINT_HASH')
        print(json.dumps({'status':'COMPLETE_NO_REPEAT','gate':status['status']}));return
    rows=prepare();plan=read(RUN/'diagnostic_protocol.v2.json');manifest_hash=sha(folder/'manifest.json')
    seeds(SEED);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if not torch.cuda.is_available():raise ValueError('CUDA_REQUIRED')
    data={}
    for row in rows:
        if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
        with np.load(row['feature_path'],allow_pickle=False) as f:
            x=f['X'].copy();ts=f['timestamp_ms']
            if str(f['schema_version'].item())!=SCHEMA or x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
        if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_FEASIBILITY')
        data[row['sample_id']]=x
    vocabulary=['<BLANK>','<UNK>']+sorted({g for r in rows for g in r['glosses']})
    values=np.concatenate(list(data.values())).astype(np.float64);mean=values.mean(0).astype(np.float32);std=values.std(0);std=np.where(std<1e-6,1.,std).astype(np.float32)
    model=CausalCTC(len(vocabulary),48,mean,std).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.0001)
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);initial=None;first=1;completed=0;prior_elapsed=0.;best=(float('inf'),float('inf'));best_path=None;best_result=None
    out.mkdir(parents=True,exist_ok=True)
    latest=out/'latest.json'
    if resume:
        pointer=read(latest);path=out/pointer['file']
        if sha(path)!=pointer['sha256']:raise ValueError('RESUME_CHECKPOINT_HASH')
        state=torch.load(path,map_location='cuda:0',weights_only=False)
        if state['manifest_sha256']!=manifest_hash or state['protocol_sha256']!=sha(RUN/'diagnostic_protocol.v2.json'):raise ValueError('RESUME_FROZEN_CONTRACT')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['epoch']+1;completed=state['epoch'];initial=state['initial'];best=tuple(state['best']);best_path=Path(state['best_path']);best_result=state['best_result'];prior_elapsed=state['elapsed_seconds']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    else:
        if latest.exists():raise ValueError('EXISTING_DIAGNOSTIC_USE_RESUME')
        initial=evaluate(model,rows,data,vocabulary,'cuda:0');write(folder/'initial_train.json',initial)
        write(folder/'config.json',{**plan,'vocabulary':vocabulary,'manifest_sha256':manifest_hash,'trainer_sha256':sha(__file__)})
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();passed=False
    for epoch in range(first,plan['max_epochs']+1):
        if prior_elapsed+time.perf_counter()-start>=plan['max_seconds']:break
        model.train();order=list(rows);random.shuffle(order);optimizer.zero_grad(set_to_none=True);losses=[];tick=time.perf_counter()
        for offset in (0,4):
            free,_=torch.cuda.mem_get_info()
            if free<268435456:raise RuntimeError('FREE_VRAM_GUARD')
            x,y,lengths,target_lengths=batch(order[offset:offset+4],data,vocabulary,'cuda:0')
            logits=model(x);loss=criterion(logits.log_softmax(-1).transpose(0,1),y,lengths,target_lengths)
            if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_LOSS')
            (loss/2).backward();losses.append(float(loss.detach()))
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not bool(torch.isfinite(gradient)):raise RuntimeError('NONFINITE_GRADIENT')
        if torch.cuda.max_memory_allocated()>2684354560:raise RuntimeError('VRAM_GUARD')
        optimizer.step();torch.cuda.synchronize();completed=epoch
        record={'epoch':epoch,'train_ctc_loss':sum(losses)/2,'epoch_seconds':time.perf_counter()-tick,'elapsed_seconds':prior_elapsed+time.perf_counter()-start,
                'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'device':str(x.device)}
        with (folder/'epoch_log.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(record))
            if f.tell()==0:writer.writeheader()
            writer.writerow(record)
        if epoch==1 or epoch%5==0 or epoch==plan['max_epochs']:
            result=evaluate(model,rows,data,vocabulary,'cuda:0');score=(result['gloss_wer'],result['loss']);path=out/f'epoch_{epoch:03d}.pt'
            if path.exists():raise ValueError('DIAGNOSTIC_CHECKPOINT_EXISTS')
            if score<best:best=score;best_result=result;best_path=path
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'vocabulary':vocabulary,
                   'manifest_sha256':manifest_hash,'protocol_sha256':sha(RUN/'diagnostic_protocol.v2.json'),'initial':initial,'best':best,
                   'best_result':best_result,'best_path':str(best_path),'elapsed_seconds':prior_elapsed+time.perf_counter()-start,
                   'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,path)
            pointer={'file':path.name,'sha256':sha(path),'epoch':epoch}
            temp=latest.with_suffix('.partial.json')
            with temp.open('x',encoding='utf-8') as f:json.dump(pointer,f)
            os.replace(temp,latest)
            print(json.dumps({'epoch':epoch,'train_wer':result['gloss_wer'],'train_loss':result['loss'],'initial_loss':initial['loss'],'seconds':record['elapsed_seconds']}),flush=True)
            passed=result['gloss_wer']<=plan['accept_train_wer_at_most'] and result['loss']<=initial['loss']*plan['accept_ctc_loss_fraction_of_initial_at_most']
            if passed:break
    if best_path is None:raise RuntimeError('NO_DIAGNOSTIC_EVALUATION_CHECKPOINT')
    selected=torch.load(best_path,map_location='cuda:0',weights_only=False);model.load_state_dict(selected['model'])
    reloaded=evaluate(model,rows,data,vocabulary,'cuda:0')
    if reloaded['predictions']!=best_result['predictions']:raise ValueError('DIAGNOSTIC_RELOAD_PARITY')
    output={'status':'PASS_LEARNABILITY_GATE' if passed else 'FAIL_LEARNABILITY_GATE','claim':'8-utterance TRAIN memorization diagnostic only; not generalization.',
            'initial_train':initial,'best_train':reloaded,'loss_ratio':reloaded['loss']/initial['loss'],'completed_epochs':completed,
            'selected_epoch':selected['epoch'],'unique_train_utterances_consumed':len(rows),'train_utterance_presentations':len(rows)*completed,
            'checkpoint_path':str(best_path),'checkpoint_sha256':sha(best_path),'checkpoint_reload_predictions_match':True,
            'manifest_sha256':manifest_hash,'trainer_sha256':sha(__file__),'used_cuda':True,'gpu':torch.cuda.get_device_name(0),
            'fit_reload_seconds':prior_elapsed+time.perf_counter()-start,'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),'worker_memory':memory(),'test_accessed':False,'development_accessed':False}
    write(status_path,output);print(json.dumps({k:v for k,v in output.items() if k not in ('initial_train','best_train')}),flush=True)
    if not passed:raise SystemExit(2)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
