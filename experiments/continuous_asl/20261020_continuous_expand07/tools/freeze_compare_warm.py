"""Freeze adaptation choice, reload on CPU and compare unchanged9DEV."""
import csv
import json
import statistics
import time
import numpy as np
import torch
from core import RUN, SOURCE, read, write, sha
from train_warm import dataset
from ctc_experiment import CausalCTC,evaluate,seeds

def measure(base,pointer,rows,data):
    path=base/pointer['path']
    if sha(path)!=pointer['sha256']:raise ValueError('CHECKPOINT_HASH')
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=CausalCTC(len(state['vocabulary']),48,state['model']['mean'].numpy(),state['model']['std'].numpy())
    model.load_state_dict(state['model']);model.eval()
    result=evaluate(model,rows,data,state['vocabulary'],'cpu');timings=[]
    with torch.inference_mode():
        for row in rows:
            x=torch.from_numpy(data[row['sample_id']])[None]
            for _ in range(5):model(x)
            for _ in range(20):
                start=time.perf_counter();model(x);timings.append((time.perf_counter()-start)*1000)
    result.update(checkpoint_path=str(path),checkpoint_sha256=pointer['sha256'],selected_epoch=pointer['epoch'],
                  cpu_model_p50_ms=float(np.percentile(timings,50)),cpu_model_p95_ms=float(np.percentile(timings,95)),
                  timing_n=len(timings),cpu_threads=torch.get_num_threads(),parameters=sum(p.numel() for p in model.parameters()),vocabulary_size=len(state['vocabulary']))
    return result

def main():
    if (RUN/'comparison_dev.json').exists():print('FROZEN_COMPARISON_ALREADY_COMPLETE');return
    seeds(20261020);rows,data=dataset();dev=[r for r in rows if r['split']=='dev']
    base=RUN/'models/warm48';status=read(base/'training_status.json');pointer=read(base/'checkpoints/best.json')
    if status['script_sha256']!=sha(RUN/'tools/train_warm.py') or status['manifest_sha256']!=sha(RUN/'data/processed/train_dev/manifest.json'):raise ValueError('FIT_CONTRACT')
    if not (RUN/'model_selection_freeze.json').exists():
        write(RUN/'model_selection_freeze.json',{'selection':'Fixed9DEV WER then loss including initialized epoch0; TEST not accessed',
              'manifest_sha256':status['manifest_sha256'],'trainer_sha256':status['script_sha256'],'plan_sha256':sha(RUN/'training_plan.v2.json'),
              'models':{'warm48':{'pointer':pointer,'status_sha256':sha(base/'training_status.json'),'config_sha256':sha(base/'config.json')}}})
    candidate=measure(base,pointer,dev,data)
    baseline_pointer=read(SOURCE/'models/reference48/checkpoints/best.json')
    baseline=measure(SOURCE/'models/reference48',baseline_pointer,dev,data)
    if candidate['predictions']!=read(base/'dev_best.json')['predictions'] or baseline['predictions']!=read(SOURCE/'models/reference48/dev_best.json')['predictions']:raise ValueError('CPU_GPU_FROZEN_PREDICTION_PARITY')
    epochs=list(csv.DictReader((base/'training_log.csv').open(newline='')))
    if len(epochs)!=status['completed_epochs']:raise ValueError('EPOCH_COUNT')
    duration=sum(float(r['epoch_seconds']) for r in epochs)
    candidate.update(completed_epochs=status['completed_epochs'],fit_reload_seconds=status['seconds'],
                     epoch_median_seconds=statistics.median(float(r['epoch_seconds']) for r in epochs),
                     peak_pytorch_allocated_vram_bytes=status['peak_allocated_vram_bytes'],peak_pytorch_reserved_vram_bytes=status['peak_reserved_vram_bytes'],
                     unique_train_utterances_consumed=status['unique_train_utterances_consumed'],train_utterance_presentations=status['train_utterance_presentations'],
                     train_utterances_per_complete_epoch_second=status['train_utterance_presentations']/duration,
                     train_observations_per_complete_epoch_second=status['train_observations_per_complete_epoch']*len(epochs)/duration,
                     initialized_epoch0_selected=pointer['epoch']==0)
    result={'status':'PASS_ACTUAL_BOUNDED_ADAPTATION_AND_FROZEN_DEV_COMPARISON','candidate':candidate,'frozen_baseline':baseline,
            'beats_frozen_baseline':candidate['gloss_wer']<baseline['gloss_wer'],'dev':len(dev),'test_evaluated':0,
            'signer_independent':False,'production_promoted':False,'normalization':'Frozen original TRAIN100 mean/std retained',
            'evaluation':'Same9DEV literal references and unchanged greedyCTC;2CPU threads,5warmup+20repeat/sequence; model-only latency, no Android claim'}
    write(RUN/'comparison_dev.json',result)
    print(json.dumps({'dev_wer':candidate['gloss_wer'],'baseline_dev_wer':baseline['gloss_wer'],'selected_epoch':pointer['epoch'],'train':status['unique_train_utterances_consumed']}),flush=True)

if __name__=='__main__':main()
