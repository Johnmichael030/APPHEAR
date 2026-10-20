"""Freeze before post-selection TRAIN measurement; fixed9DEV comparison only."""
import ast
import csv
import json
import statistics
import time
from collections import Counter
import numpy as np
import torch
from core import RUN,SOURCE,FIVE,read,write,sha
from ctc_experiment import CausalCTC,evaluate,seeds

# Reuse the already exercised CPU measurement function without changing math.
source_path=FIVE/'tools/freeze_compare.py'
text=source_path.read_text(encoding='utf-8');tree=ast.parse(text)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measured_model')
exec(compile(ast.Module(body=[function],type_ignores=[]),str(source_path),'exec'),globals())

def main():
    if (RUN/'comparison_dev.json').exists():print('FROZEN_COMPARISON_COMPLETE_NO_REPEAT');return
    from train_main import dataset
    seeds(20261020);rows,data=dataset();dev=[r for r in rows if r['split']=='dev'];fit=[r for r in rows if r['split']=='train']
    base=RUN/'models/reference48';status=read(base/'training_status.json');pointer=read(base/'checkpoints/best.json')
    if status['script_sha256']!=sha(RUN/'tools/train_main.py') or status['manifest_sha256']!=sha(RUN/'data/processed/train_dev/manifest.json'):raise ValueError('FIT_CONTRACT_HASH')
    frozen={'selection':'Exact original9DEV WER then loss; no final TEST accessed',
            'manifest_sha256':status['manifest_sha256'],'trainer_sha256':status['script_sha256'],
            'training_plan_sha256':sha(RUN/'training_plan.v2.json'),
            'models':{'reference48':{'pointer':pointer,'status_sha256':sha(base/'training_status.json'),'config_sha256':sha(base/'config.json')}}}
    if not (RUN/'model_selection_freeze.json').exists():write(RUN/'model_selection_freeze.json',frozen)
    else:
        if read(RUN/'model_selection_freeze.json')!=frozen:raise ValueError('SELECTION_FREEZE_CHANGED')
    candidate,state=measured_model(base,pointer,dev,data)
    baseline_pointer=read(SOURCE/'models/reference48/checkpoints/best.json')
    baseline,_=measured_model(SOURCE/'models/reference48',baseline_pointer,dev,data)
    if candidate['predictions']!=read(base/'dev_best.json')['predictions'] or baseline['predictions']!=read(SOURCE/'models/reference48/dev_best.json')['predictions']:raise ValueError('CPU_GPU_FROZEN_PREDICTION_PARITY')
    logs=list(csv.DictReader((base/'training_log.csv').open(newline='')))
    if len(logs)!=status['completed_epochs']:raise ValueError('EPOCH_LOG_COUNT')
    epoch_total=sum(float(r['epoch_seconds']) for r in logs)
    candidate.update(completed_epochs=status['completed_epochs'],selected_epoch=pointer['epoch'],
                     measured_fit_reload_seconds=status['seconds'],epoch_median_seconds=statistics.median(float(r['epoch_seconds']) for r in logs),
                     epoch_min_seconds=min(float(r['epoch_seconds']) for r in logs),epoch_max_seconds=max(float(r['epoch_seconds']) for r in logs),
                     peak_pytorch_allocated_vram_bytes=status['peak_allocated_vram_bytes'],peak_pytorch_reserved_vram_bytes=status['peak_reserved_vram_bytes'],
                     unique_train_utterances_consumed=status['unique_train_utterances_consumed'],train_utterance_presentations=status['train_utterance_presentations'],
                     train_utterances_per_complete_epoch_second=status['train_utterance_presentations']/epoch_total,
                     train_observations_per_complete_epoch_second=status['train_observations_per_complete_epoch']*len(logs)/epoch_total,
                     final_training_ctc_loss=float(logs[-1]['train_ctc_loss']),worker_memory=status['worker_memory'])
    model=CausalCTC(len(state['vocabulary']),48,state['model']['mean'].numpy(),state['model']['std'].numpy());model.load_state_dict(state['model'])
    start=time.perf_counter();training_result=evaluate(model,fit,data,state['vocabulary'],'cpu')
    write(RUN/'selected_checkpoint_train_diagnostic.json',{'scope':'Post-selection TRAIN inference only; does not change frozen model selection.',
          'seconds':time.perf_counter()-start,'selection_freeze_sha256':sha(RUN/'model_selection_freeze.json'),'result':training_result})
    latest_pointer=read(base/'checkpoints/latest.json');latest_path=base/latest_pointer['path']
    if sha(latest_path)!=latest_pointer['sha256']:raise ValueError('LATEST_DIAGNOSTIC_CHECKPOINT_HASH')
    if latest_pointer['sha256']==pointer['sha256']:
        latest_train=training_result
    else:
        latest_state=torch.load(latest_path,map_location='cpu',weights_only=False)
        model.load_state_dict(latest_state['model'])
        latest_train=evaluate(model,fit,data,latest_state['vocabulary'],'cpu')
    write(RUN/'last_checkpoint_train_diagnostic.json',{'scope':'Post-selection final-epoch TRAIN inference only; no candidate reselection.',
          'checkpoint_path':str(latest_path),'checkpoint_sha256':latest_pointer['sha256'],'epoch':latest_pointer['epoch'],
          'selection_freeze_sha256':sha(RUN/'model_selection_freeze.json'),'result':latest_train})
    label_counts=Counter(t for r in fit for t in r['glosses']);distribution={}
    mean=state['model']['mean'].numpy();std=state['model']['std'].numpy()
    for population,selected in [('train',fit),('fixed9dev',dev)]:
        clipped=0;values=0;observations=0
        for row in selected:
            x=data[row['sample_id']];clipped+=int((np.abs((x-mean)/std)>10).sum());values+=x.size;observations+=len(x)
        distribution[population]={'utterances':len(selected),'observations':observations,'feature_value_fraction_clipped_at10':clipped/values}
    write(RUN/'post_selection_data_diagnostic.json',{'selection_freeze_sha256':sha(RUN/'model_selection_freeze.json'),
          'label_counts':{'unique_train_labels':len(label_counts),'singleton_train_labels':sum(n==1 for n in label_counts.values()),
                          'train_target_tokens':sum(label_counts.values()),'dev_tokens_unseen_in_train':sum(t not in label_counts for r in dev for t in r['glosses']),
                          'dev_tokens_with_train_frequency_at_most5':sum(label_counts.get(t,0)<=5 for r in dev for t in r['glosses'])},
          'feature_distributions':distribution,'scope':'Read-only post-selection diagnosis; cannot uniquely identify a regression cause; no tuning or label changes.'})
    result={'status':'PASS_REAL_MAIN_FIT_AND_FROZEN_DEV_COMPARISON','candidate':candidate,'frozen_baseline':baseline,
            'beats_frozen_baseline':candidate['gloss_wer']<baseline['gloss_wer'],'train':len(fit),'dev':len(dev),'test_evaluated':0,
            'training_pool':read(RUN/'dataset_plan.json')['training_pool'],'single_label_pool_minimum_met':len(fit)>=2000,
            'signer_independent':False,'production_promoted':False,'normalization':'This frozen TRAIN pool only',
            'selected_checkpoint_full_train_wer':training_result['gloss_wer'],'last_checkpoint_full_train_wer':latest_train['gloss_wer'],
            'evaluation':'Same9DEV literal references/features/greedyCTC,2CPU threads,5warmup+20repeat/sequence. ASLLRP3→olderNCSLGR is a limited cross-corpus diagnostic; no label rewriting.',
            'timing_scope':'Windows CPU model only; excludes capture, pose/hand preprocessing, queue and delivery. Not Android latency/FPS.',
            'timing_unit':'One complete saved9DEV feature sequence per forward call; not a single video frame or a final streaming commit.',
            'CPU_measurement_source_sha256':sha(source_path)}
    write(RUN/'comparison_dev.json',result)
    print(json.dumps({'train':len(fit),'dev':len(dev),'epochs':status['completed_epochs'],'dev_wer':candidate['gloss_wer'],
          'baseline_dev_wer':baseline['gloss_wer'],'selected_checkpoint_train_wer':training_result['gloss_wer']}),flush=True)

if __name__=='__main__':main()
