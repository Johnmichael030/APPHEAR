"""Summarize completed runs without fitting or modifying their frozen evidence."""
import csv
import json
from pathlib import Path
import statistics
import torch
from core import RUN, ROOT, OLD, read, write, sha, SCHEMA

def main():
    out=RUN/'comparison_completed'
    if out.exists(): raise FileExistsError('COMPLETED_COMPARISON_ALREADY_EXISTS')
    out.mkdir()
    frozen=read(RUN/'model_selection_freeze.json');plan=read(RUN/'dataset_plan.json')
    checks={'data_plan':sha(RUN/'dataset_plan.json')==frozen['plan_sha256'],
            'training_plan':sha(RUN/'training_plan.v2.json')==frozen['training_plan_sha256'],
            'data_manifest':sha(RUN/'data/processed/train_dev/manifest.json')==frozen['manifest_sha256'],
            'training_script':sha(RUN/'tools/model_run.py')==frozen['script_sha256'],
            'preprocessing_script':sha(RUN/'tools/core.py')==frozen['core_script_sha256']}
    rows=read(RUN/'data/processed/train_dev/manifest.json')
    frames={split:sum(r['observations'] for r in rows if r['split']==split) for split in ('train','dev')}
    configs={};states={};result={};throughput=[]
    for name,info in frozen['models'].items():
        folder=RUN/'models'/name;pointer=read(folder/'checkpoints/best.json');path=folder/pointer['path']
        checks[name+'_pointer']=pointer==info['pointer']
        checks[name+'_checkpoint']=sha(path)==pointer['sha256']
        checks[name+'_status']=sha(folder/'training_status.json')==info['status_sha256']
        checks[name+'_config']=sha(folder/'config.json')==info['config_sha256']
        if not all(checks.values()):raise ValueError('FROZEN_ARTIFACT_MISMATCH')
        status=read(folder/'training_status.json');config=read(folder/'config.json');configs[name]=config
        state=torch.load(path,map_location='cpu',weights_only=False);states[name]=state
        log=list(csv.DictReader((folder/'training_log.csv').open(newline='')))
        assert len(log)==status['completed_epochs'] and len(list((folder/'checkpoints').glob('epoch_*.pt')))==len(log)
        best=min(log,key=lambda x:(float(x['dev_wer']),float(x['dev_loss'])))
        assert int(best['epoch'])==pointer['epoch'] and state['epoch']==pointer['epoch']
        assert state['manifest_sha256']==frozen['manifest_sha256'] and state['config']==config
        epoch_times=[float(r['epoch_seconds']) for r in log]
        for record in log:
            throughput.append({'model':name,'epoch':int(record['epoch']),'epoch_seconds':float(record['epoch_seconds']),
                'train_sequences':100,'dev_sequences':9,'train_observations':frames['train'],'dev_observations':frames['dev'],
                'train_observations_per_train_dev_checkpoint_epoch_second':frames['train']/float(record['epoch_seconds']),
                'train_dev_observations_per_complete_epoch_second':sum(frames.values())/float(record['epoch_seconds'])})
        result[name]={'status':status['status'],'completed_epochs':status['completed_epochs'],'selected_epoch':pointer['epoch'],
            'dev':read(folder/'dev_best.json'),'measured_training_loop_and_reload_seconds':status['seconds'],
            'epoch_seconds_sum':sum(epoch_times),'epoch_seconds_median':statistics.median(epoch_times),
            'epoch_seconds_min':min(epoch_times),'epoch_seconds_max':max(epoch_times),
            'train_observations_per_train_dev_checkpoint_epoch_second':frames['train']*len(log)/sum(epoch_times),
            'train_dev_observations_per_complete_epoch_second':sum(frames.values())*len(log)/sum(epoch_times),
            'peak_pytorch_allocated_vram_bytes':status['peak_allocated_vram_bytes'],
            'peak_pytorch_reserved_vram_bytes':status['peak_reserved_vram_bytes'],
            'minimum_measured_free_vram_bytes':min(int(r['free_vram_bytes']) for r in log),
            'peak_worker_working_set_bytes':status['worker_memory']['peak_working_set_bytes'],
            'gpu_devices':{k:sorted({r[k] for r in log}) for k in ('input_device','logits_device','loss_device','gradient_device','optimizer_state_device')},
            'checkpoint_path':str(path),'checkpoint_sha256':pointer['sha256'],'parameter_count':status['parameter_count'],
            'vocabulary_size':status['vocabulary_size'],'memory_failure_record_present':(folder/'failure.json').exists(),
            'checkpoint_reload_predictions_match':status['checkpoint_reload_predictions_match']}
    assert states['reference48']['vocabulary']==states['compact24']['vocabulary']
    assert torch.equal(states['reference48']['model']['mean'],states['compact24']['model']['mean'])
    assert torch.equal(states['reference48']['model']['std'],states['compact24']['model']['std'])
    equal_keys=('seed','batch','gradient_accumulation','lr','weight_decay','mixed_precision','patience','max_epochs',
                'schema','effective_receptive_field_observations','normalization','selection')
    assert all(configs['reference48'][k]==configs['compact24'][k] for k in equal_keys)
    fairness={'partitions_identical':True,'vocabulary_identical':True,'normalization_buffers_identical':True,
              'shared_config_values_identical':list(equal_keys),'decoder':frozen['decoder'],'only_planned_model_size_differs':True}
    final=read(RUN/'final_results.json') if (RUN/'final_results.json').exists() else None
    quality=read(RUN/'data/processed/train_dev/quality.json')
    comparison={'status':'PASS_COMPLETED_DIAGNOSTIC_COMPARISON','models':result,'frozen_checks':checks,'fairness':fairness,
        'rows':{'train':100,'dev':9},'observations':frames,'manifest_sha256':frozen['manifest_sha256'],
        'dev_wer_delta_candidate_minus_reference':result['compact24']['dev']['gloss_wer']-result['reference48']['dev']['gloss_wer'],
        'preprocessing':quality,'final_holdout':final,'real_world_accuracy_claim':False,'production_promoted':False,
        'candidate_decision':'UNSELECTED_ACCURACY_REGRESSION','throughput_scope':'Computed from measured complete epochs incl DEV/checkpoint work on precomputed features; not RGB throughput or Android FPS',
        'training_time_scope':'Recorded fit-loop, DEV evaluation, checkpoint work and final reload; excludes interpreter startup and initial dataset/model setup',
        'vram_scope':'Measured PyTorch allocator peaks; external whole-GPU physical peak was not independently sampled',
        'failures':'Pre-fit RF121 draft failed one focused check; corrected to61 in training_plan.v2.json before fitting. No completed-model memory/numerical failure artifact.',
        'determinism':'Fixed seeds; CUDA CTC backward is warn_only/nondeterministic. No bitwise reproducibility claim.'}
    write(out/'comparison.json',comparison)
    with (out/'epoch_throughput.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(throughput[0]));w.writeheader();w.writerows(throughput)
    lines=['# Completed continuous-model comparison — executed, compact candidate unselected','',
        'Both predeclared runs were already complete when the resume request arrived. Checkpoint, status, plan, script and manifest hashes match the pre-TEST selection freeze. No training or selection was repeated. No training process was active at the resume inspection.','',
        'Same100TRAIN/9DEV source-collection-disjoint utterances,117-token TRAIN-only vocabulary, identical normalization arrays/decoding/seed and hyperparameters; only the planned channel width differs. Signer IDs are unknown; no signer-independent claim. Receptive field61.','',
        '| Model | Epochs / selected | DEV WER (S/D/I;57refs) | Fit+reload seconds | Median epoch seconds | Peak allocated VRAM MiB | Parameters |',
        '|---|---:|---|---:|---:|---:|---:|']
    for name,r in result.items():
        d=r['dev'];lines.append(f"| {name} | {r['completed_epochs']} / {r['selected_epoch']} | {d['gloss_wer']*100:.2f}% ({d['substitutions']}/{d['deletions']}/{d['insertions']}) | {r['measured_training_loop_and_reload_seconds']:.4f} | {r['epoch_seconds_median']:.4f} | {r['peak_pytorch_allocated_vram_bytes']/1048576:.2f} | {r['parameter_count']} |")
    lines+=['',f"Candidate DEV WER regressed by{comparison['dev_wer_delta_candidate_minus_reference']*100:.2f}percentage points. Both DEV sets have9OOV tokens/57references. Both stopped at registered patience10; neither exhausted its30epoch/180second bound. Referencebestepoch4 and compactbestepoch12 were frozen before customTEST acquisition.",'',
        'The surprisingly short fit times are measured small-dataset GPU loops, not estimates or complete command wall times. Per-epoch CSV includes data movement, DEV and checkpoint work. Derived throughput uses actual observation counts and complete epoch time; it excludes source RGB decoding/perception and cannot establish live15FPS. GPU input/logits/loss/gradients/Adam state were all recorded cuda:0. Peak process working set and allocator/free-memory samples are in comparison.json; no OOM/nonfinite failure was recorded.','',
        f"TRAIN+DEV preparation took{quality['elapsed_seconds']:.2f}s; new observed-frame extraction p50/p95={quality['extraction_ms_p50']:.2f}/{quality['extraction_ms_p95']:.2f}ms. Old30feature sets and their decode evidence were reused. Frozen artifacts remain unchanged.",'']
    if final:
        lines+=['CustomTEST was acquired only after the recorded selection freeze. It has14utterances/86reference tokens from one previously untrained collection, with unknown signer clustering. These are exploratory diagnostic results, not current-app or signer-independent performance.','',
            '| Model | TEST WER | S/D/I | CPU model p50/p95 ms |','|---|---:|---|---|']
        for name,r in final['models'].items():
            lines.append(f"| {name} | {r['gloss_wer']*100:.2f}% | {r['substitutions']}/{r['deletions']}/{r['insertions']} | {r['cpu_model_only_p50_ms']:.3f}/{r['cpu_model_only_p95_ms']:.3f} |")
        a=final['models']['reference48'];b=final['models']['compact24'];speed=1-b['cpu_model_only_p50_ms']/a['cpu_model_only_p50_ms']
        lines+=['',f"Compact CPU median model latency is{speed*100:.2f}% lower, but TEST WER is worse by{(b['gloss_wer']-a['gloss_wer'])*100:.2f}points. Timing uses the same9DEV input sequences,2CPU threads,5warmups/20repeats per sequence (180measurements/model). It excludes perception and normal-app delivery. No claim of joint speed/accuracy improvement or15completedFPS on Android."]
        old=final['unchanged_legacy_baseline'];lines+=['',f"The unchanged older39-token baseline scores{old['gloss_wer']*100:.2f}% WER on the same customTEST, with{old['out_of_vocabulary_tokens']}OOV reference tokens. Its restricted vocabulary differs from the matched117-token comparison, so that contrast is not a controlled architecture ablation. Its checkpoint is unchanged."]
    lines+=['','Full predictions, losses, uncertainty limits, checkpoint paths and SHA-256 values are in comparison.json and ../final_results.json. Authentic source index strings are retained verbatim, including grammar/locus/HOLD notation; this is literal source-main-gloss-token transcription. Individual boundaries/no-sign exposure truth and normal-app latency remain unavailable.','',
        'Exact safe commands: Invoke-Research.ps1 -Phase Status; -Phase Reference -Resume and -Phase Candidate -Resume only verify frozen completed checkpoints and return FROZEN_COMPLETE_NO_TRAINING. Do not launch a new fit on these frozen decisions.']
    with (out/'comparison_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':comparison['status'],'dev_wer_delta_pp':comparison['dev_wer_delta_candidate_minus_reference']*100,
        'models':{k:{z:r[z] for z in ('completed_epochs','selected_epoch','measured_training_loop_and_reload_seconds','epoch_seconds_median','train_observations_per_train_dev_checkpoint_epoch_second')} for k,r in result.items()}}))

if __name__=='__main__':main()
