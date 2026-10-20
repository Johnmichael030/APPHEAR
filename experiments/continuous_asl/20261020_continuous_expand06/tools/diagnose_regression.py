"""Read-only measured TRAIN/DEV diagnosis; held-out TEST never loaded."""
from collections import Counter
import json
import numpy as np
import torch
from core import RUN, ROOT, THREE, FOUR, FIVE, SCHEMA, read, write, sha
from ctc_experiment import CausalCTC, evaluate, seeds

def main():
    output=RUN/'wer_diagnosis.json'
    if output.exists():
        print('DIAGNOSIS_COMPLETE_NO_REPEAT');return
    seeds(20261020)
    allrows=read(FIVE/'data/processed/train_dev/manifest.json')
    data={}
    for row in allrows:
        if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
        with np.load(row['feature_path'],allow_pickle=False) as f:
            if str(f['schema_version'].item())!=SCHEMA:raise ValueError('SCHEMA')
            x=f['X'].copy();ts=f['timestamp_ms']
            if x.shape[1]!=378 or x.dtype!=np.float32 or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
        if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_FEASIBILITY')
        data[row['sample_id']]=x
    original_ids={r['sample_id'] for r in read(THREE/'data/processed/train_dev/manifest.json') if r['split']=='train'}
    shared=[r for r in allrows if r['sample_id'] in original_ids]
    dev=[r for r in allrows if r['split']=='dev']
    if len(shared)!=100 or len(dev)!=9:raise ValueError('FROZEN_SPLIT')
    results={};datasets={}
    for tag,base in [('100',THREE),('181',FOUR),('455',FIVE)]:
        fit=[r for r in read(base/'data/processed/train_dev/manifest.json') if r['split']=='train']
        tokens=Counter(s for r in fit for s in r['glosses'])
        trainframes=np.concatenate([data[r['sample_id']] for r in fit])
        datasets[tag]={'train_utterances':len(fit),'target_tokens':sum(tokens.values()),'authentic_labels':len(tokens),
                       'singleton_labels':sum(n==1 for n in tokens.values()),'fingerspelled_word_tokens':sum(n for t,n in tokens.items() if 'fs-' in t),
                       'train_observations':len(trainframes),'train_hand_observed_fraction':trainframes[:,172:174].mean(0).tolist(),
                       'dev_hand_observed_fraction':np.concatenate([data[r['sample_id']] for r in dev])[:,172:174].mean(0).tolist()}
        for name in ('reference48','compact24'):
            if tag=='100' and name=='compact24':continue
            pointer=read(base/'models'/name/'checkpoints/best.json');cp=base/'models'/name/pointer['path']
            if sha(cp)!=pointer['sha256']:raise ValueError('CHECKPOINT_HASH')
            state=torch.load(cp,map_location='cpu',weights_only=False)
            model=CausalCTC(len(state['vocabulary']),state['config']['channels'],state['model']['mean'].numpy(),state['model']['std'].numpy())
            model.load_state_dict(state['model']);model.eval();populations={}
            for pop,rows in [('own_train',fit),('shared_original100_train',shared),('fixed9dev',dev)]:
                scores=evaluate(model,rows,data,state['vocabulary'],'cpu');blank=0;frames=0;post=0.;clipped=0;values=0
                with torch.inference_mode():
                    for row in rows:
                        x=torch.from_numpy(data[row['sample_id']])[None];logits=model(x)[0]
                        blank+=int((logits.argmax(-1)==0).sum());frames+=len(logits);post+=float(logits.softmax(-1)[:,0].sum())
                        z=(x-model.mean)/model.std;clipped+=int((z.abs()>10).sum());values+=z.numel()
                populations[pop]={**scores,'blank_argmax_frame_fraction':blank/frames,'mean_blank_posterior':post/frames,
                                  'feature_value_fraction_clipped_at10':clipped/values,'utterances':len(rows)}
            results[tag+'_'+name]={'checkpoint_path':str(cp),'checkpoint_sha256':pointer['sha256'],'selected_epoch':pointer['epoch'],
                                  'populations':populations,'vocabulary_size':len(state['vocabulary'])}
    checks=read(RUN/'depaul_catalog/plan.json')['existing_label_checks']
    report={'status':'MEASURED_TRAIN_DEV_DIAGNOSIS','models':results,'datasets':datasets,
            'independent_ELAN_label_event_checks':{'matched':len(checks),'conflicts':sum(not r['glosses_equal'] for r in checks)},
            'verified_gates':['unchanged9DEV identities/labels/features','literal source labels and ELAN whole-event units agree on551 joins',
                              'genuine increasing feature timestamps, same378contract, CTC lengths',
                              'standard greedy repeat-collapse blank-reset decoder unchanged','no OOM/nonfinite in completed fits'],
            'limitations':['nine DEV utterances /57 target tokens; each token changes WER by1.754percentage points',
                           'source signer IDs unknown; collection disjoint only','unmatched index-only timing cannot prove visual sign alignment',
                           'several coupled dataset/vocabulary/normalization changes; diagnosis is evidence, not a causal proof'],
            'next_justified_experiment':{'method':'Baseline-preserving warm start of unchanged48-channel model, exact token-mapped head expansion, retain original TRAIN100 mean/std, new head bias-10, lr0.0001, max20epochs/180seconds/patience5.',
                'reason':'Coverage improved while models became deletion-dominant on DEV. Preserve the baseline learned representation and normalization; avoid a cold vocabulary reset while adding authentic timed TRAIN sequences.',
                'gates':['save this diagnosis before new fit','verify mapped old-logit parity and unchanged greedy DEV predictions before fitting',
                         'new manifest frozen before smoke/pilot','new complete source-verified rows only; same9DEV decoder/selection; no reserved TEST reads'],
                'expected_effect':'Test whether conservative adaptation retains baseline generalization; no guaranteed WER improvement.'},
            'test_accessed':False,'production_modified':False,'script_sha256':sha(__file__)}
    write(output,report)
    lines=['# Measured WER regression diagnosis','', '| Model | Own TRAIN WER | Shared100 TRAIN WER | Fixed9DEV WER | DEV blank frames |','|---|---:|---:|---:|---:|']
    for name,r in results.items():
        pops=r['populations'];lines.append('| '+name+' | '+' | '.join(f"{100*pops[p]['gloss_wer']:.2f}%" for p in ['own_train','shared_original100_train','fixed9dev'])+f" | {100*pops['fixed9dev']['blank_argmax_frame_fraction']:.2f}% |")
    lines+=['','All551 independent ELAN joins agree with existing target event units. No target normalization correction is justified. DEV OOV improves9→7→3, yet recognition regresses. TRAIN/DEV WER and blank fractions above quantify generalization and deletion behavior. They do not uniquely identify an optimizer or capacity cause.','',
            'The next predeclared test keeps the same causal48-channel model, baseline normalization and exact mapped head rows, and adapts at1e-4. New word rows start with bias-10. It must reproduce the baseline DEV predictions before fitting. Selection remains fixed9DEV WER then loss; no test tuning.','',
            'Signer independence, false activations, missed-sign events, visual boundary accuracy, and live Android latency remain unmeasured without independently verified annotations and consented no-sign recordings.']
    path=RUN/'wer_diagnosis.md'
    with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:{p:round(v['gloss_wer'],5) for p,v in r['populations'].items()} for k,r in results.items()}),flush=True)

if __name__=='__main__':main()
