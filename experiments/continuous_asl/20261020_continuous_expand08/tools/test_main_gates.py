"""Data invariants for the new single-pool fit; no training or TEST inference."""
from collections import Counter
import numpy as np
from core import RUN,SOURCE,SIX,SEVEN,SCHEMA,read,sha

def test_learnability_gate_is_eligible_and_measured():
    gate=read(SEVEN/'overfit_diagnostic_v2/result.json')
    assert gate['used_cuda'] and gate['checkpoint_reload_predictions_match']
    assert gate['best_train']['gloss_wer']<=.05 and gate['loss_ratio']<=.25
    assert sha(gate['checkpoint_path'])==gate['checkpoint_sha256']
    blocked=set(read(SIX/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
    assert not blocked&{r['sample_id'] for r in read(SEVEN/'overfit_diagnostic_v2/manifest.json')}

def test_frozen_single_pool_and_original9dev():
    rows=read(RUN/'data/processed/train_dev/manifest.json');fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    old=[r for r in read(SOURCE/'data/processed/train_dev/manifest.json') if r['split']=='dev']
    assert len({r['training_pool'] for r in fit})==1
    assert [(r['sample_id'],r['glosses'],r['feature_sha256']) for r in dev]==[(r['sample_id'],r['glosses'],r['feature_sha256']) for r in old]
    assert len(dev)==9 and not (RUN/'data/processed/test').exists()
    assert not {r['sample_id'] for r in fit}&{r['sample_id'] for r in dev}

def test_real_target_encoder_has_no_blank_or_unknown_in_train_targets():
    from ctc_experiment import batch
    rows=[r for r in read(RUN/'data/processed/train_dev/manifest.json') if r['split']=='train']
    vocabulary=['<BLANK>','<UNK>']+sorted({t for r in rows for t in r['glosses']})
    data={}
    for row in rows[:2]:
        with np.load(row['feature_path'],allow_pickle=False) as f:data[row['sample_id']]=f['X'].copy()
    x,target,lengths,sizes=batch(rows[:2],data,vocabulary,'cpu')
    assert x.shape[2]==378 and bool((target>=2).all())
    assert target.tolist()==[vocabulary.index(t) for r in rows[:2] for t in r['glosses']]
    assert sizes.tolist()==[len(r['glosses']) for r in rows[:2]]
    assert lengths.tolist()==[len(data[r['sample_id']]) for r in rows[:2]]

def test_synthetic_decoder_repeat_blank_conventions_only():
    import torch
    from ctc_experiment import decode
    # Synthetic logits are a decoder unit test, never evaluation evidence.
    logits=torch.full((6,3),-20.)
    for i,token in enumerate([2,2,0,2,1,1]):logits[i,token]=20.
    assert decode(logits,['<BLANK>','<UNK>','LABEL'])==['LABEL','LABEL','<UNK>']
