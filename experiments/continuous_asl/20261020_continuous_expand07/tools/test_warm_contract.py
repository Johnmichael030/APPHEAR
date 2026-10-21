"""Warm-start invariant tests; synthetic label used only for transplant testing."""
import numpy as np
import torch
from core import SOURCE, RUN, read
from warm_start import baseline_state,build,extended_vocabulary,parity
from ctc_experiment import seeds

def test_exact_token_rows_norm_and_new_row_initialization():
    seeds(20261020)
    state,_,_=baseline_state()
    rows=[r for r in read(SOURCE/'data/processed/train_dev/manifest.json') if r['split']=='train']
    vocabulary=extended_vocabulary(rows+[{'glosses':['SYNTHETIC_TRANSPLANT_TEST_ONLY']}])
    model=build(vocabulary,'cpu')
    assert vocabulary[:len(state['vocabulary'])]==state['vocabulary']
    assert torch.equal(model.output.weight[:len(state['vocabulary'])],state['model']['output.weight'])
    assert torch.equal(model.output.bias[:len(state['vocabulary'])],state['model']['output.bias'])
    assert torch.equal(model.mean,state['model']['mean']) and torch.equal(model.std,state['model']['std'])
    assert torch.count_nonzero(model.output.weight[-1])==0 and model.output.bias[-1]==-10

def test_pre_fit_parity_and_frozen_split():
    seeds(20261020)
    rows=read(RUN/'data/processed/train_dev/manifest.json');fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    vocabulary=extended_vocabulary(fit);model=build(vocabulary,'cpu');data={}
    for row in dev:
        with np.load(row['feature_path'],allow_pickle=False) as f:data[row['sample_id']]=f['X'].copy()
    result=parity(model,dev,data,vocabulary,'cpu')
    assert result['status']=='PASS_PRE_FIT_BASELINE_PARITY' and result['dev_utterances']==9
    assert not (RUN/'data/processed/test').exists()
