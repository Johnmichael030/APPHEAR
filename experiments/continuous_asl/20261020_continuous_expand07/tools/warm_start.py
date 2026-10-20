"""Exact-token baseline transplant, preserving its normalization and backbone."""
import numpy as np
import torch
from core import SOURCE, read, sha
from ctc_experiment import CausalCTC, decode

def baseline_state():
    pointer=read(SOURCE/'models/reference48/checkpoints/best.json')
    path=SOURCE/'models/reference48'/pointer['path']
    if sha(path)!=pointer['sha256']:raise ValueError('FROZEN_BASELINE_HASH')
    return torch.load(path,map_location='cpu',weights_only=False),str(path),pointer['sha256']

def extended_vocabulary(fit):
    state,_,_=baseline_state();old=state['vocabulary']
    labels={s for row in fit for s in row['glosses']}
    if not set(old[2:])<=labels:raise ValueError('ORIGINAL_TRAIN_LABELS_NOT_PRESERVED')
    return old+sorted(labels-set(old))

def build(vocabulary,device):
    state,_,_=baseline_state()
    model=CausalCTC(len(vocabulary),48,state['model']['mean'].numpy(),state['model']['std'].numpy())
    partial={k:v for k,v in state['model'].items() if not k.startswith('output.')}
    missing,unexpected=model.load_state_dict(partial,strict=False)
    if set(missing)!={'output.weight','output.bias'} or unexpected:raise ValueError('BACKBONE_TRANSPLANT_KEYS')
    mapping={token:i for i,token in enumerate(vocabulary)}
    with torch.no_grad():
        model.output.weight.zero_();model.output.bias.fill_(-10.)
        for i,token in enumerate(state['vocabulary']):
            model.output.weight[mapping[token]].copy_(state['model']['output.weight'][i])
            model.output.bias[mapping[token]].copy_(state['model']['output.bias'][i])
    return model.to(device)

def parity(model,rows,data,vocabulary,device):
    state,path,checksum=baseline_state()
    reference=CausalCTC(len(state['vocabulary']),48,state['model']['mean'].numpy(),state['model']['std'].numpy()).to(device)
    reference.load_state_dict(state['model']);reference.eval();model.eval()
    indices=[vocabulary.index(t) for t in state['vocabulary']];maximum=0.
    with torch.inference_mode():
        for row in rows:
            x=torch.as_tensor(data[row['sample_id']][None],device=device)
            a=reference(x)[0];b=model(x)[0]
            maximum=max(maximum,float((a-b[:,indices]).abs().max()))
            if not torch.allclose(a,b[:,indices],atol=1e-5,rtol=1e-5):raise ValueError('OLD_LOGIT_PARITY')
            if decode(a,state['vocabulary'])!=decode(b,vocabulary):raise ValueError('BASELINE_GREEDY_PARITY')
    if not torch.equal(model.mean.cpu(),state['model']['mean']) or not torch.equal(model.std.cpu(),state['model']['std']):raise ValueError('BASELINE_NORMALIZATION_PARITY')
    return {'status':'PASS_PRE_FIT_BASELINE_PARITY','dev_utterances':len(rows),'maximum_mapped_logit_error':maximum,
            'baseline_path':path,'baseline_sha256':checksum,'normalization':'UNCHANGED_ORIGINAL_TRAIN100',
            'new_output_bias':-10.,'new_output_weights':'zero','test_accessed':False}
