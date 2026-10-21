"""Engineering checks, not recognition accuracy tests."""
import numpy as np
import torch
from ctc_experiment import CausalCTC,decode,edit_counts

def test_edit_counts_substitution_insertion_deletion():
    assert edit_counts(['A','B'],['A','C'])==dict(substitutions=1,deletions=0,insertions=0,reference_tokens=2)
    assert edit_counts(['A','B'],[])==dict(substitutions=0,deletions=2,insertions=0,reference_tokens=2)
    assert edit_counts([],['A'])==dict(substitutions=0,deletions=0,insertions=1,reference_tokens=0)

def test_ctc_blank_separates_repeated_gloss():
    logits=torch.full((6,3),-10.)
    for t,i in enumerate((1,1,0,1,2,2)): logits[t,i]=10.
    assert decode(logits,['<BLANK>','A','B'])==['A','A','B']

def test_causal_prefix_and_padding_do_not_change_past():
    torch.set_num_threads(2);torch.manual_seed(7)
    model=CausalCTC(9,8,np.zeros(378,np.float32),np.ones(378,np.float32)).eval()
    x=torch.randn(1,90,378)
    with torch.inference_mode():
        all_outputs=model(x)
        prefix_outputs=model(x[:,:45])
        changed=x.clone();changed[:,45:]=100.
        altered=model(changed)
    torch.testing.assert_close(all_outputs[:,:45],prefix_outputs,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(all_outputs[:,:45],altered[:,:45],rtol=0,atol=0)

def test_no_nonfinite_output_from_explicit_missingness_zero_features():
    model=CausalCTC(5,8,np.zeros(378,np.float32),np.ones(378,np.float32)).eval()
    with torch.inference_mode(): result=model(torch.zeros(1,20,378))
    assert result.shape==(1,20,5) and torch.isfinite(result).all()

def test_61_observation_history_is_sufficient_beyond_initial_clip_lengths():
    torch.manual_seed(19)
    model=CausalCTC(7,8,np.zeros(378,np.float32),np.ones(378,np.float32)).eval()
    x=torch.randn(1,120,378)  # Synthetic engineering probe, never scored as ASL.
    with torch.inference_mode():
        full=model(x)[:,-1]
        bounded=model(x[:,-61:])[:,-1]
    torch.testing.assert_close(full,bounded,rtol=1e-5,atol=1e-6)
