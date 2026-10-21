import numpy as np
import pytest
import torch
from warmstart_a import LegacyTCN
from input_contract import validate
from train_a import atomic_checkpoint

def test_padded_suffix_cannot_change_valid_clip_decision():
    torch.set_num_threads(2);torch.manual_seed(13);model=LegacyTCN(.001).eval();x=torch.randn(1,96,258)
    mask=torch.zeros(1,96,dtype=torch.bool);mask[:,:30]=True
    with torch.inference_mode():
        padded=model(x,mask);trimmed=model(x[:,:30],mask[:,:30])
    torch.testing.assert_close(padded,trimmed,rtol=1e-5,atol=1e-6)

def test_full_causal_prefix_and_bounded_history():
    model=LegacyTCN(.001).eval();x=torch.randn(1,160,258)
    with torch.inference_mode():
        full=model.per_frame(x);prefix=model.per_frame(x[:,:70]);bounded=model.per_frame(x[:,-61:])
    torch.testing.assert_close(full[:,:70],prefix,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(full[:,-1],bounded[:,-1],rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize('kind',['schema','empty','long','nan','width','mask'])
def test_invalid_inputs_are_rejected(kind):
    x=np.zeros((1,30,258),np.float32);m=np.ones((1,30),bool);schema='3.0'
    if kind=='schema':schema='recognition.v2'
    elif kind=='empty':m[:]=False
    elif kind=='long':x=np.zeros((1,160,258),np.float32);m=np.ones((1,160),bool)
    elif kind=='nan':x[0,0,0]=np.nan
    elif kind=='width':x=np.zeros((1,30,350),np.float32)
    elif kind=='mask':m=m.astype(np.float32)
    with pytest.raises(ValueError):validate(x,m,schema)

def test_atomic_checkpoints_preserve_completed_file(tmp_path):
    path=tmp_path/'epoch_001.pt';atomic_checkpoint(path,{'weight':torch.tensor([3.])});before=path.read_bytes()
    with pytest.raises(FileExistsError):atomic_checkpoint(path,{'weight':torch.tensor([9.])})
    assert path.read_bytes()==before and not path.with_suffix('.pt.partial').exists()
