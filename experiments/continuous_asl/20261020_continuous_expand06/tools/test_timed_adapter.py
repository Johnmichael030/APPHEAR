"""Synthetic adapter safety checks; never model-evaluation evidence."""
from dataclasses import dataclass
import numpy as np
import pytest
from prepare_timed import checked_slice

@dataclass
class SyntheticRaw:
    timestamp_ms: np.ndarray
    frame_index: np.ndarray
    metadata: dict
    def validate(self):
        if not np.all(np.diff(self.timestamp_ms)>0):raise ValueError('TIMESTAMPS')

def fixture():
    return SyntheticRaw(np.array([0.,100.,200.,300.,400.]), np.arange(5), {})

def test_absolute_time_half_open_source_span():
    raw,mask=checked_slice(fixture(),{'start_ms':100,'end_ms':400,'sign_times_ms':[[150,200],[250,350]]})
    assert raw.timestamp_ms.tolist()==[100,200,300]
    assert raw.frame_index.tolist()==[1,2,3]
    assert mask.tolist()==[False,True,True,True,False]

@pytest.mark.parametrize('row',[
    {'start_ms':100,'end_ms':100,'sign_times_ms':[]},
    {'start_ms':100,'end_ms':400,'sign_times_ms':[[50,150]]},
    {'start_ms':100,'end_ms':400,'sign_times_ms':[[300,450]]},
    {'start_ms':100,'end_ms':150,'sign_times_ms':[[110,120]]},
])
def test_invalid_spans_rejected_without_clipping(row):
    with pytest.raises(ValueError):checked_slice(fixture(),row)
