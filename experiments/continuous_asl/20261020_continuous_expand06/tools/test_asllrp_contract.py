"""Synthetic metadata only: valid rate/frame joins and hard rejection gates."""
import pytest
from prepare_asllrp import check_media_contract

def row():
    return {'utterance_start_frame_master':19,'utterance_end_frame_master':152,
            'events':[{'start_frame_master':34,'end_frame_master':43},{'start_frame_master':57,'end_frame_master':61}]}

def test_publisher_inclusive_frame_origin_join():
    check_media_contract(row(),30000/1001,134)

@pytest.mark.parametrize('fps,frames',[(30,134),(30000/1001,133),(30000/1001,135),(0,134)])
def test_wrong_rate_or_count_rejected(fps,frames):
    with pytest.raises(ValueError):check_media_contract(row(),fps,frames)

def test_out_of_bounds_event_rejected():
    sample=row();sample['events'][1]['end_frame_master']=153
    with pytest.raises(ValueError):check_media_contract(sample,30000/1001,134)
