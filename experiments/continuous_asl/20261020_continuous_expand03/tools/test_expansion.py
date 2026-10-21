"""Focused source/split/model checks for the newly expanded continuous task."""
import csv
import numpy as np
import torch
from core import RUN, OLD, read
from ctc_experiment import CausalCTC, edit_counts

def test_source_labels_are_verbatim_official_index_glosses():
    plan=read(RUN/'dataset_plan.json')
    rows=list(csv.DictReader(open(plan['index_path'],encoding='utf-8-sig',newline='')))
    originals={(r['XML file'],r['Occurs in utterance id']):r['main gloss'] for r in rows if r['Perspective/Camera id']=='0'}
    for row in plan['rows']:
        assert row['glosses']==originals[row['collection'],row['utterance_id']].split()
        assert row['original_split']=='NOT_PROVIDED_IN_PUBLIC_INDEX'

def test_custom_split_and_old_validation_quarantine():
    plan=read(RUN/'dataset_plan.json');rows=plan['rows']
    assert plan['split_counts']=={'train':100,'dev':9,'test':14}
    old=read(OLD/'data/processed/full/manifest.json')
    quarantine={r['collection'] for r in old if r['split']=='val'}
    assert not {r['collection'] for r in rows}&quarantine
    for key in ('collection','source_group','source_url','sample_id'):
        sets={s:{r[key] for r in rows if r['split']==s} for s in ('train','dev','test')}
        assert not sets['train']&sets['dev'] and not sets['train']&sets['test'] and not sets['dev']&sets['test']

def test_reuse_is_old_training_only():
    previous={r['sample_id'] for r in read(OLD/'data/processed/full/manifest.json') if r['split']=='train'}
    reused=[r for r in read(RUN/'dataset_plan.json')['rows'] if r['reuse']]
    assert len(reused)==30 and {r['sample_id'] for r in reused}==previous
    assert all(r['split']=='train' for r in reused)

def test_actual_receptive_field_is61():
    model=CausalCTC(5,8,np.zeros(378,np.float32),np.ones(378,np.float32))
    assert 1+sum(2*block.pad for block in model.blocks)==61
    assert read(RUN/'training_plan.v2.json')['effective_receptive_field_observations']==61

def test_model_prefix_is_causal():
    torch.set_num_threads(2);torch.manual_seed(1)
    model=CausalCTC(5,8,np.zeros(378,np.float32),np.ones(378,np.float32)).eval()
    x=torch.randn(1,160,378)
    with torch.inference_mode():
        full=model(x)[:,:80];prefix=model(x[:,:80])
    assert torch.allclose(full,prefix,atol=1e-5,rtol=0)

def test_wer_denominator_and_substitution_deletion_insertion():
    assert edit_counts(['A','B','C'],['A','X'])=={'substitutions':1,'deletions':1,'insertions':0,'reference_tokens':3}
    assert edit_counts(['A'],['A','X'])=={'substitutions':0,'deletions':0,'insertions':1,'reference_tokens':1}
