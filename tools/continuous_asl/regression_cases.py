"""Reproduce historical hazards only on tiny fixtures inside this run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from archive_safety import assert_disjoint
from event_guard import retained_event_decision
from pipeline.perception_vnext.recognition_window_materializer import ObservationRingBuffer
from pipeline.perception_vnext.segmentation.streaming_decoder import CausalStreamingSignEventDecoder, StreamingDecoderConfig
from pipeline.ingest.merge import run as legacy_merge

EVIDENCE={}

class Regressions(unittest.TestCase):
    def test_a_original_long_active_then_guard(self):
        decoder=CausalStreamingSignEventDecoder(StreamingDecoderConfig(minimum_active_ms=100,pre_gap_reset_ms=300,post_confirm_ms=300))
        ring=ObservationRingBuffer(max_observations=256,max_age_ms=30000)
        event=None
        for i in range(164):
            ms=i*200.; state=1 if i<161 else 2
            ring.append(i,ms,None)
            result=decoder.consume(ms,[0.,float(state==1),float(state==2)],state)
            if result: event=result
        self.assertIsNotNone(event)
        retained=[r.source_index for r in ring.snapshot()]
        self.assertNotIn(event.logical_active_start_index,retained)
        required=(event.logical_active_start_index,event.logical_active_end_index,event.first_post_index,event.emitted_at_index)
        # This is the Kotlin materializer precondition, exercised with Python-generated indices.
        with self.assertRaises(AssertionError): self.assertTrue(all(i in retained for i in required))
        decision=retained_event_decision(required,retained)
        self.assertEqual(decision.status,'EXPIRED_EVENT'); self.assertFalse(decision.emit_sign)
        EVIDENCE['A']={'original':'REPRODUCED_REFERENCE_PLUS_KOTLIN_PRECONDITION','event_start':required[0],
            'retained_first':retained[0],'evicted':ring.evicted_count,'guard':decision.status,'android_executed':False}

    def test_a_exact_age_boundary(self):
        ring=ObservationRingBuffer(max_observations=256,max_age_ms=30000)
        for i,ms in enumerate((0.,10000.,20000.,30000.)): ring.append(i,ms,None)
        self.assertEqual(retained_event_decision((0,1,2,3),[r.source_index for r in ring.snapshot()]).status,'MATERIALIZE')
        ring.append(4,30000.001,None)
        self.assertEqual(retained_event_decision((0,1,2,4),[r.source_index for r in ring.snapshot()]).status,'EXPIRED_EVENT')

    def test_a_count_eviction_and_missing_commit(self):
        ring=ObservationRingBuffer(max_observations=4,max_age_ms=30000)
        for i in range(5): ring.append(i,float(i),None)
        self.assertEqual(retained_event_decision((0,2,3,4),[r.source_index for r in ring.snapshot()]).status,'EXPIRED_EVENT')
        self.assertEqual(retained_event_decision((1,2,3,4),(1,2,3)).status,'EXPIRED_EVENT')
        self.assertEqual(retained_event_decision((1,2,3,4),(1,2,3,4)).status,'MATERIALIZE')

    def test_a_guard_never_commits_and_validates(self):
        self.assertFalse(retained_event_decision((0,1,2,3),range(4)).emit_sign)
        with self.assertRaises(ValueError): retained_event_decision((0,2,1,3),range(4))
        with self.assertRaises(ValueError): retained_event_decision((-1,1,2,3),range(4))

    def test_e_original_overwrite_on_disposable_fixture(self):
        fixture=FIXTURES/'merge_overlap'; fixture.mkdir()
        source=fixture/'source'; source.mkdir()
        arrays=dict(X=np.ones((1,2,3),np.float32),frame_mask=np.ones((1,2),bool),
            lengths=np.array([2],np.int32),y=np.array([0],np.int32),participant_ids=np.array(['synthetic_regression']),t_max=np.array(2))
        for split in ('train','val','test'): np.savez_compressed(source/f'{split}.npz',**arrays)
        record=fixture/'recorded.npz'; np.savez_compressed(record,**arrays)
        vocab=fixture/'vocab.json'; vocab.write_text('{"SYNTHETIC_REGRESSION_ONLY":0}')
        before=hashlib.sha256((source/'train.npz').read_bytes()).hexdigest()
        self.assertEqual(legacy_merge(record,'supplement',source,source,vocab),0)
        after=hashlib.sha256((source/'train.npz').read_bytes()).hexdigest()
        self.assertNotEqual(before,after)
        with np.load(source/'train.npz',allow_pickle=False) as data: self.assertEqual(len(data['y']),2)
        EVIDENCE['E']={'original':'OVERWRITE_REPRODUCED_ONLY_ON_NEW_SYNTHETIC_FIXTURE',
            'source_train_hash_before':before,'after':after,'new_guard':'REJECTS_IDENTICAL_AND_NESTED'}

    def test_e_new_guard(self):
        source=FIXTURES/'disjoint_source'; output=FIXTURES/'disjoint_output'
        for bad in (source,source/'child',source.parent):
            with self.assertRaises(ValueError): assert_disjoint([source],bad)
        assert_disjoint([source],output)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--case-tag',default='hazards_02'); args=p.parse_args()
    if not args.case_tag.replace('_','').isalnum(): raise ValueError('CASE_TAG')
    run=args.run_root.resolve(); run.relative_to(ROOT/'experiments/continuous_asl')
    FIXTURES=run/'regressions'/args.case_tag; FIXTURES.mkdir()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Regressions)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    EVIDENCE.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        production_modified=False,guard_integrated=False,
        B='SOURCE_EVIDENCE_ONLY; NO LABELED NEGATIVES FOR CALIBRATION',
        C='MEASURED_SEPARATELY_IN_PREPROCESSING_AND_PERFORMANCE',D='SOURCE_AUDIT_ONLY; NO RUNTIME_FLAG_FIX')
    with (run/'logs'/f'regression_{args.case_tag}.json').open('x') as f: json.dump(EVIDENCE,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
