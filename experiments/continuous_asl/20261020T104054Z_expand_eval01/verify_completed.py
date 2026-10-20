"""Read-only completion checks. No training, downloads, checkpoint loads or TEST scoring."""
from core import RUN, TREE, read, sha
from pathlib import Path
from collections import Counter
import json

def main():
    count=0
    for line in (RUN/'artifact_hashes.sha256').read_text(encoding='utf-8').splitlines():
        digest,relative=line.split('  ',1)
        p=(RUN/relative).resolve()
        assert RUN in p.parents and p.is_file(), ('MISSING_OR_OUTSIDE',relative)
        assert sha(p)==digest, ('ARTIFACT_CHANGED',relative)
        count+=1
    protected=read(TREE/'20261020T051631Z_devwer_diagnosis01/preservation_before.json')['files']
    for name,expected in protected.items():
        stat=Path(name).stat()
        assert [stat.st_size,stat.st_mtime_ns]==expected, ('PROTECTED_METADATA_CHANGED',name)
    plan=read(RUN/'dataset_plan.json'); quality=read(RUN/'eligible_corpus/quality.json')
    assert sha(plan['old_main_manifest_path'])==plan['old_main_manifest_sha256']
    assert sha(TREE/'20261020_continuous_expand06/canonical/manifest.json')==quality['parent_manifest_sha256']
    assert sha(RUN/'data/processed/train_dev/manifest.json')==plan['manifest_sha256']
    assert sha(RUN/'eligible_corpus/manifest.json')==quality['manifest_sha256']
    rows=read(RUN/'data/processed/train_dev/manifest.json')
    assert Counter(r['split'] for r in rows)=={'train':1706,'dev':413}
    train=[r for r in rows if r['split']=='train']; dev=[r for r in rows if r['split']=='dev']
    for key in ('sample_id','signer_id','collection','source_group','raw_sha256','decoded_pixel_sha256'):
        assert not ({r[key] for r in train}&{r[key] for r in dev}), ('SPLIT_OVERLAP',key)
    eligible=read(RUN/'eligible_corpus/manifest.json')
    ids={r['sample_id'] for r in eligible}
    assert len(ids)==len(eligible)==3121
    excluded=set(read(RUN/'data/processed/train_dev/quality.json')['excluded_ids'])
    excluded.update(read(RUN/'ncslgr_interval_exclusions.json')['excluded_ids'])
    assert not (excluded&ids), 'EXCLUDED_ID_REINTRODUCED'
    status=read(RUN/'pilot_fresh48_v2/training_status.json')
    assert status['used_cuda'] and status['completed_epochs']==5
    assert status['random_initialization'] and not status['old_recognition_weights_read']
    assert status['checkpoint_reload_predictions_match'] and not status['test_accessed'] and not status['production_promoted']
    assert sha(RUN/'pilot_fresh48_v2'/status['best_checkpoint']['path'])==status['best_checkpoint']['sha256']
    assert sha(RUN/'train_fresh_v2.py')==status['script_sha256']
    goal=read(RUN/'goal_update.json')
    assert goal['active_data_target']==2000 and goal['verified_eligible']==len(ids) and goal['target_met']
    print(json.dumps({'status':'PASS_READ_ONLY_ARTIFACT_VERIFICATION','new_artifacts_hashed':count,'protected_metadata_unchanged':len(protected),'eligible':len(ids),'active_target':2000,'training_epochs':5,'recognition_readiness':'PARTIAL_NO_PROMOTION','TEST_scored':False}))

if __name__=='__main__': main()
