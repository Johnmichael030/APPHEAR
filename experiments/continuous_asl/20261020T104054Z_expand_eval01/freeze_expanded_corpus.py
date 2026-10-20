"""Freeze added eligible records without touching the original eligible catalog."""
from collections import Counter
import csv,json,sys
import numpy as np
from core import RUN,OLD,read,write,sha,SCHEMA

def main():
 destination=RUN/'eligible_corpus/manifest.json'
 if destination.exists():
  if sha(destination)!=read(RUN/'eligible_corpus/quality.json')['manifest_sha256']:raise ValueError('FROZEN_EXPANDED_CATALOG_CHANGED')
  print('EXPANDED_ELIGIBLE_CATALOG_COMPLETE');return
 parent_path=OLD/'canonical/manifest.json';parent=read(parent_path)
 if sha(parent_path)!=read(OLD/'canonical/quality.json')['manifest_sha256']:raise ValueError('PARENT_CATALOG_HASH')
 old=[r for r in parent if r['canonical_role']=='eligible_train'];new=read(RUN/'timed_preparation/manifest.json')
 if len(old)!=3025 or len(new)!=96:raise ValueError('EXPECTED_COMPLETED_COUNTS_CHANGED')
 original_ids={r['sample_id'] for r in parent};original_source_hashes={r.get('raw_sha256') for r in parent}
 old_segments={r.get('decoded_segment_observation_sha256') for r in old}-{None};old_source_groups={r['source_group'] for r in parent}
 new_segment_hashes=[];checks=[]
 for row in new:
  if row['sample_id'] in original_ids or row['source_group'] in old_source_groups or row['raw_sha256'] in original_source_hashes:raise ValueError('NEW_SOURCE_OR_ID_OVERLAPS_PARENT')
  if sha(row['feature_path'])!=row['feature_sha256']:raise ValueError('NEW_FEATURE_HASH')
  with np.load(row['feature_path'],allow_pickle=False) as f:
   if str(f['schema_version'].item())!=SCHEMA:raise ValueError('NEW_FEATURE_SCHEMA')
   x=f['X'];ts=f['timestamp_ms']
   if x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('NEW_FEATURE_CONTRACT')
   if not (x[:,172:174].any() or x[:,214].any()):raise ValueError('NEW_SEGMENT_HAS_NO_BODY_OR_HAND_OBSERVATIONS')
  if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('NEW_CTC_LENGTH')
  if ts[0]<row['start_ms'] or ts[-1]>=row['end_ms']:raise ValueError('NEW_TIMESTAMP_SPAN')
  pixel=row['decoded_segment_observation_sha256']
  if pixel in old_segments:raise ValueError('EXACT_OBSERVED_PIXEL_SEGMENT_DUPLICATE')
  new_segment_hashes.append(pixel);checks.append({'sample_id':row['sample_id'],'source_group':row['source_group'],'observations':len(x),'target_tokens':len(row['glosses']),'status':'PASS'})
 if len(new_segment_hashes)!=len(set(new_segment_hashes)):raise ValueError('NEW_SEGMENT_CONTENT_DUPLICATES')
 enriched=[{**r,'canonical_role':'eligible_train','training_pool':'NCSLGR2_LITERAL_MAIN_GLOSS','schema_version':SCHEMA,
            'source_catalog':'20261020_EAF_exact_basename_URL_list_resolution','signer_independent':False} for r in new]
 combined=old+enriched
 if len({r['sample_id'] for r in combined})!=len(combined):raise ValueError('COMBINED_ID_DUPLICATES')
 excluded=set(read(RUN/'ncslgr_interval_exclusions.json')['excluded_ids'])|set(read(OLD/'asllrp2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])|{'asllrp3_8d6cbe4ca1fc7986'}
 if excluded&{r['sample_id'] for r in combined}:raise ValueError('EXCLUDED_SAMPLE_REINTRODUCED')
 write(destination,combined)
 quality={'status':'PASS_VERIFIED_ELIGIBLE_CATALOG','prior_eligible':3025,'added_unique_eligible':96,'unique_eligible':len(combined),
  'eligible_by_label_pool':dict(Counter(r['training_pool'] for r in combined)),'manifest_sha256':sha(destination),'parent_manifest_sha256':sha(parent_path),
  'new_feature_checks':checks,'new_sources':2,'new_segment_pixel_hashes_unique':True,'parent_ID_source_group_raw_hash_overlap':False,
  'duplicate_limits':'Exact IDs/source groups/raw hashes and observed pixel segment hashes checked. No exhaustive perceptual re-encoding or semantic duplicate detector claimed.',
  'RIT_training_eligible':0,'stretch_target':5000,'shortfall':5000-len(combined),
  'current_fit':'Separate frozen1706TRAIN/413DEV ASLLRP3 partition; this3121eligible data catalog is not a single mixed TRAIN manifest.',
  'new_ncslgr_in_current_fit':False,'reserved_test_used':False,'all_prior_exclusions_preserved':True}
 write(RUN/'eligible_corpus/quality.json',quality)
 rows=[
  {'stage':'Prior_NCSLGR2','candidate_utterances':'previous_completed_evidence','annotation_media_joins':'previous_completed_evidence','verified_eligible':906,'metadata_excluded':'previous_completed_evidence','timing_blocked':0,'network_media_files_this_phase':0,'raw_media_files_this_phase':0,'actually_in_new_TRAIN':0,'actually_in_new_DEV':0,'note':'Frozen verified pool reused; no extraction repeated.'},
  {'stage':'Prior_ASLLRP3','candidate_utterances':2130,'annotation_media_joins':2126,'verified_eligible':2119,'metadata_excluded':10,'timing_blocked':1,'network_media_files_this_phase':0,'raw_media_files_this_phase':0,'actually_in_new_TRAIN':1706,'actually_in_new_DEV':413,'note':'4annotation rejects+6overlap exclusions+1frame mismatch. Original2119TRAIN manifest unchanged; new random-initialized fit has a distinct signer split.'},
  {'stage':'New_NCSLGR_URL_list_resolution','candidate_utterances':98,'annotation_media_joins':96,'verified_eligible':96,'metadata_excluded':2,'timing_blocked':0,'network_media_files_this_phase':2,'raw_media_files_this_phase':2,'actually_in_new_TRAIN':0,'actually_in_new_DEV':0,'note':'Two EAF-exact media URLs; both sides of one overlap pair excluded. Features from2new fully decoded source videos,96distinct genuine utterances.'},
  {'stage':'New_RIT_sentences','candidate_utterances':280,'annotation_media_joins':278,'verified_eligible':0,'metadata_excluded':4,'timing_blocked':276,'network_media_files_this_phase':1,'raw_media_files_this_phase':278,'actually_in_new_TRAIN':0,'actually_in_new_DEV':0,'note':'One434126167byte archive,278extracted clips.2initial annotation rejects+2overlap excludes. Every clip has60fps and2extra frames; no trimming or guessed offset.'},
  {'stage':'NEW_PHASE_TOTAL','candidate_utterances':378,'annotation_media_joins':374,'verified_eligible':96,'metadata_excluded':6,'timing_blocked':276,'network_media_files_this_phase':3,'raw_media_files_this_phase':280,'actually_in_new_TRAIN':1706,'actually_in_new_DEV':413,'note':'TRAIN/DEV cells describe the new fit using existing ASLLRP records, not newly acquired examples. File counts are not utterance counts.'},
  {'stage':'ELIGIBLE_CATALOG_TOTAL','candidate_utterances':'not_summed_across_different_historical_stages','annotation_media_joins':'see_parent_and_new_stage_counts','verified_eligible':len(combined),'metadata_excluded':'see_source_specific_rows','timing_blocked':'see_source_specific_rows','network_media_files_this_phase':3,'raw_media_files_this_phase':280,'actually_in_new_TRAIN':1706,'actually_in_new_DEV':413,'note':'1002NCSLGR2+2119ASLLRP3 in separate label pools;5000target shortfall1879. Eligibility does not authorize mixing targets or fitting DEV.'}]
 with (RUN/'DATA_PROGRESS.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({k:quality[k] for k in ('status','unique_eligible','added_unique_eligible','eligible_by_label_pool','manifest_sha256','shortfall')}),flush=True)

if __name__=='__main__':main()
