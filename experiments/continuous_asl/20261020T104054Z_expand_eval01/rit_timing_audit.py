"""Cheap new-source timing audit. Reject mismatches; never trim or shift frames."""
from collections import Counter
import cv2,json
from core import RUN,read,write

def main():
 output=RUN/'rit2025_catalog/container_timing_audit.json'
 if output.exists():print('RIT_TIMING_AUDIT_COMPLETE_NO_REPEAT');return
 rows=read(RUN/'rit2025_catalog/media_manifest.json');excluded=set(read(RUN/'rit2025_catalog/annotation_boundary_audit.v2.json')['reject_sample_ids'])
 cv2.setNumThreads(1);results=[]
 for row in rows:
  capture=cv2.VideoCapture(row['video_path']);fps=capture.get(cv2.CAP_PROP_FPS);frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
  expected=row['utterance_end_frame_master']-row['utterance_start_frame_master']+1
  results.append({'sample_id':row['sample_id'],'collection':row['collection'],'publisher_inclusive_span_frames':expected,'container_frames':frames,
    'frame_difference':frames-expected,'native_fps':fps,'excluded_overlap':row['sample_id'] in excluded,
    'status':'BLOCKED_PUBLISHER_CROP_ORIGIN_AND_RATE_UNVERIFIED' if frames!=expected or abs(fps-30000/1001)>.001 else 'PASS_HEADER_ONLY_NEEDS_FULL_DECODE'})
 report={'status':'BLOCKED_NEW_SOURCE_TIMING_CONTRACT','records':results,'joined_records':len(rows),'nonoverlap_records':sum(not r['excluded_overlap'] for r in results),
   'fps_counts':dict(Counter(str(r['native_fps']) for r in results)),'frame_difference_counts':dict(Counter(str(r['frame_difference']) for r in results)),
   'exact_span_matches':sum(r['frame_difference']==0 for r in results),'accepted_training_records':0,
   'full_video_decode_or_feature_extraction_repeated':False,'repair_performed':False,
   'missing_evidence':'Publisher documentation for RIT60fps master-frame units and exact cropped-utterance origin/padding. Across observed clips container counts differ from the unchanged inclusive publisher span. No guessed one-frame offset, trimming, or frame-rate scaling is allowed.'}
 write(output,report);print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
