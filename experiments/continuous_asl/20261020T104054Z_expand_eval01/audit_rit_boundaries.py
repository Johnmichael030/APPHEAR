"""Global master-frame overlap audit, additive to the per-utterance CSV audit."""
from collections import defaultdict
import json
from core import RUN,read,write,sha

def main():
    destination=RUN/'rit2025_catalog/annotation_boundary_audit.v2.json'
    if destination.exists():print('BOUNDARY_AUDIT_EXISTS');return
    rows=read(RUN/'rit2025_catalog/plan.json')['rows'];groups=defaultdict(list)
    for r in rows:groups[r['source_group']].append(r)
    pairs=[];invalid=set()
    for group,items in groups.items():
        active=[]
        for row in sorted(items,key=lambda r:(r['utterance_start_frame_master'],r['utterance_end_frame_master'])):
            a,b=row['utterance_start_frame_master'],row['utterance_end_frame_master']
            active=[left for left in active if left['utterance_end_frame_master']>=a]
            for left in active:
                pairs.append({'master_video':group,'left_id':left['sample_id'],'right_id':row['sample_id'],
                              'left_frames':[left['utterance_start_frame_master'],left['utterance_end_frame_master']],
                              'right_frames':[a,b],'reason':'INCLUSIVE_MASTER_FRAME_UTTERANCE_OVERLAP_NO_TRIMMING'})
                invalid.update([left['sample_id'],row['sample_id']])
            active.append(row)
    report={'status':'PASS_NO_OVERLAP' if not pairs else 'PARTIAL_EXCLUDE_OVERLAPPING_UTTERANCES',
            'source_utterances':len(rows),'source_master_files':len(groups),'overlap_pairs':pairs,
            'reject_sample_ids':sorted(invalid),'nonoverlapping_annotation_candidates':len(rows)-len(invalid),
            'plan_sha256':sha(RUN/'rit2025_catalog/plan.json'),'coordinate_semantics':'Unchanged publisher inclusive master frame ranges; endpoints shared by two utterances are overlap and both are excluded.',
            'visual_boundary_accuracy_claim':False,'script_sha256':sha(__file__)}
    write(destination,report);print(json.dumps({k:v for k,v in report.items() if k not in ('overlap_pairs','reject_sample_ids')}),flush=True)

if __name__=='__main__':main()
