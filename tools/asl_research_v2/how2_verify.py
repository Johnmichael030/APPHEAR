"""Verify complete archive integrity; extract only eight complete validation clips."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
from common import ROOT,PREVIOUS,run_root,sha,write
from archive_safety import inspect_archive,member_path
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
archive=run/'data/raw/how2sign/val_rgb_front_clips.zip';receipt=json.loads(archive.with_suffix('.zip.receipt.json').read_text())
if sha(archive)!=receipt['sha256']:raise ValueError('HOW2_ARCHIVE_IDENTITY')
members=inspect_archive(archive,3*1073741824,max_entries=10000)
annotations=run/'data/raw/how2sign/how2sign_original_val.csv'
with annotations.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
by_name={r['SENTENCE_NAME']:r for r in rows}
videos=sorted(r['path'] for r in members if Path(r['path']).suffix.lower() in ('.mp4','.mov','.avi'))
paired=[name for name in videos if Path(name).stem in by_name]
out=run/'data/extracted/how2sign_pilot';out.mkdir()
records=[];total=0;cv2.setNumThreads(2)
with zipfile.ZipFile(archive) as z:
    for name in paired[:8]:
        safe=member_path(name);info=z.getinfo(name);total+=info.file_size
        if total>67108864:raise ValueError('PILOT_EXTRACTION_BUDGET')
        target=out/Path(name).name
        with z.open(info) as inp,target.open('xb') as dest:shutil.copyfileobj(inp,dest,1048576)
        cap=cv2.VideoCapture(str(target));fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));decoded=0
        try:
            while True:
                ok,image=cap.read()
                if not ok:break
                decoded+=1
        finally:cap.release()
        if decoded<2 or not np.isfinite(fps) or fps<=0:raise ValueError('PILOT_DECODE')
        row=by_name[target.stem];duration=decoded/fps;expected=float(row['END'])-float(row['START'])
        records.append({'sample_id':target.stem,'source':'How2Sign','modality':'RGB_FRONT_SENTENCE_CLIP','label_type':'ENGLISH_TRANSLATION',
            'split':'official_val','signer':None,'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,
            'frames':decoded,'fps':fps,'width':width,'height':height,'duration':duration,'annotation_duration':expected,
            'duration_abs_difference':abs(duration-expected),'timing_status':'CONSISTENT' if abs(duration-expected)<=max(.1,2/fps) else 'MISMATCH_REVIEW_REQUIRED',
            'reference_english':row['SENTENCE'],'annotation_source':str(annotations),'source_video':row['VIDEO_ID'],
            'semantic_ASL_English_alignment':'NOT_MANUALLY_VERIFIED','gloss_available':False})
existing=[]
for file in sorted((PREVIOUS/'data/raw/how2sign').glob('*')):
    if file.suffix=='.csv':existing.append({'path':str(file),'bytes':file.stat().st_size,'sha256':sha(file),'annotation_type':'ENGLISH_TRANSLATION_REALIGNED','status':'PRESENT_AND_VERIFIED','newly_acquired':False})
inventory={'scope':'repository + D:/ASL_Citizen, ASL VIDEOS DATASETS, ASLdataset, derived, gpu_train_bundle/new and classmate data bundles; unrelated D: folders not exhaustively searched',
    'before':{'RGB_VIDEOS':'NOT_FOUND_IN_SEARCHED_ROOTS','RGB_SENTENCE_CLIPS':'NOT_FOUND_IN_SEARCHED_ROOTS','POSE_2D':'NOT_FOUND_IN_SEARCHED_ROOTS',
        'ENGLISH_TRANSLATION_ORIGINAL':'NOT_FOUND_IN_SEARCHED_ROOTS','ENGLISH_TRANSLATION_REALIGNED':existing,'ASL_GLOSS_SEQUENCE':'NOT_FOUND','GLOSS_WITH_TIMESTAMPS':'NOT_FOUND'},
    'after':{'RGB_SENTENCE_CLIPS':{'status':'PRESENT_AND_VERIFIED_PILOT','archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
        'archive_video_members':len(videos),'archive_crc':'ALL_MEMBERS_PASS','annotation_pairs':len(paired),'extracted_verified_clips':len(records),'extracted_bytes':total,
        'newly_acquired':True,'not_extracted_video_members':len(videos)-len(records)},
        'ENGLISH_TRANSLATION_ORIGINAL':{'status':'PRESENT_AND_VERIFIED','path':str(annotations),'bytes':annotations.stat().st_size,'rows':len(rows),'sha256':sha(annotations),'newly_acquired':True},
        'POSE_2D':'NOT_ACQUIRED','FULL_VIDEO_LEVEL_RGB':'NOT_ACQUIRED','ASL_GLOSS_SEQUENCE':'NOT_AVAILABLE_IN_INSPECTED_OFFICIAL_RELEASE','signer_metadata':'UNKNOWN'},
    'archive_expanded_bytes':sum(r['bytes'] for r in members),'archive_members':len(members),'usable_pilot':records,
    'research_permission':'CC BY-NC 4.0 + research-only project terms; attribution required; no commercial deployment clearance',
    'CTC_gloss_training':'BLOCKED_NO_AUTHENTIC_GLOSS','translation_training':'NOT_RUN_VALIDATION_ONLY_PILOT_NOT_A_TRAINING_PARTITION',
    'site_script_discrepancy':{'stale_script_id':'1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1','stale_status':'HTTP_404','current_official_page_id':'1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO'}}
write(run/'how2sign_inventory.json',inventory)
print(json.dumps({'archive_videos':len(videos),'paired_annotations':len(paired),'pilot_clips':len(records),'pilot_bytes':total,
    'timing_statuses':[r['timing_status'] for r in records],'archive_crc':'PASS'}))
